import torch as th
import torch.nn as nn
from captum._utils.common import _run_forward
from typing import Callable, Union
from tint.models import Net
import torch.nn.functional as F
from torch.special import gammaln  # For computing log Beta function

def beta_kl_loss(mask, a=0.5, b=0.5, eps=1e-8):
    """
    Computes a KL divergence loss for the mask, encouraging it to match a Beta(a,b) distribution.
    This loss is defined as:
        L_KL = log B(a,b) - (a-1)*log(mask) - (b-1)*log(1-mask)
    where B(a,b) is the Beta function.
    """
    mask = mask.clamp(eps, 1 - eps)
    logB = gammaln(th.tensor(a)) + gammaln(th.tensor(b)) - gammaln(th.tensor(a+b))
    loss = logB - (a - 1) * th.log(mask) - (b - 1) * th.log(1 - mask)
    return loss.mean()


class MCExtremalMaskNN(nn.Module):
    """
    MC Extremal Mask NN model.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.
        model (nn.Module): A model used to recreate the original
            predictions, in addition to the mask. Default to ``None``
        batch_size (int): Batch size of the model. Default to 32
    """

    def __init__(
        self,
        forward_func: Callable,
        model: nn.Module = None,
        batch_size: int = 32,
        size_reg_factor: float = 0.5,
        time_reg_factor: float = 5.0,
        dropout_rate: float = 0.5,
        sigma: float = 0.001,
        # softdtw_reg_factor: float = 1.0

    ) -> None:
        super().__init__()
        object.__setattr__(self, "forward_func", forward_func)
        self.model = model
        self.batch_size = batch_size
        self.size_reg_factor = size_reg_factor
        self.time_reg_factor = time_reg_factor
        self.dropout_rate = dropout_rate
        self.sigma = sigma
        # self.softdtw_reg_factor = softdtw_reg_factor


        self.input_size = None
        self.register_parameter("mask", None)

        # Instantiate the SoftDTW loss function.
        # self.softdtw_loss_fn = SoftDTWLossPyTorch(gamma=0.1)

        # # These will store the last forward inputs for regularization purposes.
        # self.last_original = None
        # self.last_perturbed = None

    def init(self, input_size: tuple, batch_size: int = 32) -> None:
        self.input_size = input_size
        self.batch_size = batch_size
        self.mask = nn.Parameter(th.Tensor(*input_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.mask.data.fill_(0.5) ## old-
        # with th.no_grad():
        #     self.mask.data.copy_(th.randn_like(self.mask) * self.sigma + 0.5)


    def forward(
        self,
        x: th.Tensor,
        batch_idx: int,
        baselines: th.Tensor,
        target: th.Tensor,
        *additional_forward_args,
    ) -> (th.Tensor, th.Tensor):
        mask = self.mask

        # Subset sample to current batch
        mask = mask[self.batch_size * batch_idx : self.batch_size * (batch_idx + 1)]

        # Clamp the mask
        mask = mask.clamp(0, 1)

        # If model is provided, use it as the baselines
        if self.model is not None:
            baselines = self.model(x - baselines)

        # Mask data according to samples
        mask = mask[:, : x.shape[1], ...]
        x1 = x * mask + baselines * (1.0 - mask)
        x2 = x * (1.0 - mask) + baselines * mask


        
        # # Store original and one perturbed output for softDTW regularization
        # self.last_original = x.detach()

        # self.last_perturbed = x1.detach()  # choose x1, or average x1 and x2 if needed


        # Return f(perturbed x)
        out1 = _run_forward(forward_func=self.forward_func, inputs=x1,
                target=target, additional_forward_args=additional_forward_args)

        out2 = _run_forward(forward_func=self.forward_func, inputs=x2,
                target=target, additional_forward_args=additional_forward_args)



        return out1, out2


    def regularization(self, loss: th.Tensor) -> th.Tensor:
        # Size regularization
        mask_sorted = self.mask.reshape(len(self.mask), -1).sort().values
        size_reg = ((mask_sorted - 0.5) ** 2).mean()

        # Temporal continuity regularization
        time_reg = 0.0
        if self.time_reg_factor > 0:
            if self.mask.shape[1] > 1:  # Ensure there is a temporal dimension
                # time_reg = th.abs(self.mask[:, 1:, :] - self.mask[:, :-1, :]).mean()
                time_reg = ((self.mask[:, 1:, :] - self.mask[:, :-1, :]) ** 2).mean()

            else:
                time_reg = 0.0  # No temporal continuity regularization if time dimension is 1


        values = {
            "size_loss": self.size_reg_factor*round(size_reg.detach().cpu().item(), 5),
            "time_loss": self.time_reg_factor*round(time_reg.detach().cpu().item(), 5),
        }

        # self.log_dict(values, prog_bar=True)
        # print(values)


        return (loss + self.size_reg_factor * size_reg + self.time_reg_factor * time_reg,values)

    def representation(self, apply_dropout: bool = False) -> th.Tensor:
        rep = self.mask
        if apply_dropout:
            # Apply dropout to simulate Monte Carlo sampling.
            rep = F.dropout(rep, p=self.dropout_rate, training=True)
        return rep.detach().cpu().clamp(0, 1)




class MCExtremalMaskNet(Net):
    """
    MCExtremal mask model as a Pytorch Lightning model.
    ExtremalMaskNet wraps ExtremalMaskNN into a Lightning Module by subclassing your base Net.
    It combines the original prediction loss with an uncertainty minimization loss
    (computed via Monte Carlo dropout over the mask).
    
    Total loss is defined as:
      L_total = alpha * L_original + beta * L_uncertainty + regularization.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.
        preservation_mode (bool): If ``True``, uses the method in
            preservation mode. Otherwise, uses the deletion mode.
            Default to ``True``
        model (nn.Module): A model used to recreate the original
            predictions, in addition to the mask. Default to ``None``
        batch_size (int): Batch size of the model. Default to 32
        lambda_1 (float): Weighting for the mask loss. Default to 1.
        lambda_2 (float): Weighting for the model output loss. Default to 1.
        loss (str, callable): Which loss to use. Default to ``'mse'``
        optim (str): Which optimizer to use. Default to ``'adam'``
        lr (float): Learning rate. Default to 1e-3
        lr_scheduler (dict, str): Learning rate scheduler. Default to ``None``
        lr_scheduler_args (dict): Additional args for the scheduler. Default to ``None``
        l2 (float): L2 regularisation. Default to 0.0
    """

    def __init__(
        self,
        forward_func: Callable,
        preservation_mode: bool = True,
        model: nn.Module = None,
        batch_size: int = 32,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        loss: Union[str, Callable] = "cross_entropy",
        optim: str = "adam",
        lr: float = 0.0005,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
        alpha: float = 1.0,   # weight for prediction and mask loss
        beta: float  = 1.0,   # weight for entropy loss
        delta: float = 10.0,  # weight for uncertainty minimization loss
        num_mc_samples: int = 100,
        dropout_rate: float = 0.1,
        sigma: float = 0.1
    ):
        mask = MCExtremalMaskNN(
            forward_func=forward_func,
            model=model,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            sigma=sigma)

        super().__init__(
            layers=mask,
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )

        self.preservation_mode = preservation_mode
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.num_mc_samples = num_mc_samples

    def forward(self, *args, **kwargs) -> th.Tensor:
        return self.net(*args, **kwargs)

    def step(self, batch, batch_idx, stage):
        x, y, baselines, target, *additional_forward_args = batch

        if additional_forward_args == [None]:
            additional_forward_args = None

        if additional_forward_args is None:
            y_hat1, y_hat2 = self(x.float(), batch_idx, baselines, target)
        else:
            y_hat1, y_hat2 = self(
                x.float(),
                batch_idx,
                baselines,
                target,
                *additional_forward_args,
            )

        y_target1 = _run_forward(
            forward_func=self.net.forward_func,
            inputs=y,
            target=target,
            additional_forward_args=tuple(additional_forward_args)
            if additional_forward_args is not None
            else None,
        )
        y_target2 = _run_forward(
            forward_func=self.net.forward_func,
            inputs=th.zeros_like(y) + baselines,
            target=target,
            additional_forward_args=tuple(additional_forward_args)
            if additional_forward_args is not None
            else None,
        )

        if self.preservation_mode:
            mask_ = self.lambda_1 * self.net.mask.abs()
        else:
            mask_ = self.lambda_1 * (1.0 - self.net.mask).abs()

        if self.net.model is not None:
            mask_ = mask_[self.net.batch_size * batch_idx : self.net.batch_size * (batch_idx + 1)]
            mask_ += self.lambda_2 * self.net.model(x - baselines).abs()
        
        # loss = mask_.mean()

        if self.preservation_mode:
            pred_loss = self.loss(y_hat1, y_target1)
        else:
            pred_loss = self.loss(y_hat2, y_target2)

        base_loss = mask_.mean() + pred_loss


        # Monte Carlo dropout sampling for uncertainty.
        mc_masks = []
        # Activate dropout in the mask (set mask network to train mode).
        self.net.train()
        for _ in range(self.num_mc_samples):
            mc_masks.append(self.net.representation(apply_dropout=True))
        mc_masks = th.stack(mc_masks, dim=0)
        mask_variance = mc_masks.var(dim=0).mean()

        bias_weight = 1.0  # less penalty on (1-m) term
        entropy_loss = - (self.net.mask * th.log(self.net.mask + 1e-8) +
                              bias_weight * (1 - self.net.mask) * th.log(1 - self.net.mask + 1e-8)).mean()

        info_loss = - th.log(1 - self.net.mask + 1e-8).mean()

        ## define total losses
        total_loss = self.alpha * base_loss + self.beta*entropy_loss + self.delta*mask_variance

        total_loss, sizetime_loss = self.net.regularization(total_loss)

        values={"alpha":self.alpha,
                "base_loss": self.alpha*round(base_loss.detach().cpu().item(), 5),
                "beta":self.beta,
                "entropy_loss": self.beta*round(entropy_loss.detach().cpu().item(), 5),
                "delta":self.delta,
                "mask_variance_loss": self.delta*round(mask_variance.detach().cpu().item(), 5),
                "total_loss": round(total_loss.detach().cpu().item(), 5),

        }

        self.log_dict(values, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        params = [{"params": self.net.mask}]

        if self.net.model is not None:
            params += [{"params": self.net.model.parameters()}]

        if self._optim == "adam":
            optim = th.optim.Adam(
                params=params,
                lr=self.lr,
                weight_decay=self.l2,
            )
        elif self._optim == "sgd":
            optim = th.optim.SGD(
                params=params,
                lr=self.lr,
                weight_decay=self.l2,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise NotImplementedError

        lr_scheduler = self._lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler = lr_scheduler.copy()
            lr_scheduler["scheduler"] = lr_scheduler["scheduler"](
                optim, **self._lr_scheduler_args
            )
            return {"optimizer": optim, "lr_scheduler": lr_scheduler}

        return {"optimizer": optim}
