## Monte Carlo ExtremalMask: Uncertainty-aware Time-series Model Interpretability for Critical Care Applications

This repository provides an implementation of **Monte Carlo Extremal Mask (MCExtremalMask)**, a model-agnostic attribution method for time series data. The method estimates attribution scores and their uncertainty using Monte Carlo Dropout via multiple perturbation-based passes through the model.

---

## 🚀 Usage

### 1. Add the repo to your Python path

```python
import sys
sys.path.insert(0, "/path/to/mcem")
```
### 2. Import and initialize the explainer

```python
from mcem.attr import MCExtremalMask

explainer = MCExtremalMask(model)  # model should be a PyTorch model
```

### 3. Generate attributions with uncertainty
```python
all_mask_mean = []
all_mask_variance = []

for batch in dataloader:
    attr = explainer.attribute(batch, return_uncertainty=True)

    if isinstance(attr, tuple) and len(attr) == 2:
        mask_mean, mask_variance = attr
        all_mask_mean.append(mask_mean.cpu())
        all_mask_variance.append(mask_variance.cpu())
    else:
        all_mask_mean.append(attr.cpu())

all_mask_mean_tensor = torch.stack(all_mask_mean)

if all_mask_variance:
    all_mask_variance_tensor = torch.stack(all_mask_variance)
```

## 🚀 Requirements
```bash
pip install -r requirements.txt
```
