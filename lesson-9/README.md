# Lesson 9 — PyTorch Basics

## Objective

Learn the PyTorch building blocks — tensors, autograd, `nn.Module`, and the training loop — that every neural network surrogate in Lessons 10 and 11 is built from.

---

## Concepts

- **Tensor** — PyTorch's core data structure; like a NumPy array but with GPU support and automatic differentiation
- **`requires_grad`** — flag that tells PyTorch to track operations on a tensor for gradient computation
- **Autograd** — PyTorch's automatic differentiation engine; computes `∂loss/∂params` without manual calculus
- **`loss.backward()`** — propagates gradients back through the computation graph
- **`nn.Module`** — base class for all neural networks; holds parameters and defines `forward()`
- **`nn.Sequential`** — stacks layers in order; the simplest way to define an MLP
- **`nn.Linear(in, out)`** — a fully-connected layer: y = Wx + b
- **`nn.ReLU()`** — rectified linear unit activation: f(x) = max(0, x)
- **Optimizer** — `torch.optim.Adam` updates parameters to reduce the loss
- **Training loop** — `zero_grad → forward → loss → backward → step` — repeated each epoch
- **Standardization** — normalize inputs and outputs to zero mean, unit variance before training; critical for stable NN training

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `notebook.ipynb` | Interactive — modify the MLP, change hyperparameters, watch training |
| `main.py` | Standalone script — saves all plots to `output/` |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-9/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-9/notebook.ipynb
```

---

## What You Will Build

### Part 1 — Tensors and NumPy interop
Create tensors, convert to/from NumPy, perform basic operations. Show that `requires_grad=True` enables gradient tracking.

### Part 2 — Autograd
Compute the gradient of the Forrester function at every point using `loss.backward()`. Compare to finite differences — they should match exactly.

### Part 3 — MLP architecture and NN prior
Define the two-layer MLP that Lessons 10 and 11 will build on. Plot 5 randomly-initialised networks — this is the NN equivalent of the GP prior from Lesson 5.

### Part 4 — The training loop
Fit the MLP to 20 Forrester points. Plot the loss curve and the final fit against the true function.

### Part 5 — Standardization and train/val split
Show why normalising inputs and outputs to zero mean and unit variance is necessary. Split data into train/val, plot both loss curves, and identify where the model starts to overfit.

### Part 6 — Save and load a model
Save the trained model with `torch.save`, load it back with `torch.load`, and verify that predictions are identical before and after loading.

---

## Exercises

1. In Part 3, add a third hidden layer (`nn.Linear(64, 64)` + `nn.ReLU()`). Does the network fit Forrester better or worse with the same training budget?
2. In Part 4, change the optimizer from `Adam` to `SGD(lr=0.01)`. How many more epochs does SGD need to reach the same loss?
3. In Part 5, remove the output standardization. What happens to the training loss curve? Does the model still fit the function?

---

## What's Next

**Lesson 10** — MC Dropout surrogate: add `nn.Dropout(0.1)` to this MLP, keep dropout ON at inference, and run 50 forward passes to get a mean and uncertainty estimate.
