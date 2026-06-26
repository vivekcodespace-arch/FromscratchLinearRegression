# From-Scratch ML Fundamentals: Linear Regression & Neural Network in NumPy

**Author:** Vivek Sharma  
**Date:** June 2026  
**GitHub:** [github.com/vivekcodespace-arch](https://github.com/vivekcodespace-arch)

---

## Abstract

This project investigates whether first-principles implementations of gradient descent and backpropagation, written entirely in NumPy, can recover results matching analytical solutions and industry-standard libraries to machine precision. We implement linear regression via both gradient descent and the closed-form normal equation, verify both against `sklearn.LinearRegression` to ~10⁻¹² precision, and analyse failure modes on ill-conditioned data. We then build a 2-layer MLP from scratch with hand-derived backpropagation, validate gradients using a central-difference numerical check (~10⁻⁷ relative error), and train with mini-batch SGD + He initialisation — achieving 95.83% test accuracy on the digits dataset.

---

## 1. Motivation

Most ML practitioners use libraries like PyTorch or scikit-learn as black boxes. This project asks: *what happens when you remove the black box entirely?* The goal was not to build a production system, but to deeply understand the mathematical mechanics — where the formulas come from, why they work, and where they break down. This kind of first-principles understanding is especially important for debugging real models and contributing to ML research.

---

## 2. Part 1 — Linear Regression from Scratch

### 2.1 Problem Setup

Given a dataset of input-output pairs $(X, y)$, we want to find parameters $\theta$ that minimise the Mean Squared Error (MSE):

$$\text{MSE}(\theta) = \frac{1}{n} \|X\theta - y\|^2$$

There are two ways to solve this:
1. **Gradient Descent (GD):** Iteratively update $\theta$ in the direction of steepest descent
2. **Normal Equation:** Compute the closed-form solution directly: $\theta = (X^TX)^{-1}X^Ty$

### 2.2 Deriving the Gradient

The key derivative we need is $\partial\text{MSE}/\partial\theta$. Deriving it from first principles (both element-wise and in vectorised matrix form):

$$\frac{\partial \text{MSE}}{\partial \theta} = \frac{2}{n} X^T(X\theta - y)$$

This derivation was done manually — not looked up — to ensure genuine understanding of why the update rule takes this form.

### 2.3 Implementation

Both methods were implemented in vectorised NumPy (no Python loops over data points):

- **Gradient Descent:** Standard batch GD with a fixed learning rate, run until convergence
- **Normal Equation:** Direct matrix inversion using `np.linalg.solve` (numerically stable vs `np.linalg.inv`)

### 2.4 Verification

Both implementations were verified against `sklearn.LinearRegression` on the same dataset:

| Method | Max parameter error vs sklearn |
|--------|-------------------------------|
| Normal Equation | ~10⁻¹² (machine precision) |
| Gradient Descent | ~10⁻⁸ (converged, not exact) |

**Key finding:** GD converges slowly on ill-conditioned problems. On a dataset with condition number ~5×10⁴, GD required significantly more iterations to approach the normal equation solution — demonstrating the practical importance of feature scaling and preconditioning in real ML pipelines.

---

## 3. Part 2 — Neural Network from Scratch

### 3.1 Architecture

A 2-layer MLP trained on the `load_digits` dataset (8×8 pixel handwritten digit images, 1797 samples, 10 classes):

```
Input (64) → Dense → ReLU → Dense → Softmax → Output (10)
```

### 3.2 Forward Pass

The forward pass computes activations layer by layer:

$$z^{(1)} = W^{(1)}x + b^{(1)}, \quad a^{(1)} = \text{ReLU}(z^{(1)})$$
$$z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}, \quad \hat{y} = \text{Softmax}(z^{(2)})$$

Loss is cross-entropy: $\mathcal{L} = -\sum_i y_i \log(\hat{y}_i)$

### 3.3 Backpropagation — Derived by Hand

Rather than relying on automatic differentiation, all gradients were derived manually using the chain rule. The key gradients are:

$$\frac{\partial \mathcal{L}}{\partial z^{(2)}} = \hat{y} - y \quad \text{(Softmax + Cross-Entropy combined)}$$

$$\frac{\partial \mathcal{L}}{\partial W^{(2)}} = \frac{\partial \mathcal{L}}{\partial z^{(2)}} \cdot (a^{(1)})^T$$

$$\frac{\partial \mathcal{L}}{\partial a^{(1)}} = (W^{(2)})^T \cdot \frac{\partial \mathcal{L}}{\partial z^{(2)}}$$

$$\frac{\partial \mathcal{L}}{\partial z^{(1)}} = \frac{\partial \mathcal{L}}{\partial a^{(1)}} \odot \mathbf{1}[z^{(1)} > 0] \quad \text{(ReLU derivative)}$$

### 3.4 Gradient Verification

Before trusting the backprop implementation, gradients were verified numerically using the **central-difference approximation**:

$$\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta_i + \epsilon) - \mathcal{L}(\theta_i - \epsilon)}{2\epsilon}$$

The relative error between analytical and numerical gradients was ~10⁻⁷ — confirming correctness of the implementation.

### 3.5 Training Details & Results

| Setting | Value |
|---------|-------|
| Optimiser | Mini-batch SGD |
| Initialisation | He initialisation |
| Batch size | 32 |
| Epochs | 20 |
| Test accuracy | **95.83%** |

**Key finding:** With random initialisation, the network failed to reach 90% accuracy within 20 epochs. With He initialisation, the 90% target was reached by epoch 3. This demonstrates that **weight initialisation dominates early convergence behaviour** — a result with direct implications for training deep networks.

---

## 4. Summary of Findings

| Experiment | Finding |
|------------|---------|
| GD vs Normal Equation | Both recover optimal parameters; GD degrades on ill-conditioned data |
| Gradient check | Backprop correct to ~10⁻⁷ relative error |
| He vs random init | He init reaches 90% accuracy 6× faster |
| Final test accuracy | 95.83% with mini-batch SGD + He init |

---

## 5. What I Learned

- The normal equation is exact but breaks on ill-conditioned matrices — explaining why iterative optimisers dominate in practice
- The Softmax + Cross-Entropy gradient simplifies elegantly to $\hat{y} - y$, which only becomes obvious when you derive it yourself
- Gradient checking is an essential debugging tool — it caught two subtle bugs in the ReLU backward pass before the final implementation
- He initialisation is not just a heuristic; it directly controls the variance of activations at initialisation, preventing vanishing/exploding gradients

---

## 6. References

1. Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.
2. Bishop, C. (2006). *Pattern Recognition and Machine Learning*. Springer.
3. scikit-learn documentation: [scikit-learn.org](https://scikit-learn.org)