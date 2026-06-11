# ML Project 1 — From-Scratch Fundamentals

Linear regression and a 2-layer neural network, both implemented from scratch in NumPy. `sklearn` is used only for datasets and as a correctness oracle — none of the actual modelling, fitting, or gradients comes from a framework.

## Repo

- `day1_linear_regression.ipynb` — linear regression on `load_diabetes` via gradient descent and the normal equation. Compared against `sklearn.LinearRegression` for sanity.
- `day2_neural_network.ipynb` — 2-layer MLP on `load_digits`, hand-written forward and backward passes, trained with mini-batch SGD.

---

## Day 1 — Linear regression: deriving ∂J/∂θ

### Setup

We have `m` training examples, each with `n` features. Stack them into a design matrix `X` of shape `(m, n+1)`, where we prepend a column of 1s so the intercept `θ₀` has its own dummy feature. The targets are a vector `y` of shape `(m,)`. The parameters are a vector `θ` of shape `(n+1,)`.

The model:

    ŷ = X θ

### Cost function (MSE)

The residual on example `i` is `rᵢ = xᵢᵀθ − yᵢ`. The cost is half the mean squared residual:

    J(θ) = (1 / 2m) · Σᵢ (xᵢᵀθ − yᵢ)²
         = (1 / 2m) · (Xθ − y)ᵀ (Xθ − y)

Design decisions baked into this expression:

- **Squared error** rather than absolute error: differentiable everywhere, penalises outliers, and makes `J` a convex quadratic in θ — so the minimum is unique and reachable by gradient descent.
- **Mean (1/m)** rather than total sum: keeps the magnitude of the cost comparable across dataset sizes, so learning rates don't have to be re-tuned when `m` changes.
- **Half (1/2)**: cosmetic. The 2 that drops out of differentiating the square will cancel it, leaving a clean gradient.

### Deriving the gradient

#### Element-wise

Take the partial derivative with respect to a single component `θₖ`:

    ∂J/∂θₖ = (1/2m) · Σᵢ 2 · (xᵢᵀθ − yᵢ) · ∂/∂θₖ (xᵢᵀθ − yᵢ)
           = (1/m)  · Σᵢ     (xᵢᵀθ − yᵢ) · ∂/∂θₖ (xᵢᵀθ − yᵢ)

The inner derivative `∂(xᵢᵀθ − yᵢ)/∂θₖ` is just `xᵢₖ`: of all the terms in `xᵢᵀθ = Σⱼ xᵢⱼθⱼ`, only the one with `θₖ` survives, and `yᵢ` has no θ in it. So:

    ∂J/∂θₖ = (1/m) · Σᵢ (xᵢᵀθ − yᵢ) · xᵢₖ
           = (1/m) · Σᵢ rᵢ · xᵢₖ

In words: the k-th component of the gradient is the *average over training examples of (residual × k-th feature)*.

#### Vectorised

`Σᵢ xᵢₖ rᵢ` is the k-th entry of `Xᵀr`. Stacking all `n+1` partials gives:

    ∇J(θ) = (1/m) · Xᵀ(Xθ − y)

Shape check: `Xᵀ` is `(n+1, m)`, `(Xθ − y)` is `(m,)`, product is `(n+1,)` — same shape as θ. ✓

This is the line of code we'll actually run in NumPy. One matrix-vector product, no Python loop over examples.

### Update rule

`∇J` is the direction of steepest *increase* of `J`, so we step against it:

    θ ← θ − α · ∇J(θ) = θ − (α/m) · Xᵀ(Xθ − y)

`α` is the learning rate. Too large diverges; too small converges slowly. Because `J` is convex for linear regression, a small-enough `α` is guaranteed to reach the global minimum.

### The normal equation falls out for free

Convex `J` ⇒ the minimum is where the gradient vanishes:

    (1/m) · Xᵀ(Xθ − y) = 0
    XᵀX θ = Xᵀy
    θ     = (XᵀX)⁻¹ Xᵀy

This gives the exact optimal θ in one matrix inversion — no iteration. We implement it as a second method and use it as a check on the gradient-descent result, in addition to `sklearn.LinearRegression`.

### Why bother with gradient descent if the closed form exists?

- The normal equation costs `O((n+1)³)` to invert `XᵀX`. Fine for `n=10`, intractable for `n ≈ 10⁴`.
- More importantly: it only exists because the MSE loss is quadratic. The moment we move to logistic regression, neural networks, or anything else, there is no closed form. Gradient descent is the general-purpose tool — Day 2's neural net will need it.

---

## Day 2 — Neural network: deriving backprop

A 2-layer multilayer perceptron (MLP) for 10-class digit classification on `load_digits`.

### Architecture

    input X (m × 64)
       │  linear:  Z₁ = X W₁ + b₁          W₁ : (64, h)   b₁ : (h,)
       ▼
    pre-activations Z₁ (m × h)
       │  ReLU:    A₁ = max(0, Z₁)
       ▼
    activations A₁ (m × h)
       │  linear:  Z₂ = A₁ W₂ + b₂          W₂ : (h, 10)  b₂ : (10,)
       ▼
    pre-activations Z₂ (m × 10)
       │  softmax: A₂ = softmax(Z₂)         (row-wise)
       ▼
    probabilities A₂ (m × 10)
       │  cross-entropy against one-hot labels Y (m × 10)
       ▼
    scalar loss L

`m` is the batch size; `h` is the hidden width (we'll use 64).

### Why every linear layer reuses the Day 1 gradient pattern

On Day 1 we derived

    ∂J/∂θ = (1/m) · Xᵀ (Xθ − y)

For a multi-output linear layer `Z = X W + b`, the same chain rule gives

    ∂L/∂W = (1/m) · (input to this layer)ᵀ · (gradient arriving at this layer)
    ∂L/∂b = (1/m) · sum over batch of (gradient arriving at this layer)

Backprop is this pattern applied layer-by-layer, with activation derivatives stitched between layers via the chain rule.

### Softmax + cross-entropy: the magic simplification

Softmax (with the standard subtract-max trick for numerical stability):

    softmax(z)ₖ = exp(zₖ − max(z)) / Σⱼ exp(zⱼ − max(z))

Cross-entropy against one-hot labels Y:

    L = − (1/m) · Σᵢ Σₖ Yᵢₖ · log(A₂)ᵢₖ

The point of pairing these two operations: their gradient *combined* collapses to a clean form. Derivation for a single example with one-hot label `y` and pre-activations `z`:

    softmax(z)ₖ = exp(zₖ) / S                 where S = Σⱼ exp(zⱼ)
    L           = − Σₖ yₖ · log(softmax(z)ₖ)
                = − Σₖ yₖ · (zₖ − log S)
                = log S − Σₖ yₖ zₖ            (using Σₖ yₖ = 1 for one-hot)

Differentiate w.r.t. a specific `zₖ`:

    ∂L/∂zₖ = ∂(log S)/∂zₖ − yₖ
           = (1/S) · exp(zₖ) − yₖ
           = softmax(z)ₖ − yₖ

Two ugly chain-rule terms cancelled. Across the whole batch in matrix form:

    ∂L/∂Z₂ = (1/m) · (A₂ − Y)

Same shape as the linear-regression residual: *predicted minus actual*.

### Full backward pass

Walking backward through the forward graph:

**Output layer (W₂, b₂):**

    dZ₂ = (1/m) · (A₂ − Y)                    (m, 10)
    dW₂ = A₁ᵀ · dZ₂                           (h, 10)
    db₂ = column-sum of dZ₂                   (10,)

The `1/m` is already baked into `dZ₂`, so we do not divide again.

**Through the ReLU:**

    dA₁ = dZ₂ · W₂ᵀ                            (m, h)        ← chain rule across the linear layer
    dZ₁ = dA₁ ⊙ 1[Z₁ > 0]                       (m, h)        ← ReLU derivative: 1 where Z₁ > 0, else 0

The `⊙` is element-wise product. Intuition: ReLU clamps negative pre-activations to zero in the forward pass; in the backward pass it kills the gradient for those same units. A "dead" neuron (always-negative pre-activation) receives no gradient and never updates — this is the "dying ReLU" failure mode you'll hear about.

**Input layer (W₁, b₁):**

    dW₁ = Xᵀ · dZ₁                             (64, h)
    db₁ = column-sum of dZ₁                    (h,)

**Parameter update (mini-batch SGD):**

    W₁ ← W₁ − α · dW₁
    b₁ ← b₁ − α · db₁
    W₂ ← W₂ − α · dW₂
    b₂ ← b₂ − α · db₂

That's the entire algorithm. Six gradient expressions and four updates. Every modern deep-learning framework, underneath the syntactic sugar of autograd, is doing exactly this — just generalized to arbitrary computational graphs.
