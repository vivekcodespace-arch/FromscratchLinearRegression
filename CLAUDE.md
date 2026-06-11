# ML Project 1 — From-Scratch Fundamentals

A 2-day project to learn the math and implementation of linear regression and a neural network from scratch in NumPy, with sklearn used only as (1) a source of datasets and (2) a correctness oracle.

## Mode of work

**Teaching mode.** The user is learning. Do not generate the whole notebook in one go.

- Derive the math first, in plain text, before writing any code.
- Implement piece by piece. After each piece, explain *why* it exists — what would break if it were missing or wrong.
- Pause and confirm understanding between major concepts (hypothesis → cost → gradient → update rule → loop).
- Keep code vectorized. If a `for` loop appears over training examples, stop and rewrite — that itself is a teaching moment.
- The README derivation (∂MSE/∂θ for Day 1, backprop for Day 2) is the *primary* deliverable. It's what separates this from a copy-pasted tutorial.

## Constraints

- **Allowed:** NumPy, Matplotlib, `sklearn.datasets.*` (load_diabetes, fetch_california_housing, load_digits), `sklearn.LinearRegression` purely as a correctness check, `train_test_split`, `StandardScaler` for preprocessing.
- **Not allowed for the model itself:** any framework that does the gradient or fitting for us — no sklearn estimators, no PyTorch, no TensorFlow, no JAX, no autograd.

## Day 1 — Linear regression (NumPy)

Dataset: `sklearn.datasets.load_diabetes` (small, clean, no download).

Implement, vectorized:
1. Hypothesis  `ŷ = Xθ`
2. Cost  `J(θ) = (1/2m) Σ (ŷ - y)²`
3. Gradient  `∇J(θ) = (1/m) Xᵀ(ŷ - y)`
4. Gradient-descent loop with cost-per-iteration tracking
5. Closed-form normal equation  `θ = (XᵀX)⁻¹ Xᵀy`

Validate: both methods should match `sklearn.LinearRegression` coefficients to ~4 decimal places.

Plots: cost vs iterations (should be monotonically decreasing), predicted vs actual (should hug the diagonal).

Deliverable: `day1_linear_regression.ipynb` + a `README.md` section deriving `∂J/∂θ`.

## Day 2 — Neural network (NumPy)

Dataset: `sklearn.datasets.load_digits` (8×8 digit images, ~1,800 samples).

Architecture: 2-layer MLP — `input (64) → hidden (e.g. 64, ReLU) → output (10, softmax)`.

Implement:
1. Forward pass: `Z1 = XW1 + b1`, `A1 = ReLU(Z1)`, `Z2 = A1 W2 + b2`, `A2 = softmax(Z2)`
2. Cross-entropy loss
3. Backward pass: derive `dZ2`, `dW2`, `db2`, `dA1`, `dZ1`, `dW1`, `db1` by hand
4. Mini-batch SGD training loop

Target: ≥90% test accuracy on digits.

Deliverable: `day2_neural_network.ipynb` + a `README.md` section with the full backprop derivation for at least one layer.

## What "done" looks like

- Both notebooks run top-to-bottom with no errors.
- Day 1 numbers match sklearn.
- Day 2 hits ≥90% test accuracy.
- README contains both math derivations in the user's own words (not copy-pasted).
- Repo pushed to GitHub by end of Day 2.
