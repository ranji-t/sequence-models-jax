# sequence-models-jax

GRU implemented from scratch in pure functional JAX — no Flax, no Haiku, no parameter registries. All weights are plain JAX arrays stored in a `NamedTuple`. Sequences unroll via `lax.scan`. Batches parallelize via `vmap`. Every computation is explicit.

The repo is a progression: notebooks 01–06 build RNN, LSTM, and early GRU variants using Python `for` loops and `tqdm` for training; notebooks 07 replace the outer loops with a fully compiled XLA training stack. The three final notebooks are what matter.

---

## Notebooks

### `07-GRU-Airline-ARegr-Batch.py` — Univariate Regression, Airlines Dataset

Classic airline passenger benchmark: 144 monthly observations from 1949–1960, a univariate time series with compound seasonality and exponential trend. The model is configured as `input_size=1, hidden_size=16, output_size=1` with a sliding window of 12 timesteps and `batch_size=5`. Training runs for 100 epochs with the entire loop — epoch permutation, batch iteration, and parameter updates — compiled into a single `jax.lax.scan`. Evaluation is in-sample; the dataset is small enough that the model is assessed on its ability to track trend and seasonal amplitude, not generalization.

### `07-GRU-Iron-Regr-Batch.py` — Multivariate Regression, Korean Steel Industry

Target: `Usage_kWh` (log-transformed), a power consumption signal recorded at 15-minute intervals over ~35,000 timesteps. Input dimensionality is 12 after feature engineering. The model uses `input_size=12, hidden_size=16, output_size=1`, `timesteps=12`, `batch_size=5`. With 35k samples and `batch_size=5`, one epoch passes through ~7,000 batches inside the inner `lax.scan`. Training runs 100 epochs. The signal has sharp on/off regime transitions typical of industrial load schedules. Evaluation is in-sample.

### `07-GRU-Iron-Clf-Batch.py` — 3-Class Classification, Korean Steel Industry

Same dataset, different target: `Load_Type`, a 3-class label (Light Load = 0, Medium Load = 1, Maximum Load = 2), one-hot encoded. Input dimensionality is 12. Model: `input_size=12, hidden_size=16, output_size=3`, `timesteps=12`, `batch_size=256`. Training runs 50 epochs. The dataset is split temporally at the 75th percentile — the first 75% of windowed samples form the training set, the last 25% are held out for evaluation. No shuffling across the boundary. The loss is categorical cross-entropy applied to raw logits via `log_softmax`. Test-set accuracy is 91%.

---

## Feature Engineering

Both Korean steel notebooks share the same preprocessing pipeline. Raw features include lagging/leading reactive power, power factor, CO2 emissions, and NSM (number of seconds from midnight). Transformations applied before modeling:

- **Log transforms** on `Usage_kWh` and `Lagging_Current_Reactive_Power_kVarh` to compress right-skewed distributions
- **Power factor inversion**: `1 - PF/100`, converting from percentage efficiency to a complement
- **Cyclic time encoding**: sin/cos pairs for weekday (period 7), month (period 12), and time-of-day in minutes (period 1440) — six additional features encoding temporal position without ordinal assumptions
- **Scaling**: `RobustScaler` (median/IQR) on `NSM` and `Usage_kWh`; `RobustScaler` without centering using 20–80th percentile range on reactive power columns; `MaxAbsScaler` on CO2

---

## GRU Cell

The cell is implemented as three composable functions:

```
r_t = σ(X Wxr + h_{t-1} Whr + br)          # reset gate
z_t = σ(X Wxz + h_{t-1} Whz + bz)          # update gate
h̃_t = tanh(X Wxh + (r_t ⊙ h_{t-1}) Whh + bh)  # candidate state
h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t    # hidden state update
```

Each matrix multiplication is expressed as `jnp.einsum` rather than `@` or `jnp.dot` to keep index contractions readable. The output projection is a single linear layer: `y_t = h_T Wy + by`, where `h_T` is the final hidden state after scanning all 12 timesteps.

---

## Design Decisions

**Weights outside the carry.** The `lax.scan` carry holds only the hidden state `h_t` — a vector of shape `(hidden_size,)`. Weights are passed as a closed-over argument to the scan function, not as part of the carry. This matches JAX's functional model: `jax.value_and_grad` differentiates with respect to `ModelWeights` (the first argument to `forward`), not with respect to anything in the scan state. Putting weights in the carry would make them part of the scanned sequence and complicate gradient computation.

**`ModelWeights` as a `NamedTuple`.** All 11 weight tensors and biases (`Wxr, Whr, br, Wxz, Whz, bz, Wxh, Whh, bh, Wy, by`) are stored in a single `NamedTuple`. JAX treats `NamedTuple`s as pytrees natively, so `jax.value_and_grad`, `optax.apply_updates`, and `optimizer.init` all operate on the full weight tree without any registration boilerplate.

**Orthogonal initialization on hidden-to-hidden weights.** `Whr`, `Whz`, and `Whh` are initialized as orthogonal matrices via the economy SVD: `U, *_ = jnp.linalg.svd(W); return U`. Orthogonal matrices have unit singular values, which means the hidden-to-hidden transformation neither amplifies nor attenuates the gradient norm during backpropagation through time. Input-to-hidden matrices (`Wxr`, `Wxz`, `Wxh`) use scaled Glorot normal initialization: `N(0, 2/(fan_in + fan_out)) × 0.1`. All biases initialize to zero.

**`vmap` for batch parallelism, `lax.scan` for sequential state.** These two primitives address different axes. `lax.scan` is the right tool for sequential dependencies — each GRU step depends on the previous hidden state, so it cannot be parallelized over time. `vmap` is the right tool for independent batch instances — each sequence in a batch shares weights but has an independent hidden state trajectory, so they can be vectorized. The composition is: `vmap(gru_predict)` maps over batch dimension; inside `gru_predict`, `lax.scan` steps through the 12-timestep window.

**Nested `lax.scan` for the full training loop.** The final 07 notebooks compile the entire training procedure into XLA. The structure has three levels: an outer `lax.scan` iterates over epochs; at each epoch, `epoch_train_step` (a `jax.jit`-compiled function) permutes the dataset using an evolving PRNG key, assembles batches, then runs an inner `lax.scan` over all batches in the epoch. The PRNG key is threaded through the epoch state so each epoch shuffles differently without breaking XLA's static control flow requirement. The consequence is that no Python interpreter overhead occurs during training — the entire computation graph is lowered once and executed on the accelerator. Earlier notebooks (01–06) used Python `for` loops with `tqdm`; this approach is appropriate when dataset size makes materializing all intermediate scan outputs across epochs a memory concern.

**Temporal train/test split.** The classification notebook cuts the dataset at index `3 * (N // 4)`. Because the data is ordered chronologically, this guarantees the model never sees future labels during training. Random splits are inappropriate for time series data and would cause label leakage through overlapping sliding windows.

---

## Training Infrastructure

```
Optimizer:  AdamW, lr=1e-3
Loss:       MSE (regression) | categorical cross-entropy on logits (classification)
Gradients:  jax.value_and_grad(forward, has_aux=True)
Sliding window construction: vmap over jax.lax.dynamic_slice
```

The window stacking function (`stack_func`) builds the full `(N-T, T, F)` input array in one vectorized call using `jax.lax.dynamic_slice` inside `vmap`, avoiding the Python loop that earlier notebooks used.

---

## Results

**Airlines regression** converges smoothly over 100 epochs (MSE loss plotted per epoch). The model tracks the exponential trend and 12-month seasonal pattern on the in-sample data; the Plotly overlay of actual vs. predicted shows close alignment across the full 132-sample training range.

**Korean steel regression** trains on the full ~35,000 timestep series. The loss curve decreases over 100 epochs. The predicted signal follows the general shape of `Usage_kWh` across the in-sample range.

**Korean steel classification** achieves **91% accuracy** on the held-out test set (final 25% of the temporal sequence). A confusion matrix is generated at inference time using `sklearn.metrics.confusion_matrix` against `softmax(logits).argmax(axis=1)` vs `one_hot_targets.argmax(axis=1)`.

---

## Stack

| Component | Library |
|---|---|
| Array computation, autodiff, JIT | JAX |
| Optimizers | Optax |
| Notebook environment | Marimo 0.23.6 |
| Visualization | Plotly |
| Preprocessing | scikit-learn (`RobustScaler`, `MaxAbsScaler`, `ColumnTransformer`) |
| Data | pandas, NumPy |
