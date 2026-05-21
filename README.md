# sequence-models-jax

Gated Recurrent Unit (GRU) implemented from scratch in pure functional JAX — no Flax, no Haiku, no abstraction layers.

## What's here
Three notebooks built progressively over one week:
- **GRU Regression — Airlines**: Univariate time series forecasting on the classic Airlines passenger dataset. Tracks trend and seasonality with tight in-sample fit.
- **GRU Regression — Korean Steel**: Industrial power consumption forecasting on 35k timesteps. Regime-switching signal with sharp on/off transitions.
- **GRU Classification — Korean Steel**: 3-class operating state classification on the same dataset. 91% accuracy on held-out test data.

## What makes this different
- GRU cell implemented from first principles — reset gate, update gate, candidate state, all explicit
- Weights treated as parameters, not carry — correct functional JAX pattern
- Orthogonal initialization on hidden-to-hidden weights for gradient stability
- Mini-batching via `vmap`, sequence unrolling via `lax.scan`
- Full training loop compiled to XLA via nested `lax.scan` on small datasets
- Proper temporal train/test split — no data leakage

## Stack
JAX · Optax · Marimo · Plotly