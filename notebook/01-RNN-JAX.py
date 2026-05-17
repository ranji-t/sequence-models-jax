import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full", app_title="RRN JAX")


@app.cell
def _():
    # Standard Imports
    from typing import NamedTuple, Any

    # Third Party Imports
    import jax
    import optax
    import marimo as mo
    import pandas as pd
    import jax.numpy as jnp
    import plotly.graph_objects as go

    return Any, NamedTuple, go, jax, jnp, mo, optax, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Air Lines DataSet**
    """)
    return


@app.cell
def _(pd):
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url)
    df
    return (df,)


@app.cell
def _(df, jnp):
    # The Time Array
    x = jnp.array(df.Passengers.values)
    x = jnp.reshape(x, (-1, 1))
    x
    return (x,)


@app.cell
def _(jnp, x):
    x_mean = x.mean()
    x_std = x.std()
    X = (x - x_mean) / jnp.maximum(x_std, 1e-6)
    X
    return X, x_mean, x_std


@app.cell
def _(df, go):
    # The Scafold
    fig1 = go.Figure()

    # Add Trace
    fig1.add_trace(go.Scatter(x=df.Month, y=df.Passengers))

    # Layout
    fig1.update_layout(
        title=dict(text="<b>Time Series Data</b>", x=0.5, font={"size": 25}),
        xaxis_title="<b>Time</b>",
        yaxis_title="<b>Passenger Count</b>",
    )

    # Show Figure
    fig1.show()
    return


@app.cell
def _():
    input_size = 1  # each timestep is ONE number (the count)
    timesteps = 12  # window size — how far back you look
    hidden_size = 16  # memory capacity — your choice
    output_size = 1  # predicting ONE number (next month)
    return hidden_size, input_size, output_size, timesteps


@app.cell
def _(NamedTuple, jax, jnp):
    class ModelWeights(NamedTuple):
        # RNN step
        Wx: jax.Array
        Wh: jax.Array
        bh: jax.Array
        # Output weights
        Wy: jax.Array
        by: jax.Array


    class RNNCarry(NamedTuple):
        weights: ModelWeights
        hidden_state: jax.Array


    def get_ortho_matrix(W: jax.Array, val: float = 0.1) -> jax.Array:
        # Get  Orthogonal matrix
        U, _, Vt = jnp.linalg.svd(W)
        # Return the Ortogonal Matrix
        return U * val


    def init_weights(
        input_size: int,
        hidden_size: int,
        output_size: int,
        key: jax.random.PRNGKey = jax.random.key(101),
    ) -> ModelWeights:
        # Split the keys
        key_Wx, key_Wh, key_Wy = jax.random.split(key, 3)

        # Init Weights
        Wx = (
            jax.random.normal(key_Wx, (input_size, hidden_size))
            * jnp.sqrt(2 / (input_size + hidden_size))
            * 0.1
        )
        Wh = get_ortho_matrix(
            jax.random.normal(key_Wh, (hidden_size, hidden_size)), 1.0
        )
        Wy = (
            jax.random.normal(key_Wy, (hidden_size, output_size))
            * jnp.sqrt(2 / (hidden_size + output_size))
            * 0.1
        )

        # Bias
        bh = jnp.zeros((hidden_size,))
        by = jnp.zeros((output_size,))

        # Return the Wights
        return ModelWeights(Wx, Wh, bh, Wy, by)


    def rnn_step(carry: RNNCarry, X: jax.Array) -> tuple[RNNCarry, jax.Array]:
        # Get Individual peices
        weights = carry.weights
        h_tm1 = carry.hidden_state

        # Extract Weights
        Wx = weights.Wx
        Wh = weights.Wh
        bh = weights.bh
        Wy = weights.Wy
        by = weights.by

        # Compute the new hidden state
        h_t = jax.nn.tanh(
            jnp.einsum("f,fh->h", X, Wx) + jnp.einsum("hd,h->d", Wh, h_tm1) + bh
        )

        # Return the Carry
        return RNNCarry(ModelWeights(Wx, Wh, bh, Wy, by), h_t), h_t


    def rnn_forward(rnn_carry: RNNCarry, x_ser: jax.Array):
        rnn_carry_new, h_hist = jax.lax.scan(rnn_step, rnn_carry, x_ser)
        return rnn_carry_new, h_hist


    def predict(rnn_carry: RNNCarry) -> jax.Array:
        # Get Hidden state &  Weights
        ht = rnn_carry.hidden_state
        weights = rnn_carry.weights

        # Get Final EWeight  and Bias
        Wy = weights.Wy
        by = weights.by

        # Get prediction
        y = jnp.einsum("h, ho -> o", ht, Wy) + by
        return y


    def loss_mse(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
        return jnp.mean(jnp.square(y_true - y_pred))


    def loss_func(
        weights: ModelWeights,
        hidden_state: jax.Array,
        x_ser: jax.Array,
        target: jax.Array,
    ) -> tuple[RNNCarry, jax.Array]:
        # Compute the carry
        rnn_carry, _ = rnn_forward(RNNCarry(weights, hidden_state), x_ser)
        # get the Pred
        pred = predict(rnn_carry)
        # compute loss
        loss = loss_mse(target, pred)
        # Return the data
        return loss, rnn_carry


    # Get the Gradinet nad loss function
    loss_func_grad_val = jax.jit(jax.value_and_grad(loss_func, has_aux=True))
    return RNNCarry, init_weights, loss_func_grad_val, predict, rnn_forward


@app.cell
def _(Any, NamedTuple, RNNCarry, jax, jnp, loss_func_grad_val, optax):
    class TrainState(NamedTuple):
        rnn_carry: RNNCarry
        opt_state: Any


    def reset_hiddenstate(train_state: TrainState) -> TrainState:
        return TrainState(
            RNNCarry(
                train_state.rnn_carry.weights,
                jnp.zeros_like(train_state.rnn_carry.hidden_state),
            ),
            train_state.opt_state,
        )


    def factory_train_step(optimizer):
        @jax.jit
        def train_step(
            train_state: TrainState, Xy: tuple[jax.Array, jax.Array]
        ) -> TrainState:
            # New Train state
            train_state_reset = reset_hiddenstate(train_state)

            # unpack things
            weights = train_state_reset.rnn_carry.weights
            hidden_state = train_state_reset.rnn_carry.hidden_state
            opt_state = train_state_reset.opt_state

            # upack Data
            X_batch = Xy[0]
            target_batch = Xy[1]

            # Carry the result
            (loss, rnn_carry_new), grad = loss_func_grad_val(
                weights,
                hidden_state,
                X_batch,
                target_batch,
            )

            ## Optimization step
            # Update thw sates
            update, opt_state_new = optimizer.update(
                grad, opt_state, rnn_carry_new.weights
            )
            # Update the wights
            weights_new = optax.apply_updates(rnn_carry_new.weights, update)

            # New Train State
            train_state_new = TrainState(
                RNNCarry(weights_new, rnn_carry_new.hidden_state), opt_state_new
            )

            # Returns
            return train_state_new, loss

        return train_step

    return TrainState, factory_train_step


@app.cell
def _(
    RNNCarry,
    TrainState,
    X,
    factory_train_step,
    hidden_size,
    init_weights,
    input_size,
    jnp,
    optax,
    output_size,
    timesteps,
):
    # Get Sample Numbers
    num_rows = X.shape[0]

    # Find the Batches
    batches = [
        (X[idx : idx + timesteps, :], X[idx + timesteps, :])
        for idx in range(num_rows - timesteps)
    ]


    # Init the Weights
    weights = init_weights(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size
    )

    # The initial Hidden State
    h0 = jnp.zeros((hidden_size,))

    # The RNN Carry
    rnn_carry = RNNCarry(weights, h0)

    # Set up optimizer
    optimizer = optax.adamw(learning_rate=1e-3)
    # Intialize the Opimizere state
    opt_state = optimizer.init(params=weights)

    # set up train state init
    train_state = TrainState(rnn_carry, opt_state)

    # Carry the result
    train_step = factory_train_step(optimizer)

    losses = []
    # Run the Batch
    for epoch in range(100):
        for bat in batches:
            train_state, loss = train_step(train_state, bat)
            losses.append(loss)
    return batches, h0, losses, train_state


@app.cell
def _(go, losses):
    fig2 = go.Figure()

    # Raw loss
    fig2.add_trace(
        go.Scatter(
            y=[float(l) for l in losses],
            mode="lines",
            name="loss",
            line=dict(color="#378ADD", width=1.5),
        )
    )

    # Smoothed loss — rolling mean window=8
    window = 8
    smoothed = [
        sum(losses[max(0, i - window) : i + 1])
        / len(losses[max(0, i - window) : i + 1])
        for i in range(len(losses))
    ]
    fig2.add_trace(
        go.Scatter(
            y=[float(s) for s in smoothed],
            mode="lines",
            name="smoothed",
            line=dict(color="#1D9E75", width=2, dash="dash"),
        )
    )

    fig2.update_layout(
        title=dict(text="<b>Training Loss</b>", x=0.5, font={"size": 20}),
        xaxis_title="<b>Batch</b>",
        yaxis_title="<b>MSE Loss</b>",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        hovermode="x unified",
    )

    fig2.show()
    return


@app.cell
def _(RNNCarry, batches, h0, predict, rnn_forward, train_state):
    # The predictions
    preds = []

    # Get the Final weights
    final_weights = train_state.rnn_carry.weights

    # Loop Through the bachhes
    for bats in batches:
        # Get Predctions
        rnn_carry_new, h_t = rnn_forward(RNNCarry(final_weights, h0), bats[0])

        # predict appended to pred
        preds.append(predict(rnn_carry_new))
    return (preds,)


@app.cell
def _(X, go, jnp, preds, timesteps, x_mean, x_std):
    # Denormalize
    y_pred = (jnp.array(preds) * x_std) + x_mean
    y_actual = (
        X[timesteps:] * x_std
    ) + x_mean  # align actual to prediction window

    fig3 = go.Figure()

    # Actual
    fig3.add_trace(
        go.Scatter(
            y=[float(v) for v in y_actual.flatten()],
            mode="lines",
            name="Actual",
            line=dict(color="#378ADD", width=2),
        )
    )

    # Predicted
    fig3.add_trace(
        go.Scatter(
            y=[float(v) for v in y_pred.flatten()],
            mode="lines",
            name="Predicted",
            line=dict(color="#E8724A", width=2, dash="dash"),
        )
    )

    fig3.update_layout(
        title=dict(
            text="<b>Actual vs Predicted — Airline Passengers</b>",
            x=0.5,
            font={"size": 20},
        ),
        xaxis_title="<b>Month</b>",
        yaxis_title="<b>Passenger Count</b>",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        hovermode="x unified",
    )

    fig3.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
