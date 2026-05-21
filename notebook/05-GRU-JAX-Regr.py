import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full", app_title="GRU Airlines Regression")


@app.cell
def _():
    # Standard Imports
    from functools import partial
    from typing import NamedTuple, Any

    # Third Party Imports
    import jax
    import optax
    import marimo as mo
    import pandas as pd
    import jax.numpy as jnp
    import plotly.graph_objects as go
    from tqdm import tqdm

    return Any, NamedTuple, go, jax, jnp, mo, optax, pd, tqdm


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
    # Initial Sizes
    input_size = 1  # each timestep is ONE number (the count)
    hidden_size = 16  # memory capacity — your choice
    output_size = 1  # predicting ONE number (next month)
    timesteps = 12  # window size — how far back you look
    return hidden_size, input_size, output_size, timesteps


@app.cell
def _(NamedTuple, jax, jnp):
    class ModelWeights(NamedTuple):
        # Rest Gate Weights
        Wxr: jax.Array
        Whr: jax.Array
        br: jax.Array
        # Update Gate Weights
        Wxz: jax.Array
        Whz: jax.Array
        bz: jax.Array
        # Candidate generate Weights
        Wxh: jax.Array
        Whh: jax.Array
        bh: jax.Array
        # Output Layer
        Wy: jax.Array
        by: jax.Array


    class GRUCarry(NamedTuple):
        weights: ModelWeights
        h: jax.Array


    def ortogonalize_matrix(W: jax.Array) -> jax.Array:
        U, *_ = jnp.linalg.svd(W)
        return U


    def weights_init(
        input_size: int,
        hidden_size: int,
        output_size: int,
        key: jax.random.PRNGKey = jax.random.key(42),
    ) -> ModelWeights:
        # Key split
        (
            key_Wxr,
            key_Whr,
            key_Wxz,
            key_Whz,
            key_Wxh,
            key_Whh,
            key_Wy,
        ) = jax.random.split(key, 7)

        # Reset Weights Init
        Wxr = (
            jax.random.normal(key_Wxr, (input_size, hidden_size))
            * jnp.sqrt(2 / (input_size + hidden_size))
            * 0.1
        )
        Whr = ortogonalize_matrix(
            jax.random.normal(key_Whr, (hidden_size, hidden_size))
        )
        br = jnp.zeros((hidden_size,))

        # Update Weights Init
        Wxz = (
            jax.random.normal(key_Wxz, (input_size, hidden_size))
            * jnp.sqrt(2 / (input_size + hidden_size))
            * 0.1
        )
        Whz = ortogonalize_matrix(
            jax.random.normal(key_Whz, (hidden_size, hidden_size))
        )
        bz = jnp.zeros((hidden_size,))

        # Candidate Weights Init
        Wxh = (
            jax.random.normal(key_Wxh, (input_size, hidden_size))
            * jnp.sqrt(2 / (input_size + hidden_size))
            * 0.1
        )
        Whh = ortogonalize_matrix(
            jax.random.normal(key_Whh, (hidden_size, hidden_size))
        )
        bh = jnp.zeros((hidden_size,))

        # Output layer
        Wy = (
            jax.random.normal(key_Wy, (hidden_size, output_size))
            * jnp.sqrt(2 / (hidden_size + output_size))
            * 0.1
        )
        by = jnp.zeros((output_size,))

        # Return Model Weights
        return ModelWeights(
            Wxr,
            Whr,
            br,
            # Input Gate Weights
            Wxz,
            Whz,
            bz,
            # Output Gate Weights
            Wxh,
            Whh,
            bh,
            # Output Layer
            Wy,
            by,
        )


    def gate_func(
        Wx: jax.Array, Wh: jax.Array, b: jax.Array, X: jax.Array, h_tm1: jax.Array
    ) -> jax.Array:
        return jax.nn.sigmoid(
            jnp.einsum("f,fh->h", X, Wx) + jnp.einsum("h,hi->i", h_tm1, Wh) + b
        )


    def candidate_func(
        Wxh: jax.Array,
        Whh: jax.Array,
        bh: jax.Array,
        X: jax.Array,
        h_tm1: jax.Array,
        reset_gate: jax.Array,
    ) -> jax.Array:
        return jax.nn.tanh(
            jnp.einsum("f,fh->h", X, Wxh)
            + jnp.einsum("h,hi->i", (reset_gate * h_tm1), Whh)
            + bh
        )


    def hiddenstate_update_func(
        update_gate: jax.Array, h_candidate: jax.Array, h_tm1: jax.Array
    ) -> jax.Array:
        return ((1 - update_gate) * h_tm1) + (update_gate * h_candidate)


    def output_functions(
        Wy: jax.Array, by: jax.Array, h_t: jax.Array
    ) -> jax.Array:
        return jnp.einsum("h,ho->o", h_t, Wy) + by


    def mse_loss(target: jax.Array, predict: jax.Array):
        return jnp.mean(jnp.square(target - predict))

    return (
        GRUCarry,
        ModelWeights,
        candidate_func,
        gate_func,
        hiddenstate_update_func,
        mse_loss,
        output_functions,
        weights_init,
    )


@app.cell
def _(
    GRUCarry,
    ModelWeights,
    candidate_func,
    gate_func,
    hiddenstate_update_func,
    jax,
    mse_loss,
    output_functions,
):
    def gru_step(gru_carry: GRUCarry, X: jax.Array) -> tuple[GRUCarry, jax.Array]:
        # Extract contents
        weights: ModelWeights = gru_carry.weights
        h_tm1: jax.Array = gru_carry.h

        # Reset Gate
        reset_gate = gate_func(
            Wx=weights.Wxr, Wh=weights.Whr, b=weights.br, X=X, h_tm1=h_tm1
        )
        # Update Gate
        update_gate = gate_func(
            Wx=weights.Wxz, Wh=weights.Whz, b=weights.bz, X=X, h_tm1=h_tm1
        )

        # Update candidate
        h_candidate = candidate_func(
            Wxh=weights.Wxh,
            Whh=weights.Whh,
            bh=weights.bh,
            X=X,
            h_tm1=h_tm1,
            reset_gate=reset_gate,
        )

        # Get the Hidden state
        h_t = hiddenstate_update_func(
            update_gate=update_gate, h_candidate=h_candidate, h_tm1=h_tm1
        )

        # Return
        return GRUCarry(weights, h_t), h_t


    def gru_chain(gru_carry: GRUCarry, X: jax.Array) -> tuple[GRUCarry, jax.Array]:
        gru_carry, h_t = jax.lax.scan(gru_step, gru_carry, X)
        return gru_carry, h_t


    def gru_forward(
        weights: ModelWeights,
        h: jax.Array,
        DS: tuple[jax.Array, jax.Array],
    ):
        # The data points
        X, target = DS

        # Form Carry
        gru_carry = GRUCarry(weights, h)

        # Comnpute the new H_t
        gru_carry_new, _ = gru_chain(gru_carry, X)

        # latest H-t
        h_t = gru_carry_new.h
        Wy = gru_carry_new.weights.Wy
        by = gru_carry_new.weights.by

        # Make Predictions
        predict = output_functions(Wy=Wy, by=by, h_t=h_t)

        # Loss Func
        loss = mse_loss(target=target, predict=predict)

        # Return
        return loss, (gru_carry_new, predict)


    # The Gradients
    gru_forward_grad = jax.value_and_grad(gru_forward, has_aux=True)
    return (gru_forward_grad,)


@app.cell
def _(Any, GRUCarry, NamedTuple, gru_forward_grad, jax, jnp, optax):
    class TrainState(NamedTuple):
        gru_carry: GRUCarry
        opt_state: Any


    def train_step_factory(optimizer):
        @jax.jit
        def train_step(
            train_state: TrainState, DS: tuple[jax.Array, jax.Array]
        ) -> TrainState:
            # Calculate the gradients
            (loss, (gru_carry, pred)), grad = gru_forward_grad(
                train_state.gru_carry.weights,
                jnp.zeros_like(train_state.gru_carry.h),
                DS,
            )

            # Update the optimizer
            updates, opt_state = optimizer.update(
                params=gru_carry.weights,
                state=train_state.opt_state,
                updates=grad,
            )
            # update the grads
            new_weights = optax.apply_updates(
                params=gru_carry.weights, updates=updates
            )

            # Latest GRU Carry
            gru_carry_new = GRUCarry(new_weights, gru_carry.h)
            train_state_new = TrainState(gru_carry_new, opt_state)
            return train_state_new, (loss, pred, DS[1])

        return train_step

    return TrainState, train_step_factory


@app.cell
def _(X, jnp, timesteps):
    # Get Sample Numbers
    num_rows = X.shape[0]

    # Find the Batches
    batches = [
        (X[idx : idx + timesteps, :], X[idx + timesteps, :])
        for idx in range(num_rows - timesteps)
    ]

    X_stack = jnp.stack([_[0] for _ in batches])
    target_stack = jnp.stack([_[1] for _ in batches])
    return X_stack, target_stack


@app.cell
def _(
    GRUCarry,
    TrainState,
    X_stack,
    hidden_size,
    input_size,
    jax,
    jnp,
    optax,
    output_size,
    target_stack,
    tqdm,
    train_step_factory,
    weights_init,
):
    ## Hyper Paraments
    epochs = 75
    learning_rate = 1e-3

    ## init Zone
    # initialize the weights
    weights = weights_init(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size
    )
    # Initialize the weights
    h0 = jnp.zeros((hidden_size,))

    # LSTM Carry
    gru_carry = GRUCarry(weights, h0)

    ## Set up Optimizer
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(weights)

    # The trian func
    train_step_func = train_step_factory(optimizer=optimizer)

    # Set Train state
    train_state = TrainState(gru_carry, opt_state)

    train_history = []
    for _ in tqdm(range(epochs)):
        train_state, aux = jax.lax.scan(
            train_step_func,
            init=train_state,
            xs=(
                X_stack,
                target_stack,
            ),
        )
        train_history.append(aux)
    return (train_history,)


@app.cell
def _(go, jnp, train_history):
    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            y=jnp.concat([h[0] for h in train_history]),
            mode="lines",
            name="Training Loss",
            line=dict(color="#378ADD", width=1.5),
        )
    )

    fig2.update_layout(
        title=dict(text="<b>LSTM Training Loss</b>", x=0.5, font={"size": 22}),
        xaxis_title="<b>Step</b>",
        yaxis_title="<b>MSE Loss</b>",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig2.show()
    return


@app.cell
def _(go, train_history, x_mean, x_std):
    target = (train_history[-1][2].squeeze() * x_std) + x_mean  # Target
    pred = (train_history[-1][1].squeeze() * x_std) + x_mean  # Prediction

    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            y=target.tolist(),
            mode="lines",
            name="Actual",
            line=dict(color="#378ADD", width=2),
        )
    )

    fig3.add_trace(
        go.Scatter(
            y=pred.tolist(),
            mode="lines",
            name="Predicted",
            line=dict(color="#E8724A", width=2, dash="dash"),
        )
    )

    fig3.update_layout(
        title=dict(
            text="<b>LSTM — Actual vs Predicted</b>", x=0.5, font={"size": 22}
        ),
        xaxis_title="<b>Step</b>",
        yaxis_title="<b>Passenger Count</b>",
        hovermode="x unified",
    )

    fig3.show()
    return


if __name__ == "__main__":
    app.run()
