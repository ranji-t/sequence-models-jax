import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full", app_title="LSTM JAX")


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
        # Forget Gate Weights
        Wxf: jax.Array
        Whf: jax.Array
        bf: jax.Array
        # Input Gate Weights
        Wxi: jax.Array
        Whi: jax.Array
        bi: jax.Array
        # Output Gate Weights
        Wxo: jax.Array
        Who: jax.Array
        bo: jax.Array
        # Candidate wieghts
        Wxc: jax.Array
        Whc: jax.Array
        bc: jax.Array
        # Output Layer
        Wy: jax.Array
        by: jax.Array


    class LSTMCarry(NamedTuple):
        weights: ModelWeights
        h_tm1: jax.Array
        c_tm1: jax.Array


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
            key_Wxf,
            key_Whf,
            key_Wxi,
            key_Whi,
            key_Wxo,
            key_Who,
            key_Wxc,
            key_Whc,
            key_Wy,
        ) = jax.random.split(key, 9)

        # Forget Weights Init
        Wxf = (
            jax.random.normal(key_Wxf, (input_size, hidden_size))
            * jnp.sqrt(2 / (input_size + hidden_size))
            * 0.1
        )
        Whf = ortogonalize_matrix(
            jax.random.normal(key_Whf, (hidden_size, hidden_size))
        )
        bf = jnp.zeros((hidden_size,))

        # Input Weights Init
        Wxi = (
            jax.random.normal(key_Wxi, (input_size, hidden_size))
            * jnp.sqrt(2 / (input_size + hidden_size))
            * 0.1
        )
        Whi = ortogonalize_matrix(
            jax.random.normal(key_Whi, (hidden_size, hidden_size))
        )
        bi = jnp.zeros((hidden_size,))

        # Output Weights Init
        Wxo = (
            jax.random.normal(key_Wxo, (input_size, hidden_size))
            * jnp.sqrt(2 / (input_size + hidden_size))
            * 0.1
        )
        Who = ortogonalize_matrix(
            jax.random.normal(key_Who, (hidden_size, hidden_size))
        )
        bo = jnp.zeros((hidden_size,))

        # Caidates Generations Weights Init
        Wxc = (
            jax.random.normal(key_Wxc, (input_size, hidden_size))
            * jnp.sqrt(2 / (input_size + hidden_size))
            * 0.1
        )
        Whc = ortogonalize_matrix(
            jax.random.normal(key_Whc, (hidden_size, hidden_size))
        )
        bc = jnp.zeros((hidden_size,))

        # Output layer
        Wy = (
            jax.random.normal(key_Wy, (hidden_size, output_size))
            * jnp.sqrt(2 / (hidden_size + output_size))
            * 0.1
        )
        by = jnp.zeros((output_size,))

        # Return Model Weights
        return ModelWeights(
            Wxf,
            Whf,
            bf,
            # Input Gate Weights
            Wxi,
            Whi,
            bi,
            # Output Gate Weights
            Wxo,
            Who,
            bo,
            # Candidate wieghts
            Wxc,
            Whc,
            bc,
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
        Wxc: jax.Array,
        Whc: jax.Array,
        bc: jax.Array,
        X: jax.Array,
        h_tm1: jax.Array,
    ) -> jax.Array:
        return jax.nn.tanh(
            jnp.einsum("f,fh->h", X, Wxc) + jnp.einsum("h,hi->i", h_tm1, Whc) + bc
        )


    def memory_update_func(
        forget_gate: jax.Array,
        input_gate: jax.Array,
        old_cadidate: jax.Array,
        new_cadidate: jax.Array,
    ) -> jax.Array:
        return (forget_gate * old_cadidate) + (input_gate * new_cadidate)


    def hidden_state_func(
        output_gate: jax.Array, updated_candidate: jax.Array
    ) -> jax.Array:
        return output_gate * jax.nn.tanh(updated_candidate)


    def output_functions(
        Wy: jax.Array, by: jax.Array, h_t: jax.Array
    ) -> jax.Array:
        return jnp.einsum("h,ho->o", h_t, Wy) + by


    def mse_loss(target: jax.Array, predict: jax.Array):
        return jnp.mean(jnp.square(target - predict))

    return (
        LSTMCarry,
        ModelWeights,
        candidate_func,
        gate_func,
        hidden_state_func,
        memory_update_func,
        mse_loss,
        output_functions,
        weights_init,
    )


@app.cell
def _(
    LSTMCarry,
    ModelWeights,
    candidate_func,
    gate_func,
    hidden_state_func,
    jax,
    memory_update_func,
    mse_loss,
    output_functions,
):
    def lstm_step(
        lstm_carry: LSTMCarry, X: jax.Array
    ) -> tuple[LSTMCarry, jax.Array]:
        # Extract contents
        weights = lstm_carry.weights
        h_tm1 = lstm_carry.h_tm1
        c_tm1 = lstm_carry.c_tm1

        # Forget Gate
        forget_gate = gate_func(
            Wx=weights.Wxf, Wh=weights.Whf, b=weights.bf, X=X, h_tm1=h_tm1
        )

        # Input Gate
        input_gate = gate_func(
            Wx=weights.Wxi, Wh=weights.Whi, b=weights.bi, X=X, h_tm1=h_tm1
        )

        # Input Gate
        output_gate = gate_func(
            Wx=weights.Wxo, Wh=weights.Who, b=weights.bo, X=X, h_tm1=h_tm1
        )

        # New cadiate generation
        c_new = candidate_func(
            Wxc=weights.Wxc, Whc=weights.Whc, bc=weights.bc, X=X, h_tm1=h_tm1
        )

        # Update candidate
        c_t = memory_update_func(
            forget_gate=forget_gate,
            input_gate=input_gate,
            old_cadidate=c_tm1,
            new_cadidate=c_new,
        )

        # Get the Hidden state
        h_t = hidden_state_func(output_gate=output_gate, updated_candidate=c_t)

        # Return
        return LSTMCarry(weights, h_t, c_t), h_t


    def lstm_chain(
        lstm_carry: LSTMCarry, X: jax.Array
    ) -> tuple[LSTMCarry, jax.Array]:
        lstm_carry, h_t = jax.lax.scan(lstm_step, lstm_carry, X)
        return lstm_carry, h_t


    def lstm_forward(
        weights: ModelWeights,
        h_tm1: jax.Array,
        c_tm1: jax.Array,
        DS: tuple[jax.Array, jax.Array],
    ):
        # The data points
        X, target = DS

        # Form Carry
        lstm_carry = LSTMCarry(weights, h_tm1, c_tm1)

        # Comnpute the new H_t
        lstm_carry_new, _ = lstm_chain(lstm_carry, X)

        # latest H-t
        h_t = lstm_carry_new.h_tm1
        Wy = lstm_carry_new.weights.Wy
        by = lstm_carry_new.weights.by

        # Make Predictions
        predict = output_functions(Wy=Wy, by=by, h_t=h_t)

        # Loss Func
        loss = mse_loss(target=target, predict=predict)

        # Return
        return loss, (lstm_carry_new, predict)


    # The Gradients
    lstm_forward_grad = jax.value_and_grad(lstm_forward, has_aux=True)
    return (lstm_forward_grad,)


@app.cell
def _(Any, LSTMCarry, NamedTuple, jax, jnp, lstm_forward_grad, optax):
    class TrainState(NamedTuple):
        lstm_carry: LSTMCarry
        opt_state: Any


    def train_step_factory(optimizer):
        @jax.jit
        def train_step(
            train_state: TrainState, DS: tuple[jax.Array, jax.Array]
        ) -> TrainState:
            # Calculate the gradients
            (loss, (lstm_carry, pred)), grad = lstm_forward_grad(
                train_state.lstm_carry.weights,
                jnp.zeros_like(train_state.lstm_carry.h_tm1),
                jnp.zeros_like(train_state.lstm_carry.c_tm1),
                DS,
            )

            # Update the optimizer
            updates, opt_state = optimizer.update(
                params=lstm_carry.weights,
                state=train_state.opt_state,
                updates=grad,
            )
            # update the grads
            new_weights = optax.apply_updates(
                params=lstm_carry.weights, updates=updates
            )

            # Latest LSTM Carry
            lstm_carry_new = LSTMCarry(
                new_weights, lstm_carry.h_tm1, lstm_carry.c_tm1
            )
            train_state_new = TrainState(lstm_carry_new, opt_state)
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
    LSTMCarry,
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
    c0 = jnp.zeros((hidden_size,))
    # LSTM Carry
    lstm_carry = LSTMCarry(weights, h0, c0)

    ## Set up Optimizer
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(weights)

    # The trian func
    train_step_func = train_step_factory(optimizer=optimizer)

    # Set Train state
    train_state = TrainState(lstm_carry, opt_state)

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


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
