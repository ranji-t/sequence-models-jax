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

    return Any, NamedTuple, go, jax, jnp, mo, optax, partial, pd


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
    batch_size = 5  # batch size
    return batch_size, hidden_size, input_size, output_size, timesteps


@app.cell
def _(jax, jnp):
    def stack_func(
        X: jax.Array,
        y: jax.Array,
        timesteps: int,
    ) -> tuple[jax.Array, jax.Array]:
        # Get Sample Numbers
        num_rows = X.shape[0]

        # Find the Batches
        ids = jnp.arange(num_rows - timesteps)

        # Creat the stacks
        X_stack = jax.vmap(
            lambda i: jax.lax.dynamic_slice(X, (i, 0), (timesteps, X.shape[1]))
        )(ids)
        y_stack = jax.vmap(lambda i: y[i + timesteps])(ids)

        # The data
        return X_stack, y_stack


    def permute_array_func(arr: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        # get permuted indices
        permute = jax.random.permutation(key, jnp.arange(arr.shape[0]))
        # Return permuted arrya
        return arr[permute]


    def batcher_func(arr: jax.Array, batch_size: int) -> jax.Array:
        # size before batching
        n_samples = arr.shape[0]
        # The New Samples Numbers
        n_samples_trunc = (n_samples // batch_size) * batch_size
        # Print the data loss
        # print(f"Dropped rows are: {n_samples - n_samples_trunc}")
        # return batches
        return arr[:n_samples_trunc, ...].reshape(
            n_samples_trunc // batch_size, batch_size, *arr.shape[1:]
        )


    def permute_n_batch_func(
        X_stack: jax.Array,
        y_stack: jax.Array,
        batch_size: int,
        key: jax.random.PRNGKey = jax.random.key(0),
    ) -> tuple[jax.Array, jax.Array]:
        # permute the data
        X_perm = permute_array_func(arr=X_stack, key=key)
        y_perm = permute_array_func(arr=y_stack, key=key)

        # Now Batch the data
        X_batch, y_batch = (
            batcher_func(arr=X_perm, batch_size=batch_size),
            batcher_func(arr=y_perm, batch_size=batch_size),
        )
        # Return the data
        return X_batch, y_batch

    return permute_n_batch_func, stack_func


@app.cell
def _(NamedTuple, jax, jnp):
    class ModelWeights(NamedTuple):
        # Reset Gate Weights
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

    return ModelWeights, weights_init


@app.cell
def _(ModelWeights, jax, jnp):
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


    def gru_cell(
        weights: ModelWeights, X: jnp.array, h_tm1: jax.Array
    ) -> tuple[jax.Array, jax.Array]:

        # Reset Gate
        reset_gate = gate_func(
            Wx=weights.Wxr,
            Wh=weights.Whr,
            b=weights.br,
            X=X,
            h_tm1=h_tm1,
        )
        # Update Gate
        update_gate = gate_func(
            Wx=weights.Wxz,
            Wh=weights.Whz,
            b=weights.bz,
            X=X,
            h_tm1=h_tm1,
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
        return h_t, h_t


    def output_functions(
        Wy: jax.Array, by: jax.Array, h_t: jax.Array
    ) -> jax.Array:
        return jnp.einsum("h,ho->o", h_t, Wy) + by


    def mse_loss(target: jax.Array, predict: jax.Array):
        return jnp.mean(jnp.square(target - predict))

    return gru_cell, mse_loss, output_functions


@app.cell
def _(ModelWeights, gru_cell, jax, jnp, mse_loss, output_functions):
    def gru_predict(
        weights: ModelWeights, X_t: jax.Array, hidden_size: int
    ) -> jax.Array:
        # The Initial_state of the H0
        h_init = jnp.zeros((hidden_size,))

        # Traverse the time step
        h_final, _ = jax.lax.scan(
            lambda h, X: gru_cell(weights=weights, h_tm1=h, X=X),
            init=h_init,
            xs=X_t,
        )
        # Call the output functions
        y_t = output_functions(Wy=weights.Wy, by=weights.by, h_t=h_final)

        # return pred
        return y_t


    def batch_predict(
        weights: ModelWeights, X_b: jax.Array, hidden_size: int
    ) -> jax.Array:
        # Get prections in the batch
        pred_batch = jax.vmap(gru_predict, in_axes=(None, 0, None))(
            weights, X_b, hidden_size
        )
        # Return the data
        return pred_batch


    def forward(
        weights: ModelWeights, X_b: jax.Array, y_b: jax.Array, hidden_size: int
    ):
        # Get predictions
        pred_b = batch_predict(weights=weights, X_b=X_b, hidden_size=hidden_size)

        # Loss Functions
        return mse_loss(predict=pred_b, target=y_b), pred_b

    return forward, gru_predict


@app.cell
def _(
    Any,
    ModelWeights,
    NamedTuple,
    forward,
    jax,
    optax,
    partial,
    permute_n_batch_func,
):
    class TrainState(NamedTuple):
        weights: ModelWeights
        opt_state: Any


    class EpochTrainState(NamedTuple):
        train_state: TrainState
        key_permute: jax.random.PRNGKey


    class TrainHistory(NamedTuple):
        loss: jax.Array
        pred: jax.Array


    def get_batch_train_step(optimizer):

        def batch_train_step(
            train_state: TrainState,
            DS_b: tuple[jax.Array, jax.Array],
            hidden_size: int,
        ):
            # Extract the value
            X_b, y_b = DS_b

            # Extract form trian state
            weights, opt_state = train_state

            # Get Loss, Grad & predictions
            (loss, pred), grads = jax.value_and_grad(forward, has_aux=True)(
                weights, X_b=X_b, y_b=y_b, hidden_size=hidden_size
            )

            # Udate step and opt state
            updates, opt_state = optimizer.update(grads, opt_state, params=weights)
            weights = optax.apply_updates(weights, updates)

            # Return the data
            return TrainState(weights, opt_state), TrainHistory(loss, pred)

        # Return callable
        return batch_train_step


    def get_epoch_train_step(optimizer):
        # Get Train Functions
        batch_train_step = get_batch_train_step(optimizer)

        @partial(jax.jit, static_argnames=["batch_size", "hidden_size"])
        def epoch_train_step(
            epoch_train_state: EpochTrainState,
            X_stack: jax.Array,
            y_stack: jax.Array,
            batch_size: int,
            hidden_size: int,
        ):
            # Unpack the epoch trian state
            train_state, key_permute = epoch_train_state

            # Split the key
            key_permute, _ = jax.random.split(key_permute, 2)

            # Batch the data X: (batch_no, batch_size, time_step, output)
            # Batch the data Y: (batch_no, batch_size, output)
            X_batch, y_batch = permute_n_batch_func(
                X_stack=X_stack,
                y_stack=y_stack,
                batch_size=batch_size,
                key=key_permute,
            )

            # One Epoch Of Training
            train_state, train_history = jax.lax.scan(
                lambda train_state, DS_b: batch_train_step(
                    train_state, DS_b=DS_b, hidden_size=hidden_size
                ),
                init=train_state,
                xs=(X_batch, y_batch),
            )

            # New epoch state
            epoch_train_state = EpochTrainState(train_state, key_permute)

            return epoch_train_state, (train_history, y_batch)

        return epoch_train_step

    return EpochTrainState, TrainState, get_epoch_train_step


@app.cell
def _(X, stack_func, timesteps):
    # Stack the data first
    X_stack, y_stack = stack_func(X=X, y=X, timesteps=timesteps)
    return X_stack, y_stack


@app.cell
def _(
    EpochTrainState,
    TrainState,
    X_stack,
    batch_size,
    get_epoch_train_step,
    hidden_size,
    input_size,
    jax,
    optax,
    output_size,
    weights_init,
    y_stack,
):
    # Controls
    epoch = 100
    key_permute = jax.random.key(3534)

    # Get the model weights n bias
    weights = weights_init(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size
    )

    # Set optimizer
    optimizer = optax.adamw(learning_rate=1e-3)
    # Get Optimizer states
    opt_state = optimizer.init(weights)

    # init train state
    train_state = TrainState(weights, opt_state)
    epoch_train_state = EpochTrainState(train_state, key_permute)

    # Get Train Functions
    epoch_train_step = get_epoch_train_step(optimizer)

    # Loop thorgh the epoches
    epoch_train_state, train_history = jax.lax.scan(
        lambda epoch_train_state, _: epoch_train_step(
            epoch_train_state,
            X_stack=X_stack,
            y_stack=y_stack,
            batch_size=batch_size,
            hidden_size=hidden_size,
        ),
        init=epoch_train_state,
        length=epoch,
    )
    return epoch_train_state, train_history


@app.cell
def _(go, jnp, train_history):
    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            y=jnp.mean(train_history[0].loss, axis=1),
            mode="lines",
            name="Training Loss",
            line=dict(color="#378ADD", width=1.5),
        )
    )

    fig2.update_layout(
        title=dict(text="<b>GRU Training Loss</b>", x=0.5, font={"size": 22}),
        xaxis_title="<b>Epoch</b>",
        yaxis_title="<b>MSE Loss</b>",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig2.show()
    return


@app.cell
def _(
    X_stack,
    epoch_train_state,
    gru_predict,
    hidden_size,
    jax,
    x_mean,
    x_std,
    y_stack,
):
    # Get couuurent predictions
    y_pred = jax.vmap(gru_predict, in_axes=(None, 0, None))(
        epoch_train_state[0].weights, X_stack, hidden_size
    )

    # The predictions
    y_pred = (y_pred * x_std + x_mean).ravel()
    y_true = (y_stack * x_std + x_mean).ravel()
    return y_pred, y_true


@app.cell
def _(go, y_pred, y_true):
    # Get Figure
    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            y=y_true,
            mode="lines",
            name="Actual",
            line=dict(color="#378ADD", width=2),
        )
    )

    fig3.add_trace(
        go.Scatter(
            y=y_pred,
            mode="lines",
            name="Predicted",
            line=dict(color="#E8724A", width=2, dash="dash"),
        )
    )

    fig3.update_layout(
        title=dict(
            text="<b>GRU — Actual vs Predicted</b>", x=0.5, font={"size": 22}
        ),
        xaxis_title="<b>Step</b>",
        yaxis_title="<b>Passenger Count</b>",
        hovermode="x unified",
    )

    fig3.show()
    return


if __name__ == "__main__":
    app.run()
