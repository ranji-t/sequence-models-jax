import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full", app_title="GRU Korean Iron Regression")


@app.cell
def _():
    # Standard Imports
    from functools import partial
    from typing import NamedTuple, Any

    # Third Party Imports
    import jax
    import optax
    import marimo as mo
    import numpy as np
    import pandas as pd
    import jax.numpy as jnp
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import RobustScaler, MaxAbsScaler
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    return (
        Any,
        ColumnTransformer,
        ConfusionMatrixDisplay,
        MaxAbsScaler,
        NamedTuple,
        RobustScaler,
        confusion_matrix,
        go,
        jax,
        jnp,
        make_subplots,
        mo,
        np,
        optax,
        partial,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Air Lines DataSet**
    """)
    return


@app.cell
def _(np, pd):
    # Read The file path
    url = r"D:\Codebase\NN-Architectures\data\Steel_industry_data.csv"

    # Read the data
    df = (
        pd.read_csv(url)
        .assign(
            date=lambda x: pd.to_datetime(
                x.loc[:, "date"].str.strip(), format=r"%d/%m/%Y %H:%M"
            )
        )
        .sort_values("date")
        .reset_index(drop=True)
        .rename(
            {
                "date": "timestamp",
                "CO2(tCO2)": "CO2",
                "Lagging_Current_Reactive.Power_kVarh": "Lagging_Current_Reactive_Power_kVarh",
            },
            axis=1,
        )
        .assign(
            # Time Harmonics
            week_day_sin=lambda x: np.sin(
                2 * np.pi * x.loc[:, "timestamp"].dt.weekday / 7
            ),
            week_day_cos=lambda x: np.cos(
                2 * np.pi * x.loc[:, "timestamp"].dt.weekday / 7
            ),
            month_sin=lambda x: np.sin(
                2 * np.pi * x.loc[:, "timestamp"].dt.month / 12
            ),
            month_cos=lambda x: np.cos(
                2 * np.pi * x.loc[:, "timestamp"].dt.month / 12
            ),
            time_sin=lambda x: np.sin(
                2
                * np.pi
                * (
                    (x.loc[:, "timestamp"].dt.hour * 60)
                    + x.loc[:, "timestamp"].dt.minute
                )
                / (24 * 60)
            ),
            time_cos=lambda x: np.cos(
                2
                * np.pi
                * (
                    (x.loc[:, "timestamp"].dt.hour * 60)
                    + x.loc[:, "timestamp"].dt.minute
                )
                / (24 * 60)
            ),
            # Categorical
            Load_Type=lambda x: x.loc[:, "Load_Type"].map(
                {"Light_Load": 0, "Medium_Load": 1, "Maximum_Load": 2}
            ),
            # Invert
            Usage_kWh=lambda x: np.log1p(x.loc[:, "Usage_kWh"]),
            Lagging_Current_Reactive_Power_kVarh=lambda x: np.log1p(
                x.loc[:, "Lagging_Current_Reactive_Power_kVarh"]
            ),
            Leading_Current_Reactive_Power_kVarh=lambda x: x.loc[
                :, "Leading_Current_Reactive_Power_kVarh"
            ],
            Lagging_Current_Power_Factor=lambda x: (
                1 - x.loc[:, "Lagging_Current_Power_Factor"] / 100
            ),
            Leading_Current_Power_Factor=lambda x: (
                1 - x.loc[:, "Leading_Current_Power_Factor"] / 100
            ),
        )
        .drop(["timestamp", "WeekStatus", "Day_of_week", "Usage_kWh"], axis=1)
    )

    # Display data
    df
    return (df,)


@app.cell
def _(df, go, make_subplots):
    # Get numeric columns only
    _cols = df.select_dtypes(include="number").columns.tolist()
    _n = len(_cols)
    _fig_hist = make_subplots(rows=(_n + 2) // 3, cols=3, subplot_titles=_cols)
    # ── Histograms ──────────────────────────────────────────
    for _i, _col in enumerate(_cols):
        _row, _col_idx = divmod(_i, 3)
        _fig_hist.add_trace(
            go.Histogram(
                x=df[_col].dropna(),
                name=_col,
                marker_color="#378ADD",
                showlegend=False,
            ),
            row=_row + 1,
            col=_col_idx + 1,
        )
    _fig_hist.update_layout(
        title=dict(text="<b>Feature Distributions</b>", x=0.5, font={"size": 22}),
        height=350 * ((_n + 2) // 3),
        bargap=0.1,
    )
    _fig_hist.show()
    _fig_box = make_subplots(rows=(_n + 2) // 3, cols=3, subplot_titles=_cols)
    for _i, _col in enumerate(_cols):
        _row, _col_idx = divmod(_i, 3)
        _fig_box.add_trace(
            go.Box(
                y=df[_col].dropna(),
                name=_col,
                marker_color="#E8724A",
                showlegend=False,
            ),
            row=_row + 1,
            col=_col_idx + 1,
        )
    _fig_box.update_layout(
        title=dict(text="<b>Feature Box Plots</b>", x=0.5, font={"size": 22}),
        height=350 * ((_n + 2) // 3),
    )
    # ── Box Plots ───────────────────────────────────────────
    _fig_box.show()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(ColumnTransformer, MaxAbsScaler, RobustScaler, df):
    df_tr = (
        ColumnTransformer(
            [
                (
                    "robust",
                    RobustScaler(with_centering=True),
                    ["NSM"],
                ),
                (
                    "robust-nm",
                    RobustScaler(
                        with_centering=False, quantile_range=(20.0, 80.0)
                    ),
                    [
                        "Lagging_Current_Reactive_Power_kVarh",
                        "Leading_Current_Reactive_Power_kVarh",
                    ],
                ),
                ("MaxAbsScaler", MaxAbsScaler(), ["CO2"]),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        .set_output(transform="pandas")
        .fit_transform(df)
    )

    df_tr
    return (df_tr,)


@app.cell
def _(df_tr, go, make_subplots):
    # Get numeric columns only
    _cols = df_tr.select_dtypes(include="number").columns.tolist()
    _n = len(_cols)
    _fig_hist = make_subplots(rows=(_n + 2) // 3, cols=3, subplot_titles=_cols)
    # ── Histograms ──────────────────────────────────────────
    for _i, _col in enumerate(_cols):
        _row, _col_idx = divmod(_i, 3)
        _fig_hist.add_trace(
            go.Histogram(
                x=df_tr[_col].dropna(),
                name=_col,
                marker_color="#378ADD",
                showlegend=False,
            ),
            row=_row + 1,
            col=_col_idx + 1,
        )
    _fig_hist.update_layout(
        title=dict(text="<b>Feature Distributions</b>", x=0.5, font={"size": 22}),
        height=350 * ((_n + 2) // 3),
        bargap=0.1,
    )
    _fig_hist.show()
    _fig_box = make_subplots(rows=(_n + 2) // 3, cols=3, subplot_titles=_cols)
    for _i, _col in enumerate(_cols):
        _row, _col_idx = divmod(_i, 3)
        _fig_box.add_trace(
            go.Box(
                y=df_tr[_col].dropna(),
                name=_col,
                marker_color="#E8724A",
                showlegend=False,
            ),
            row=_row + 1,
            col=_col_idx + 1,
        )
    _fig_box.update_layout(
        title=dict(text="<b>Feature Box Plots</b>", x=0.5, font={"size": 22}),
        height=350 * ((_n + 2) // 3),
    )
    # ── Box Plots ───────────────────────────────────────────
    _fig_box.show()
    return


@app.cell
def _(df_tr):
    df_tr.describe()
    return


@app.cell
def _(df_tr, jnp):
    # The Time Array
    X = jnp.array(df_tr.drop("Load_Type", axis=1).values)
    X.shape
    return (X,)


@app.cell
def _(df_tr, jax, jnp):
    # Number of classes
    num_classes = 3

    # convert to jax Array then Onehot the targets
    y = jnp.array(df_tr.loc[:, ["Load_Type"]].values)
    y = jax.nn.one_hot(y.squeeze(), num_classes=num_classes)
    y.shape
    return num_classes, y


@app.cell
def _():
    # # The Scafold
    # fig1 = go.Figure()

    # # Add Trace
    # fig1.add_trace(go.Scatter(y=y.ravel()))

    # # Layout
    # fig1.update_layout(
    #     title=dict(text="<b>Time Series Data</b>", x=0.5, font={"size": 25}),
    #     xaxis_title="<b>Time</b>",
    #     yaxis_title="<b>Passenger Count</b>",
    # )

    # # Show Figure
    # fig1.show()
    return


@app.cell
def _(num_classes):
    # Initial Sizes
    input_size = 12  # each timestep is ONE number (the count)
    hidden_size = 16  # memory capacity — your choice
    output_size = num_classes  # predicting ONE number (next month)
    timesteps = 12  # window size — how far back you look
    batch_size = 256  # batch size
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


    def log_loss(target: jax.Array, logit: jax.Array) -> jax.Array:
        return jnp.mean(
            -jnp.sum(target * jax.nn.log_softmax(logit, axis=1), axis=1)
        )

    return gru_cell, log_loss, output_functions


@app.cell
def _(ModelWeights, gru_cell, jax, jnp, log_loss, output_functions):
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
        return log_loss(target=y_b, logit=pred_b), pred_b

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
def _(X, stack_func, timesteps, y):
    # Stack the data first
    X_stack, y_stack = stack_func(X=X, y=y, timesteps=timesteps)

    # Train idx boundary
    train_idx = 3 * (X_stack.shape[0] // 4)

    # Test data set
    X_stack_test = X_stack[train_idx:, ...]
    y_stack_test = y_stack[train_idx:, ...]

    # Train data set
    X_stack = X_stack[:train_idx, ...]
    y_stack = y_stack[:train_idx, ...]
    return X_stack, X_stack_test, y_stack, y_stack_test


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
    epoch = 50
    key_permute = jax.random.key(3534)
    learning_rate = 1e-3

    # Get the model weights n bias
    weights = weights_init(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size
    )

    # Set optimizer
    optimizer = optax.adamw(learning_rate=learning_rate)
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

    # Grt losses
    losses = jax.block_until_ready(train_history[0].loss)
    return epoch_train_state, train_history


@app.cell
def _(go, jnp, train_history):
    # Get Figure
    fig2 = go.Figure()

    # Add Trace
    fig2.add_trace(
        go.Scatter(
            y=jnp.mean(train_history[0].loss, axis=1),
            mode="lines",
            name="Training Loss",
            line=dict(color="#378ADD", width=1.5),
        )
    )

    # layout
    fig2.update_layout(
        title=dict(text="<b>GRU Training Loss</b>", x=0.5, font={"size": 22}),
        xaxis_title="<b>Epoch</b>",
        yaxis_title="<b>Log Loss</b>",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig2.show()
    return


@app.cell
def _(
    X_stack_test,
    epoch_train_state,
    gru_predict,
    hidden_size,
    jax,
    y_stack_test,
):
    # Get current predictions
    y_pred = jax.vmap(gru_predict, in_axes=(None, 0, None))(
        epoch_train_state[0].weights, X_stack_test, hidden_size
    )

    # The predictions
    y_pred = y_pred
    y_true = y_stack_test
    return y_pred, y_true


@app.cell
def _(ConfusionMatrixDisplay, confusion_matrix, jax, plt, y_pred, y_true):
    # Confusion Matrix
    ConfusionMatrixDisplay(
        confusion_matrix(
            y_true=y_true.argmax(axis=1),
            y_pred=jax.nn.softmax(y_pred, axis=1).argmax(axis=1),
        )
    ).plot()
    # Show plot
    plt.show()
    return


if __name__ == "__main__":
    app.run()
