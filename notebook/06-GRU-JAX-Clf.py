import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full", app_title="GRU Korean-Iron Classification")


@app.cell
def _():
    # Standard Imports
    from typing import NamedTuple, Any

    # Third Party Imports
    import jax
    import optax
    import marimo as mo
    import numpy as np
    import pandas as pd
    import jax.numpy as jnp
    from tqdm import tqdm
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import (
        MaxAbsScaler,
        RobustScaler,
    )

    return (
        Any,
        ColumnTransformer,
        MaxAbsScaler,
        NamedTuple,
        RobustScaler,
        go,
        jax,
        jnp,
        make_subplots,
        mo,
        np,
        optax,
        pd,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Korean Iron DataSet**
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
        .drop(["Usage_kWh", "timestamp", "WeekStatus", "Day_of_week"], axis=1)
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
def _(df, df_tr, go, make_subplots):
    # Get numeric columns only
    _cols = df.select_dtypes(include="number").columns.tolist()
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
def _(df_tr, jnp):
    # The Time Array
    X = jnp.array(df_tr.drop("Load_Type", axis=1).values)
    X.shape
    return (X,)


@app.cell
def _(df_tr, jax, jnp):
    y = jnp.array(df_tr.loc[:, ["Load_Type"]].values)
    y = jax.nn.one_hot(y.squeeze(), 3)
    y.shape
    return (y,)


@app.cell
def _():
    input_size = 12  # each timestep is ONE number (the count)
    timesteps = 12  # window size — how far back you look
    hidden_size = 16  # memory capacity — your choice
    output_size = 3  # predicting ONE number (next month)
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


    def log_loss(target: jax.Array, logits: jax.Array) -> jax.Array:
        return jnp.mean(jnp.sum(target * jax.nn.log_softmax(logits), axis=-1))

    return (
        GRUCarry,
        ModelWeights,
        candidate_func,
        gate_func,
        hiddenstate_update_func,
        log_loss,
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
    log_loss,
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
        logits = output_functions(Wy=Wy, by=by, h_t=h_t)

        # Loss Func
        loss = log_loss(target=target, logits=logits)

        # Return
        return loss, (gru_carry_new, logits)


    # The Gradients
    gru_forward_grad = jax.value_and_grad(gru_forward, has_aux=True)
    return (gru_forward_grad,)


@app.cell
def _(Any, GRUCarry, NamedTuple, gru_forward_grad, jax, optax):
    class TrainState(NamedTuple):
        gru_carry: GRUCarry
        opt_state: Any


    def train_step_factory(optimizer):
        @jax.jit
        def train_step(
            train_state: TrainState, DS: tuple[jax.Array, jax.Array]
        ) -> TrainState:
            # The data
            _, y = DS

            # Weights
            weights = train_state.gru_carry.weights
            h = train_state.gru_carry.h

            # Calculate the gradients
            (loss, (gru_carry, logits)), grad = jax.vmap(
                gru_forward_grad, in_axes=(None, None, 0)
            )(
                weights,
                h,
                DS,
            )

            # Update the optimizer
            updates, opt_state = optimizer.update(
                params=weights,
                state=train_state.opt_state,
                updates=grad,
            )
            # update the grads
            new_weights = optax.apply_updates(params=weights, updates=updates)

            # Latest GRU Carry
            gru_carry_new = GRUCarry(new_weights, h)
            train_state_new = TrainState(gru_carry_new, opt_state)

            # The returns
            return train_state_new, (loss, logits, y)

        return train_step

    return TrainState, train_step_factory


@app.cell
def _(Jax, jax, jnp, timesteps, tqdm):
    def batch_data(X: jax.Array, y: jax.Array) -> jax.Array:
        # Get Sample Numbers
        num_rows = X.shape[0]
        # Containers
        x_list = []
        y_list = []

        # Find the Batches
        for idx in tqdm(range(num_rows - timesteps)):
            x_list.append(X[idx : idx + timesteps, :])
            y_list.append(y[idx + timesteps, :])

        # Stack the batches
        X_stack = jnp.stack(x_list)
        y_stack = jnp.stack(y_list)

        # Return
        return X_stack, y_stack


    def batch_data_func(arr: Jax.Array, batch_size: int):
        # The Samples slected for batches
        n_samples = (arr.shape[0] // batch_size) * batch_size
        # Remaining Axis
        _, *axes = arr.shape
        # The Out put after batching
        return arr[:n_samples, ...].reshape(
            n_samples // batch_size, batch_size, *axes
        )

    return batch_data, batch_data_func


@app.cell
def _(X, batch_data, batch_data_func, y):
    # Batch Size
    batch_size = 32

    # The Batches
    X_stack, target_stack = batch_data(X=X, y=y)

    # Batches
    X_batches, target_batches = (
        batch_data_func(X_stack, batch_size),
        batch_data_func(target_stack, batch_size),
    )
    return X_batches, target_batches


@app.cell
def _(
    GRUCarry,
    TrainState,
    X_batches,
    hidden_size,
    input_size,
    jax,
    jnp,
    optax,
    output_size,
    target_batches,
    tqdm,
    train_step_factory,
    weights_init,
):
    ## Hyper Paraments
    epochs = 150
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
                X_batches,
                target_batches,
            ),
        )
        train_history.append(aux)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    # _fig = go.Figure()

    # _fig.add_trace(
    #     go.Scatter(
    #         y=jnp.array([h[0].mean() for h in train_history]),
    #         mode="lines",
    #         name="Training Loss",
    #         # line=dict(color="#378ADD", width=1.5),
    #     )
    # )

    # _fig.update_layout(
    #     title=dict(text="<b>GRU Training Loss</b>", x=0.5, font={"size": 22}),
    #     xaxis_title="<b>Step</b>",
    #     yaxis_title="<b>MSE Loss</b>",
    #     hovermode="x unified",
    #     plot_bgcolor="rgba(0,0,0,0)",
    #     paper_bgcolor="rgba(0,0,0,0)",
    # )

    # _fig.show()
    return


if __name__ == "__main__":
    app.run()
