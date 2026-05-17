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
            week_day_sin=lambda x: np.sin(2 * np.pi * x.loc[:, "timestamp"].dt.weekday / 7),
            week_day_cos=lambda x: np.cos(2 * np.pi * x.loc[:, "timestamp"].dt.weekday / 7),
            month_sin=lambda x: np.sin(2 * np.pi * x.loc[:, "timestamp"].dt.month / 12),
            month_cos=lambda x: np.cos(2 * np.pi * x.loc[:, "timestamp"].dt.month / 12),
            time_sin=lambda x: np.sin(
                2
                * np.pi
                * ((x.loc[:, "timestamp"].dt.hour * 60) + x.loc[:, "timestamp"].dt.minute)
                / (24 * 60)
            ),
            time_cos=lambda x: np.cos(
                2
                * np.pi
                * ((x.loc[:, "timestamp"].dt.hour * 60) + x.loc[:, "timestamp"].dt.minute)
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
        .drop(["timestamp", "WeekStatus", "Day_of_week"], axis=1)
    )

    # Display data
    df
    return (df,)


@app.cell
def _(df, go, make_subplots):
    # Get numeric columns only
    _cols = df.select_dtypes(include='number').columns.tolist()
    _n = len(_cols)
    _fig_hist = make_subplots(rows=(_n + 2) // 3, cols=3, subplot_titles=_cols)
    # ── Histograms ──────────────────────────────────────────
    for _i, _col in enumerate(_cols):
        _row, _col_idx = divmod(_i, 3)
        _fig_hist.add_trace(go.Histogram(x=df[_col].dropna(), name=_col, marker_color='#378ADD', showlegend=False), row=_row + 1, col=_col_idx + 1)
    _fig_hist.update_layout(title=dict(text='<b>Feature Distributions</b>', x=0.5, font={'size': 22}), height=350 * ((_n + 2) // 3), bargap=0.1)
    _fig_hist.show()
    _fig_box = make_subplots(rows=(_n + 2) // 3, cols=3, subplot_titles=_cols)
    for _i, _col in enumerate(_cols):
        _row, _col_idx = divmod(_i, 3)
        _fig_box.add_trace(go.Box(y=df[_col].dropna(), name=_col, marker_color='#E8724A', showlegend=False), row=_row + 1, col=_col_idx + 1)
    _fig_box.update_layout(title=dict(text='<b>Feature Box Plots</b>', x=0.5, font={'size': 22}), height=350 * ((_n + 2) // 3))
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
                ("robust", RobustScaler(with_centering=True), ["NSM", "Usage_kWh"]),
                (
                    "robust-nm",
                    RobustScaler(with_centering=False, quantile_range=(20.0, 80.0)),
                    [
                        "Lagging_Current_Reactive_Power_kVarh",
                        "Leading_Current_Reactive_Power_kVarh",
                    ],
                ),
                ("MaxAbsScaler", MaxAbsScaler(), ["CO2", "Load_Type"]),
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
    _cols = df.select_dtypes(include='number').columns.tolist()
    _n = len(_cols)
    _fig_hist = make_subplots(rows=(_n + 2) // 3, cols=3, subplot_titles=_cols)
    # ── Histograms ──────────────────────────────────────────
    for _i, _col in enumerate(_cols):
        _row, _col_idx = divmod(_i, 3)
        _fig_hist.add_trace(go.Histogram(x=df_tr[_col].dropna(), name=_col, marker_color='#378ADD', showlegend=False), row=_row + 1, col=_col_idx + 1)
    _fig_hist.update_layout(title=dict(text='<b>Feature Distributions</b>', x=0.5, font={'size': 22}), height=350 * ((_n + 2) // 3), bargap=0.1)
    _fig_hist.show()
    _fig_box = make_subplots(rows=(_n + 2) // 3, cols=3, subplot_titles=_cols)
    for _i, _col in enumerate(_cols):
        _row, _col_idx = divmod(_i, 3)
        _fig_box.add_trace(go.Box(y=df_tr[_col].dropna(), name=_col, marker_color='#E8724A', showlegend=False), row=_row + 1, col=_col_idx + 1)
    _fig_box.update_layout(title=dict(text='<b>Feature Box Plots</b>', x=0.5, font={'size': 22}), height=350 * ((_n + 2) // 3))
    # ── Box Plots ───────────────────────────────────────────
    _fig_box.show()
    return


@app.cell
def _(df_tr, jnp):
    # The Time Array
    X = jnp.array(df_tr.drop("Usage_kWh", axis=1).values)
    X.shape
    return (X,)


@app.cell
def _(df_tr, jnp):
    y = jnp.array(df_tr.loc[:, ["Usage_kWh"]].values)
    y.shape
    return (y,)


@app.cell
def _(df_tr, go):
    # The Scafold
    fig1 = go.Figure()

    # Add Trace
    fig1.add_trace(go.Scatter(y=df_tr.Usage_kWh))

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
    input_size = 13  # each timestep is ONE number (the count)
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
        Wh = get_ortho_matrix(jax.random.normal(key_Wh, (hidden_size, hidden_size)), 1.0)
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
            X_batch, target_batch = Xy

            # Carry the result
            (loss, rnn_carry_new), grad = loss_func_grad_val(
                weights,
                hidden_state,
                X_batch,
                target_batch,
            )

            ## Optimization step
            # Update thw sates
            update, opt_state_new = optimizer.update(grad, opt_state, rnn_carry_new.weights)
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
def _(X, jnp, timesteps, tqdm, y):
    # Get Sample Numbers
    num_rows = X.shape[0]

    X_stack = list()
    y_stack = list()
    batches = list()

    # Find the Batches
    for idx in tqdm(range(num_rows - timesteps)):
        batches.append((X[idx : idx + timesteps, :], y[idx + timesteps, :]))
        X_stack.append(X[idx : idx + timesteps, :])
        y_stack.append(y[idx + timesteps, :])

    X_stack = jnp.stack(X_stack)
    y_stack = jnp.stack(y_stack)
    return X_stack, batches, y_stack


@app.cell
def _(
    RNNCarry,
    TrainState,
    X_stack,
    factory_train_step,
    hidden_size,
    init_weights,
    input_size,
    jax,
    jnp,
    optax,
    output_size,
    tqdm,
    y_stack,
):
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
    for epoch in tqdm(range(100)):
        train_state, loss = jax.lax.scan(train_step, init=train_state, xs=(X_stack, y_stack))
        losses.append(loss)
    return h0, losses, train_state


@app.cell
def _(go, jnp, losses):
    fig2 = go.Figure()

    # Raw loss
    fig2.add_trace(
        go.Scatter(
            y=jnp.concat(losses),
            mode="lines",
            name="loss",
            line=dict(color="#378ADD", width=1.5),
        )
    )


    fig2.update_layout(
        title=dict(text="<b>Training Loss</b>", x=0.5, font={"size": 20}),
        xaxis_title="<b>Batch</b>",
        yaxis_title="<b>MSE Loss</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    fig2.show()
    return


@app.cell
def _(RNNCarry, batches, h0, predict, rnn_forward, tqdm, train_state):
    # The predictions
    preds = []

    # Get the Final weights
    final_weights = train_state.rnn_carry.weights

    # Loop Through the bachhes
    for bats in tqdm(batches):
        # Get Predctions
        rnn_carry_new, h_t = rnn_forward(RNNCarry(final_weights, h0), bats[0])

        # predict appended to pred
        preds.append(predict(rnn_carry_new))
    return (preds,)


@app.cell
def _(go, jnp, preds, timesteps, y):
    # Denormalize
    y_pred = jnp.concat(preds)
    y_actual = y[timesteps:, :].ravel()  # align actual to prediction window

    fig3 = go.Figure()

    # Actual
    fig3.add_trace(
        go.Scatter(
            y=y_actual,
            mode="lines",
            name="Actual",
            line=dict(color="#378ADD", width=2),
        )
    )

    # Predicted
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
            text="<b>Actual vs Predicted — Airline Passengers</b>",
            x=0.5,
            font={"size": 20},
        ),
        xaxis_title="<b>Month</b>",
        yaxis_title="<b>Passenger Count</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    fig3.show()
    return


if __name__ == "__main__":
    app.run()
