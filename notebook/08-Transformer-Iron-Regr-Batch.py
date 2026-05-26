import marimo

__generated_with = "0.23.8"
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import RobustScaler, MaxAbsScaler
    from sklearn.metrics import (
        r2_score,
        mean_squared_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
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
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        optax,
        partial,
        pd,
        r2_score,
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
        .drop(["timestamp", "WeekStatus", "Day_of_week", "Load_Type"], axis=1)
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
                    ["NSM", "Usage_kWh"],
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
    X = jnp.array(df_tr.drop("Usage_kWh", axis=1).values)
    X.shape
    return (X,)


@app.cell
def _(df_tr, jnp):
    y = jnp.array(df_tr.loc[:, ["Usage_kWh"]].values)
    y.shape
    return (y,)


@app.cell
def _(go, y):
    # The Scafold
    fig1 = go.Figure()

    # Add Trace
    fig1.add_trace(go.Scatter(y=y.ravel()))

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
    input_size: int = 12  # each timestep is ONE number (the count)
    output_size: int = 1  # predicting ONE number (next month)
    timesteps: int = 14  # window size — how far back you look
    batch_size: int = 32  # batch size

    # Attension Dimensions
    head_dim: int = 8  # Dimensioun of the head
    d_model: int = 32  # Model dimensions
    n_heads: int = 4  # N Attension Heads
    ffn_out_1: int = 128
    return (
        batch_size,
        d_model,
        ffn_out_1,
        head_dim,
        input_size,
        n_heads,
        output_size,
        timesteps,
    )


@app.cell
def _(jax, jnp, partial):
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


    @partial(jax.jit, static_argnames=["batch_size"])
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
def _(X, stack_func, timesteps: int, y):
    # Stack the data first
    X_stack, y_stack = stack_func(X=X, y=y, timesteps=timesteps)

    # Number of samples
    n_samples = X_stack.shape[0]
    # The Index of Train cutoff
    train_idx = 3 * (n_samples // 4)

    # Test Split
    X_stack_test = X_stack[train_idx:]
    y_stack_test = y_stack[train_idx:]

    # Train Stack
    X_stack = X_stack[:train_idx]
    y_stack = y_stack[:train_idx]
    return X_stack, X_stack_test, y_stack, y_stack_test


@app.cell
def _(NamedTuple, jax):
    class ModelWeights(NamedTuple):
        # Input Layer
        Wi: jax.Array  # (input_size, d_model)
        bi: jax.Array  # (d_model,)

        # Attension Weights
        W_Q: jax.Array
        W_K: jax.Array
        W_V: jax.Array
        # output Weights
        Wo: jax.Array
        # FFN
        W_ffn1: jax.Array
        W_ffn2: jax.Array
        b_ffn2: jax.Array
        # Final Layer
        W_final: jax.Array
        b_final: jax.Array

    return (ModelWeights,)


@app.cell
def init_weights(ModelWeights, ffn_out_1: int, jax, jnp):
    def init_weights(
        key: jax.random.PRNGKey,
        *,
        input_size: int,
        d_model: int,
        n_heads: int,
        head_dim: int,
        output_size: int,
    ) -> ModelWeights:
        # Split Keys
        (
            key_w1,
            key_wq,
            key_wk,
            key_wv,
            key_wo,
            key_wffn1,
            key_wffn2,
            key_Wfinal,
        ) = jax.random.split(key, num=8)

        # Init Weights
        Wi = jax.random.normal(key_w1, (input_size, d_model)) * jnp.sqrt(
            2 / (input_size + d_model)
        )
        bi = jnp.zeros((d_model,))

        # Attention weights
        W_Q = jax.random.normal(
            key=key_wq, shape=(d_model, n_heads, head_dim)
        ) * jnp.sqrt(2 / (d_model + (n_heads * head_dim)))
        W_K = jax.random.normal(
            key=key_wk, shape=(d_model, n_heads, head_dim)
        ) * jnp.sqrt(2 / (d_model + (n_heads * head_dim)))
        W_V = jax.random.normal(
            key=key_wv, shape=(d_model, n_heads, head_dim)
        ) * jnp.sqrt(2 / (d_model + (n_heads * head_dim)))

        # The Out  weights
        Wo = jax.random.normal(key_wo, (head_dim, n_heads, d_model)) * jnp.sqrt(
            2 / (d_model + (n_heads * head_dim))
        )

        # FFN Layers
        W_ffn1 = jax.random.normal(key_wffn1, (d_model, ffn_out_1)) * jnp.sqrt(
            2 / (d_model + ffn_out_1)
        )
        W_ffn2 = jax.random.normal(key_wffn2, (ffn_out_1, d_model)) * jnp.sqrt(
            2 / (ffn_out_1 + d_model)
        )
        b_ffn2 = jnp.zeros((d_model,))

        # Final output
        W_final = jax.random.normal(key_Wfinal, (d_model, output_size)) * jnp.sqrt(
            2 / (d_model + output_size)
        )
        b_final = jnp.zeros((output_size,))

        # Return the data
        return ModelWeights(
            Wi, bi, W_Q, W_K, W_V, Wo, W_ffn1, W_ffn2, b_ffn2, W_final, b_final
        )


    def positional_encoder(*, timesteps: int, d_model: int) -> jax.Array:
        # The postion vector
        pos = jnp.arange(timesteps)[:, None]

        # The dimension Indices
        i = jnp.arange(d_model)[None, :]

        # Get the denominator
        denominator = jnp.power(10_000, (2 * (i // 2)) / d_model)

        # Get angles
        angles = pos / denominator

        # The Positional encoding
        PE = jnp.where(i % 2 == 0, jnp.sin(angles), jnp.cos(angles))

        # Return Positional Encoder
        return PE

    return init_weights, positional_encoder


@app.cell
def _(ModelWeights, jax, jnp):
    def func_input_projection(
        params: ModelWeights, X_input: jax.Array
    ) -> jax.Array:
        return jnp.einsum("tf, fb -> tb", X_input, params.Wi) + params.bi

    return (func_input_projection,)


@app.cell
def _(ModelWeights, head_dim: int, jax, jnp):
    def func_scaled_dot_prod_attension(
        params: ModelWeights, X_proj: jax.Array
    ) -> jax.Array:
        # 1. Q, K & V Projections
        Q = jnp.einsum("td, dhk -> thk", X_proj, params.W_Q)
        K = jnp.einsum("td, dhk -> thk", X_proj, params.W_K)
        V = jnp.einsum("td, dhk -> thk", X_proj, params.W_V)

        # 2. Scores
        scores = jnp.einsum("qhn, khn -> hqk", Q, K) / jnp.sqrt(head_dim)

        # 3. Weights
        weights = jax.nn.softmax(scores, axis=-1)

        # 4. Weighted Sums
        weighted_sums = jnp.einsum("htk, khn -> thn", weights, V)

        # Return weighted Means
        return weighted_sums


    def func_output_projection(params: ModelWeights, weighted_sums: jax.Array):
        # 5. Output pojections
        return jnp.einsum("thn, nhd -> td", weighted_sums, params.Wo)


    def func_residual(*, out: jax.Array, X_proj: jax.Array) -> jax.Array:
        # 6. Residual
        return out + X_proj


    def func_layer_norm(z: jax.Array) -> jax.Array:
        # 7. Layer Normalization
        ## Layer Means and standard Deviation
        layer_mean = jnp.mean(z, axis=-1, keepdims=True)
        layer_std = jnp.std(z, axis=-1, keepdims=True)

        ## Normaize the layers
        z = (z - layer_mean) / jnp.maximum(layer_std, 1e-6)

        ## Return Layers
        return z


    def func_ffn(model_weights: ModelWeights, z: jax.Array) -> jax.Array:
        out1 = jax.nn.gelu(jnp.einsum("do, td -> to", model_weights.W_ffn1, z))
        out2 = (
            jnp.einsum("do, id -> io", model_weights.W_ffn2, out1)
            + model_weights.b_ffn2
        )
        return out2


    def func_multihead_attension(
        params: ModelWeights, X_proj: jax.Array
    ) -> jax.Array:
        # Generate weighted Sums
        weighted_sums = func_scaled_dot_prod_attension(
            params=params, X_proj=X_proj
        )
        # Output projection
        out = func_output_projection(params=params, weighted_sums=weighted_sums)

        # Residual
        X_proj = func_residual(out=out, X_proj=X_proj)

        # Layer Normalization
        X_proj = func_layer_norm(z=X_proj)

        # Layer Normalization
        return X_proj


    def pred_layer(params: ModelWeights, X_proj: jax.Array) -> jax.Array:
        y_pred = jnp.einsum("do, d -> o", params.W_final, X_proj)
        return y_pred


    def func_mse_loss(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
        return jnp.mean(jnp.square(y_true - y_pred))

    return (
        func_ffn,
        func_layer_norm,
        func_mse_loss,
        func_multihead_attension,
        func_residual,
        pred_layer,
    )


@app.cell
def _(
    Any,
    ModelWeights,
    NamedTuple,
    func_ffn,
    func_input_projection,
    func_layer_norm,
    func_mse_loss,
    func_multihead_attension,
    func_residual,
    jax,
    optax,
    pred_layer,
):
    def predict(
        params: ModelWeights,
        X: jax.Array,
        PE: jax.Array,
    ):
        # The Input Projection
        X_proj = func_input_projection(params, X)

        # Add projections added wtih Positional Encoding
        X_proj = X_proj + PE

        # Muti Head Attension
        X_proj = func_multihead_attension(params, X_proj=X_proj)

        # Residual And Layer Norm Again
        X_proj = func_layer_norm(
            func_residual(out=func_ffn(params, z=X_proj), X_proj=X_proj)
        )

        # Get predictions
        y_pred = pred_layer(params, X_proj=X_proj[-1])

        # Return
        return y_pred


    # Forward Function for Batch
    func_predict_batch = jax.jit(jax.vmap(predict, in_axes=(None, 0, None)))


    @jax.jit
    def forward(
        params: ModelWeights,
        X: jax.Array,
        y: jax.Array,
        PE: jax.Array,
    ) -> jax.Array:
        # predict
        y_pred = func_predict_batch(params, X, PE)
        # loss values
        loss = func_mse_loss(y_true=y, y_pred=y_pred)

        # Return
        return loss


    # Grad of Forward Functions
    gradint_forward = jax.value_and_grad(forward, has_aux=False)


    class TrainState(NamedTuple):
        params: jax.Array
        opt_state: Any


    def get_train_step(optimizer):
        @jax.jit
        def train_step(
            params: ModelWeights,
            DS: tuple[jax.Array, jax.Array],
            PE: jax.Array,
            opt_state: Any,
        ) -> tuple[TrainState, jax.Array]:
            # Split data aset
            X_batch, y_batch = DS

            # Loss & Grad Function
            loss, grad = gradint_forward(params, X_batch, y_batch, PE)

            # Optimizer update state
            updates, opt_state = optimizer.update(grad, opt_state, params=params)
            params = optax.apply_updates(params, updates)

            # return
            return TrainState(params, opt_state), loss

        # Return Function
        return train_step

    return TrainState, get_train_step, predict


@app.cell
def _(
    TrainState,
    X_stack,
    batch_size: int,
    d_model: int,
    get_train_step,
    head_dim: int,
    init_weights,
    input_size: int,
    jax,
    jnp,
    n_heads: int,
    optax,
    output_size: int,
    permute_n_batch_func,
    positional_encoder,
    timesteps: int,
    tqdm,
    y_stack,
):
    # Hyper parameters
    epoch: int = 150
    learning_rate = 1e-3

    # History
    history = []

    # Model Weights
    params = init_weights(
        jax.random.key(84),
        input_size=input_size,
        d_model=d_model,
        n_heads=n_heads,
        head_dim=head_dim,
        output_size=output_size,
    )

    # Get Projection Vector
    PE = positional_encoder(d_model=d_model, timesteps=timesteps)

    # Optimizer
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    # Train Step Function
    train_step = get_train_step(optimizer)
    train_state = TrainState(params, opt_state)

    # Int the Batch Keys
    batch_key = jax.random.key(100)

    for _ in tqdm(range(epoch)):
        # split Keys
        batch_key, _ = jax.random.split(batch_key)

        # Key for spltting & new batch
        X_batch, y_batch = permute_n_batch_func(
            X_stack,
            y_stack,
            batch_size,
            batch_key,
        )

        # The Train Step
        train_state, loss = jax.lax.scan(
            lambda carry, DS: train_step(carry.params, DS, PE, carry.opt_state),
            init=train_state,
            xs=(X_batch, y_batch),
        )

        # Update History
        history.append(loss)

    # Block till over
    history = jnp.array(history)
    _ = jax.block_until_ready(history)
    return PE, history, train_state


@app.cell
def _(go, history):
    _fig = go.Figure()

    _fig.add_trace(
        go.Scatter(
            y=history.mean(axis=1),
            mode="lines",
            name="MSE Training Loss",
            line=dict(color="#378ADD", width=1.5),
        )
    )

    _fig.update_layout(
        title=dict(
            text="<b>Transformer Training Loss</b>", x=0.5, font={"size": 22}
        ),
        xaxis_title="<b>Epoch</b>",
        yaxis_title="<b>MSE Loss</b>",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    _fig.show()
    return


@app.cell
def _(PE, X_stack_test, go, jax, predict, train_state, y_stack_test):
    # Get Predictions & True values flattened
    y_pred = jax.vmap(predict, in_axes=(None, 0, None))(
        train_state.params, X_stack_test, PE
    ).ravel()
    y_true = y_stack_test.ravel()

    # Get Figure
    _fig = go.Figure()

    _fig.add_trace(
        go.Scatter(
            y=y_true,
            mode="lines",
            name="Actual",
            line=dict(color="#378ADD", width=2),
        )
    )

    _fig.add_trace(
        go.Scatter(
            y=y_pred,
            mode="lines",
            name="Predicted",
            line=dict(color="#E8724A", width=2, dash="dash"),
        )
    )

    _fig.update_layout(
        title=dict(
            text="<b>Transformers — Actual vs Predicted</b>",
            x=0.5,
            font={"size": 22},
        ),
        xaxis_title="<b>Step</b>",
        yaxis_title="<b>KWh Power Consumed</b>",
        hovermode="x unified",
    )

    _fig.show()
    return y_pred, y_true


@app.cell
def _(mean_absolute_error, mean_squared_error, r2_score, y_pred, y_true):
    # Regression Metrics
    print(f"R2  = {r2_score(y_true=y_true, y_pred=y_pred)}")
    print(f"MSE = {mean_squared_error(y_true=y_true, y_pred=y_pred)}")
    print(f"MAE = {mean_absolute_error(y_true=y_true, y_pred=y_pred)}")
    return


if __name__ == "__main__":
    app.run()
