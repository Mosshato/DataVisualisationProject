import math
from pathlib import Path
import numpy as np
import pandas as pd

from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


# ---------------- Paths (adjust if needed) ----------------
# PlotDashboard/tabs/model_tab.py -> parents[2] = DVA_Final_Project root
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "Model" / "ss_model.keras"
TRAIN_ML_CSV = ROOT / "Dataset" / "Processed" / "drug_consumption_processed_ml.csv"

TARGET_COL = "SS"
ID_COL = "ID"

# columns to encode for model input (your list)
COLUMNS_TO_ENCODE_M = [
    "Age",
    "Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis",
    "Choc", "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine",
    "LSD", "Meth", "Mushrooms", "Nicotine", "VSA"
]

NUMERIC_INPUTS = ["Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive"]

DRUG_STATUS_LABELS = [
    "Never Used",
    "Used over a Decade Ago",
    "Used in Last Decade",
    "Used in Last Year",
    "Used in Last Month",
    "Used in Last Week",
    "Used in Last Day",
]

AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

def ss_to_percent(ss: float, lo: float = -3.0, hi: float = 3.0) -> float:
    # clamp
    ss = max(lo, min(hi, float(ss)))
    return (ss - lo) / (hi - lo) * 100.0

# ---------------- Binary encoding (exactly your function) ----------------
def encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    from math import ceil, log2

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found for one-hot: {missing}")
    out = df.copy()
    for col in cols:
        s = out[col].astype("object").where(out[col].notna(), "__MISSING__")
        codes, uniques = pd.factorize(s, sort=True)

        k = len(uniques)
        n_bits = max(1, ceil(log2(k)))

        for bit in range(n_bits):
            out[f"{col}_b{bit}"] = ((codes >> bit) & 1).astype("int8")

        out.drop(columns=[col], inplace=True)

    return out


# ---------------- Training schema + category codebooks ----------------
_model = None
_model_err = None

_feature_names: list[str] = []
_codebooks: dict[str, dict[str, int]] = {}     # col -> label -> code
_nbits: dict[str, int] = {}                    # col -> n_bits


def _load_model_once():
    global _model, _model_err
    if _model is not None or _model_err is not None:
        return
    try:
        from tensorflow import keras
        _model = keras.models.load_model(MODEL_PATH)
    except Exception as e:
        _model_err = str(e)


def _init_from_training_csv():
    """
    Read the ML training CSV to:
    - get feature order (all columns except ID and target)
    - build factorize(sort=True) codebooks for encoded columns
    - infer n_bits per encoded column from training schema (e.g., Age_b0..Age_b2)
    """
    global _feature_names, _codebooks, _nbits

    if not TRAIN_ML_CSV.exists():
        raise FileNotFoundError(f"Training ML CSV not found: {TRAIN_ML_CSV}")

    df = pd.read_csv(TRAIN_ML_CSV)

    # model input features are: drop target + id (same as your prepare_data)
    drop_cols = [TARGET_COL]
    if ID_COL in df.columns:
        drop_cols.append(ID_COL)
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    _feature_names = list(feature_df.columns)

    # infer n_bits for each encoded column from existing feature names
    for col in COLUMNS_TO_ENCODE_M:
        bits = []
        prefix = f"{col}_b"
        for c in _feature_names:
            if c.startswith(prefix):
                try:
                    bits.append(int(c.replace(prefix, "")))
                except:
                    pass
        if bits:
            _nbits[col] = max(bits) + 1
        else:
            # fallback: if training schema doesn't have col_b*, skip
            _nbits[col] = 0

    # build codebooks using the ORIGINAL categorical columns from training raw before encoding.
    # If your TRAIN_ML_CSV already contains encoded columns and not raw columns,
    # we can't rebuild codes from it. In that case, set a separate RAW/processed-with-labels CSV.
    # However, in your project you still have decoded labels in dashboard csv, so we use that.
    #
    # Here: assume TRAIN_ML_CSV contains raw categorical labels for Age/drugs (before encode).
    # If it doesn't, see note below.
    for col in COLUMNS_TO_ENCODE_M:
        if col not in df.columns:
            continue
        s = df[col].astype("object").where(df[col].notna(), "__MISSING__")
        # factorize(sort=True) like in encode()
        codes, uniques = pd.factorize(s, sort=True)
        # build mapping label -> code
        cb = {str(u): int(i) for i, u in enumerate(list(uniques))}
        _codebooks[col] = cb

    # sanity: ensure numeric inputs exist
    for c in NUMERIC_INPUTS:
        if c not in _feature_names:
            # ok if model didn't use all, but usually they exist
            pass


# initialize once at import
_load_model_once()
try:
    _init_from_training_csv()
except Exception as e:
    # store error; show in UI
    _model_err = _model_err or str(e)


# ---------------- UI: Sidebar (Model Inputs) ----------------
def sidebar_layout(df_dashboard: pd.DataFrame) -> html.Div:
    # infer slider ranges from dashboard df if possible
    def _range(col, fallback=(-3.5, 3.5)):
        if df_dashboard is not None and col in df_dashboard.columns:
            s = pd.to_numeric(df_dashboard[col], errors="coerce").dropna()
            if len(s) > 0:
                mn, mx = float(s.min()), float(s.max())
                if mn == mx:
                    return mn - 1.0, mx + 1.0
                return mn, mx
        return fallback

    score_controls = []
    for col in NUMERIC_INPUTS:
        mn, mx = _range(col)
        score_controls.append(
            html.Div(
                [
                    html.Label(col),
                    dcc.Slider(
                        id=f"model-{col}",
                        min=mn, max=mx,
                        step=(mx - mn) / 200.0 if (mx - mn) > 0 else 0.01,
                        value=float(np.clip(0.0, mn, mx)),
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(style={"height": "10px"}),
                ]
            )
        )

    drug_controls = []
    for d in COLUMNS_TO_ENCODE_M[1:]:
        drug_controls.append(
            html.Div(
                [
                    html.Label(d),
                    dcc.Dropdown(
                        id=f"model-{d}",
                        options=[{"label": x, "value": x} for x in DRUG_STATUS_LABELS],
                        value="Never Used",
                        clearable=False,
                    ),
                    html.Div(style={"height": "10px"}),
                ]
            )
        )

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Model Inputs", className="sidebar-section-title"),
                html.Div("Psychometrics scores", className="sidebar-subtitle"),
                *score_controls,

                html.Hr(),

                html.Div("Age", className="sidebar-subtitle"),
                dcc.Dropdown(
                    id="model-Age",
                    options=[{"label": x, "value": x} for x in AGE_LABELS],
                    value="25-34",
                    clearable=False,
                ),

                html.Hr(),

                html.Div("Drug consumption classes", className="sidebar-subtitle"),
                *drug_controls,

                dbc.Button("Predict SS", id="model-predict-btn", color="primary", className="w-100"),
            ]
        ),
        className="sidebar-card",
    )


# ---------------- UI: Main content ----------------
def layout(df_dashboard: pd.DataFrame) -> html.Div:
    err = None
    if _model_err is not None:
        err = _model_err

    controls = sidebar_layout(df_dashboard)  # REUSE: conține sliders + dropdowns + predict button

    output_panel = dbc.Card(
        dbc.CardBody(
            [
                html.Div("Prediction output", className="section-title"),
                html.Div(id="model-pred-text", className="section-placeholder"),
                html.Div(style={"height": "10px"}),
                dcc.Graph(id="model-gauge"),
                html.Div(style={"height": "10px"}),
                dbc.Collapse(
                    [
                        html.Div("Feature vector (debug)", className="section-title"),
                        html.Pre(id="model-feature-debug", style={"whiteSpace": "pre-wrap"}),
                    ],
                    id="model-debug-collapse",
                    is_open=False,
                ),
                dbc.Button("Toggle debug", id="model-toggle-debug", color="secondary", outline=True, size="sm"),
            ]
        ),
        className="panel-card",
    )

    description_panel = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                "As drug use has increased significantly in recent years among young people, "
                "this application proposes an AI model that helps anyone get an approximate idea of "
                "how prone a person might be to developing a future addiction of any type.\n\n"
                "After selecting all the parameters, you will see an approximate score.\n\n"
                "Warning! This model provides only an approximation, without any medical certification and without specialists behind it. "
                "For any questions related to drug addiction, please contact qualified professionals!",
                className="section-placeholder",
                style={"whiteSpace": "pre-line"},
            ),
            ]
        ),
        className="panel-card",
    )

    return html.Div(
        [
            html.Div("Model", className="section-title"),
            html.Div(style={"height": "10px"}),

            dbc.Row(
                [
                    dbc.Col(controls, md=4),
                    dbc.Col(
                        [
                            description_panel,
                            html.Div(style={"height": "14px"}),
                            output_panel,
                            (dbc.Alert(err, color="danger") if err else html.Div()),
                        ],
                        md=8,
                    ),
                ],
                className="g-3",
            ),
        ]
    )

# ---------------- Helpers: build input vector exactly like training ----------------
def _encode_single_row_with_codebooks(raw_row: dict) -> pd.DataFrame:
    """
    Build a 1-row dataframe with raw columns, then binary-encode them exactly like encode(),
    but using training-derived codebooks for stable mapping.

    raw_row contains:
      - NUMERIC_INPUTS
      - Age label
      - drugs labels
    """
    # start with numeric columns in df
    base = {}
    for c in NUMERIC_INPUTS:
        base[c] = float(raw_row.get(c, 0.0))

    # include categorical raw columns so we can encode them
    base["Age"] = raw_row.get("Age", "25-34")
    for d in COLUMNS_TO_ENCODE_M[1:]:
        base[d] = raw_row.get(d, "Never Used")

    df1 = pd.DataFrame([base])

    # Now, instead of calling encode() directly (which would re-factorize on 1 row),
    # we reproduce its bit logic using training codebooks + inferred n_bits.
    out = df1.copy()

    for col in COLUMNS_TO_ENCODE_M:
        if _nbits.get(col, 0) <= 0:
            # nothing to encode for this col in training schema
            if col in out.columns:
                out.drop(columns=[col], inplace=True)
            continue

        label = str(out.loc[0, col]) if col in out.columns else "__MISSING__"
        if label is None or label == "nan":
            label = "__MISSING__"

        cb = _codebooks.get(col, {})
        if label not in cb:
            # fallback if unseen: map to "__MISSING__" if exists, else 0
            code = cb.get("__MISSING__", 0)
        else:
            code = cb[label]

        n_bits = _nbits[col]
        for bit in range(n_bits):
            out[f"{col}_b{bit}"] = ((code >> bit) & 1)

        out.drop(columns=[col], inplace=True)

    # align to training feature order
    for c in _feature_names:
        if c not in out.columns:
            out[c] = 0

    out = out[_feature_names].astype(np.float32)
    return out


def _gauge(percent: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(percent),
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}},
            title={"text": "Addiction Sensitivity (0–100%)"},
        )
    )
    fig.update_layout(height=320, margin=dict(l=30, r=30, t=60, b=30), template="plotly_dark")
    return fig


# ---------------- Callbacks ----------------
def register_callbacks(app, df_dashboard: pd.DataFrame) -> None:
    @app.callback(
        Output("model-debug-collapse", "is_open"),
        Input("model-toggle-debug", "n_clicks"),
        State("model-debug-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_debug(n, is_open):
        return not is_open

    @app.callback(
        Output("model-pred-text", "children"),
        Output("model-gauge", "figure"),
        Output("model-feature-debug", "children"),
        Input("model-predict-btn", "n_clicks"),
        State("model-Age", "value"),
        *(State(f"model-{c}", "value") for c in NUMERIC_INPUTS),
        *(State(f"model-{d}", "value") for d in COLUMNS_TO_ENCODE_M[1:]),
        prevent_initial_call=True,
    )
    def _predict(n_clicks, age_label, *rest):
        if _model is None:
            fig = _gauge(0.0)
            return "Model not loaded.", fig, ""

        # unpack states
        num_vals = rest[: len(NUMERIC_INPUTS)]
        drug_vals = rest[len(NUMERIC_INPUTS):]

        raw = {"Age": age_label}
        for i, c in enumerate(NUMERIC_INPUTS):
            raw[c] = float(num_vals[i]) if num_vals[i] is not None else 0.0

        for i, d in enumerate(COLUMNS_TO_ENCODE_M[1:]):
            raw[d] = drug_vals[i] if drug_vals[i] is not None else "Never Used"

        # build feature vector aligned to training
        Xdf = _encode_single_row_with_codebooks(raw)
        X = Xdf.to_numpy(dtype=np.float32)

        # predict
        yhat = _model.predict(X, verbose=0)
        # handle keras outputs
        pred = float(np.ravel(yhat)[0])
        pct = ss_to_percent(pred)

        text = f"Predicted sensitivity: {pct:.1f}% (raw SS: {pred:.4f})"
        fig = _gauge(pct)


        debug = Xdf.iloc[0].to_string()
        return text, fig, debug
