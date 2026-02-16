import pandas as pd
import numpy as np

from dash import html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

# IMPORTANT: pe Windows folosește raw string sau \\ ca să nu se interpreteze \ ca escape
DATA_PATH = r"C:\CORUNA\DataVi\Project\DVA_Final_Project\Dataset\Processed\drug_consumption_processed_dashboard.csv"

CATEGORICAL_COLS = ["Age", "Gender", "Education", "Country"]
NUMERIC_COLS = ["Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "SS"]

DRUG_COLS = [
    "Alcohol","Amphet","Amyl","Benzos","Caff","Cannabis","Choc","Coke","Crack",
    "Ecstasy","Heroin","Ketamine","LSD","Meth","Mushrooms","Nicotine","VSA"
]


def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    expected = ["ID"] + CATEGORICAL_COLS + NUMERIC_COLS + DRUG_COLS
    existing = [c for c in expected if c in df.columns]
    df = df[existing].copy()

    # categorice
    for c in CATEGORICAL_COLS + DRUG_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # numerice
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def kpi_card(title: str, value: str, subtitle: str = ""):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="kpi-title"),
                html.Div(value, className="kpi-value"),
                html.Div(subtitle, className="kpi-subtitle"),
            ]
        ),
        className="kpi-card",
    )


def layout(df: pd.DataFrame) -> html.Div:
    return html.Div(
        [
            html.Div("Data", className="section-title"),
            html.Div(style={"height": "14px"}),

            # KPI row
            dbc.Row(
                [
                    dbc.Col(html.Div(id="kpi-rows"), md=2),
                    dbc.Col(html.Div(id="kpi-cols"), md=2),
                    dbc.Col(html.Div(id="kpi-num"), md=2),
                    dbc.Col(html.Div(id="kpi-cat"), md=2),
                    dbc.Col(html.Div(id="kpi-miss"), md=2),
                    dbc.Col(html.Div(id="kpi-ids"), md=2),
                ],
                className="g-3",
            ),

            html.Div(style={"height": "14px"}),

            # ROW 1: Summary (left) + Histogram (right)
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Summary statistics", className="section-title"),
                                    dash_table.DataTable(
                                        id="eda-stats-table",
                                        page_size=9,
                                        style_table={"overflowX": "auto", "height": "340px", "overflowY": "auto"},
                                        style_header={"fontWeight": "700"},
                                        style_cell={"fontSize": "12px", "padding": "8px"},
                                        sort_action="native",
                                    ),
                                ]
                            ),
                            className="panel-card panel-fixed",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Histogram", className="section-title"),
                                    dcc.Graph(id="eda-hist", className="graph-fixed"),
                                ]
                            ),
                            className="panel-card panel-fixed",
                        ),
                        md=6,
                    ),
                ],
                className="g-3",
            ),

            html.Div(style={"height": "14px"}),

            # ROW 2: Boxplot (left) + Correlation (right)
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Boxplot", className="section-title"),
                                    dcc.Graph(id="eda-box", className="graph-fixed"),
                                ]
                            ),
                            className="panel-card panel-fixed",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Correlation heatmap", className="section-title"),
                                    dcc.Graph(id="eda-corr", className="graph-fixed"),
                                ]
                            ),
                            className="panel-card panel-fixed",
                        ),
                        md=6,
                    ),
                ],
                className="g-3",
            ),

            html.Div(style={"height": "14px"}),

            # ROW 3: Data table full width
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Data preview", className="section-title"),
                                    dash_table.DataTable(
                                        id="eda-data-table",
                                        page_size=10,
                                        filter_action="native",
                                        sort_action="native",
                                        page_action="native",
                                        style_table={"overflowX": "auto", "height": "340px", "overflowY": "auto"},
                                        style_header={"fontWeight": "700"},
                                        style_cell={"fontSize": "12px", "padding": "8px"},
                                    ),
                                ]
                            ),
                            className="panel-card panel-fixed",
                        ),
                        md=12,
                    )
                ],
                className="g-3",
            ),
            html.Div(style={"height": "14px"}),

            # ROW 3: Age distribution + Violin (numeric by Gender)
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Age distribution", className="section-title"),
                                    dcc.Graph(id="eda-age-bar", className="graph-fixed"),
                                ]
                            ),
                            className="panel-card panel-fixed",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Violin: numeric by Gender", className="section-title"),
                                    dcc.Graph(id="eda-violin", className="graph-fixed"),
                                ]
                            ),
                            className="panel-card panel-fixed",
                        ),
                        md=6,
                    ),
                ],
                className="g-3",
            ),

            html.Div(style={"height": "14px"}),

            # ROW 4: Scatter + Education distribution
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Scatter: Nscore vs Escore", className="section-title"),
                                    dcc.Graph(id="eda-scatter", className="graph-fixed"),
                                ]
                            ),
                            className="panel-card panel-fixed",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Education distribution (Top 10)", className="section-title"),
                                    dcc.Graph(id="eda-edu-bar", className="graph-fixed"),
                                ]
                            ),
                            className="panel-card panel-fixed",
                        ),
                        md=6,
                    ),
                ],
                className="g-3",
            ),

            html.Div(style={"height": "14px"}),

        ]
    )


def register_callbacks(app, df: pd.DataFrame) -> None:
    # 1) Populate sidebar dropdown options when Data tab is active
    @app.callback(
        Output("eda-filter-age", "options"),
        Output("eda-filter-gender", "options"),
        Output("eda-filter-education", "options"),
        Output("eda-filter-country", "options"),
        Output("eda-numeric-col", "options"),
        Output("eda-numeric-col", "value"),
        Input("main-tabs", "value"),
    )
    def init_sidebar_options(tab_value):
        if tab_value != "tab-data":
            return [], [], [], [], [], None

        ages = sorted([x for x in df["Age"].dropna().unique()]) if "Age" in df.columns else []
        genders = sorted([x for x in df["Gender"].dropna().unique()]) if "Gender" in df.columns else []
        educs = sorted([x for x in df["Education"].dropna().unique()]) if "Education" in df.columns else []
        countries = sorted([x for x in df["Country"].dropna().unique()]) if "Country" in df.columns else []

        numeric_options = [c for c in df.select_dtypes("number").columns if c != "ID"]
        default_numeric = "Nscore" if "Nscore" in numeric_options else (numeric_options[0] if numeric_options else None)

        return (
            [{"label": str(x), "value": str(x)} for x in ages],
            [{"label": str(x), "value": str(x)} for x in genders],
            [{"label": str(x), "value": str(x)} for x in educs],
            [{"label": str(x), "value": str(x)} for x in countries],
            [{"label": c, "value": c} for c in numeric_options],
            default_numeric
        )

    # 2) Main EDA update callback (filters -> KPIs, tables, plots + extra 4 charts)
    @app.callback(
        Output("kpi-rows", "children"),
        Output("kpi-cols", "children"),
        Output("kpi-num", "children"),
        Output("kpi-cat", "children"),
        Output("kpi-miss", "children"),
        Output("kpi-ids", "children"),
        Output("eda-stats-table", "data"),
        Output("eda-stats-table", "columns"),
        Output("eda-hist", "figure"),
        Output("eda-box", "figure"),
        Output("eda-corr", "figure"),
        Output("eda-age-bar", "figure"),
        Output("eda-violin", "figure"),
        Output("eda-scatter", "figure"),
        Output("eda-edu-bar", "figure"),
        Output("eda-data-table", "data"),
        Output("eda-data-table", "columns"),
        Input("eda-filter-age", "value"),
        Input("eda-filter-gender", "value"),
        Input("eda-filter-education", "value"),
        Input("eda-filter-country", "value"),
        Input("eda-numeric-col", "value"),
    )
    def update_eda(ages, genders, educs, countries, numeric_col):
        dff = df.copy()
        template = "plotly_dark"

        # Fixed sizes for symmetry
        H = 340
        M = dict(l=40, r=20, t=50, b=40)

        # Filters (categorical)
        if ages and "Age" in dff.columns:
            dff = dff[dff["Age"].astype(str).isin([str(x) for x in ages])]
        if genders and "Gender" in dff.columns:
            dff = dff[dff["Gender"].astype(str).isin([str(x) for x in genders])]
        if educs and "Education" in dff.columns:
            dff = dff[dff["Education"].astype(str).isin([str(x) for x in educs])]
        if countries and "Country" in dff.columns:
            dff = dff[dff["Country"].astype(str).isin([str(x) for x in countries])]

        # KPIs
        rows = len(dff)
        cols = dff.shape[1]
        num_cols = len(dff.select_dtypes("number").columns)
        cat_cols = cols - num_cols
        missing_cells = int(dff.isna().sum().sum())
        unique_ids = int(dff["ID"].nunique()) if "ID" in dff.columns else 0

        kpi_rows = kpi_card("Rows", f"{rows:,}", "after filters")
        kpi_cols = kpi_card("Columns", f"{cols:,}", "total")
        kpi_num = kpi_card("Numeric cols", f"{num_cols:,}", "")
        kpi_cat = kpi_card("Categorical cols", f"{cat_cols:,}", "")
        kpi_miss = kpi_card("Missing cells", f"{missing_cells:,}", "after filters")
        kpi_ids = kpi_card("Unique IDs", f"{unique_ids:,}", "after filters")

        # Empty safeguard
        if dff.empty:
            empty_fig = px.scatter(template=template, title="No data after filters")
            empty_fig.update_layout(height=H, margin=M)
            corr_empty = px.imshow(np.zeros((1, 1)), template=template, title="Correlation", aspect="auto")
            corr_empty.update_layout(height=H, margin=M)

            return (
                kpi_rows, kpi_cols, kpi_num, kpi_cat, kpi_miss, kpi_ids,
                [], [],
                empty_fig, empty_fig, corr_empty,
                empty_fig, empty_fig, empty_fig, empty_fig,
                [], [{"name": c, "id": c} for c in df.columns]
            )

        # Summary stats
        num_df = dff.select_dtypes("number")
        if len(num_df.columns) > 0:
            stats = num_df.describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index().rename(columns={"index": "feature"})
            keep = ["feature", "mean", "std", "min", "25%", "50%", "75%", "max"]
            stats = stats[[c for c in keep if c in stats.columns]]
            stats_data = stats.round(4).to_dict("records")
            stats_cols = [{"name": c, "id": c} for c in stats.columns]
        else:
            stats_data, stats_cols = [], []

        # Base charts
        if numeric_col and numeric_col in dff.columns:
            hist_fig = px.histogram(dff, x=numeric_col, nbins=30, template=template, title=f"Histogram: {numeric_col}")
            hist_fig.update_layout(height=H, margin=M)

            box_fig = px.box(dff, y=numeric_col, template=template, title=f"Boxplot: {numeric_col}")
            box_fig.update_layout(height=H, margin=M)
        else:
            hist_fig = px.scatter(template=template, title="Select a numeric column")
            hist_fig.update_layout(height=H, margin=M)

            box_fig = px.scatter(template=template, title="Select a numeric column")
            box_fig.update_layout(height=H, margin=M)

        if len(num_df.columns) > 0:
            corr = num_df.corr(numeric_only=True).round(3)
            corr_fig = px.imshow(corr, template=template, title="Correlation (numeric)", aspect="auto")
            corr_fig.update_layout(height=H, margin=M)
        else:
            corr_fig = px.imshow(np.zeros((1, 1)), template=template, title="Correlation (numeric)", aspect="auto")
            corr_fig.update_layout(height=H, margin=M)

        # EXTRA 4 chart types
        # 1) Age distribution bar
        if "Age" in dff.columns:
            age_counts = dff["Age"].astype(str).value_counts().reset_index()
            age_counts.columns = ["Age", "count"]
            age_bar = px.bar(age_counts, x="Age", y="count", template=template, title="")
        else:
            age_bar = px.scatter(template=template, title="Age not available")
        age_bar.update_layout(height=H, margin=M)

        # 2) Violin: numeric by Gender
        if numeric_col and numeric_col in dff.columns and "Gender" in dff.columns:
            violin_fig = px.violin(
                dff, x="Gender", y=numeric_col, box=True, points="outliers",
                template=template, title=""
            )
        else:
            violin_fig = px.scatter(template=template, title="Need Gender + numeric")
        violin_fig.update_layout(height=H, margin=M)

        # 3) Scatter: Nscore vs Escore
        if "Nscore" in dff.columns and "Escore" in dff.columns:
            scatter_fig = px.scatter(
                dff, x="Nscore", y="Escore",
                color="Gender" if "Gender" in dff.columns else None,
                template=template, title=""
            )
        else:
            scatter_fig = px.scatter(template=template, title="Nscore/Escore not available")
        scatter_fig.update_layout(height=H, margin=M)

        # 4) Education distribution Top 10
        if "Education" in dff.columns:
            edu_counts = dff["Education"].astype(str).value_counts().head(10).reset_index()
            edu_counts.columns = ["Education", "count"]
            edu_bar = px.bar(edu_counts, x="Education", y="count", template=template, title="")
            edu_bar.update_layout(xaxis_tickangle=-25)
        else:
            edu_bar = px.scatter(template=template, title="Education not available")
        edu_bar.update_layout(height=H, margin=M)

        # Data table
        data_cols = [{"name": c, "id": c} for c in dff.columns]
        data_rows = dff.head(500).to_dict("records")

        return (
            kpi_rows, kpi_cols, kpi_num, kpi_cat, kpi_miss, kpi_ids,
            stats_data, stats_cols,
            hist_fig, box_fig, corr_fig,
            age_bar, violin_fig, scatter_fig, edu_bar,
            data_rows, data_cols
        )
