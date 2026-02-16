import pandas as pd
import numpy as np

from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# --- columns (keep local, simple) ---
CATEGORICAL = ["Age", "Gender", "Education", "Country"]
NUMERIC = ["Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "SS"]
DRUGS = [
    "Alcohol","Amphet","Amyl","Benzos","Caff","Cannabis","Choc","Coke","Crack",
    "Ecstasy","Heroin","Ketamine","LSD","Meth","Mushrooms","Nicotine","VSA"
]

# order for drug status (best-effort)
STATUS_ORDER = [
    "Never Used",
    "Used in Last Decade",
    "Used in Last Year",
    "Used in Last Month",
    "Used in Last Week",
    "Used in Last Day",
]


def _filter_df(df: pd.DataFrame, ages, genders, educs, countries) -> pd.DataFrame:
    dff = df.copy()
    if ages and "Age" in dff.columns:
        dff = dff[dff["Age"].astype(str).isin([str(x) for x in ages])]
    if genders and "Gender" in dff.columns:
        dff = dff[dff["Gender"].astype(str).isin([str(x) for x in genders])]
    if educs and "Education" in dff.columns:
        dff = dff[dff["Education"].astype(str).isin([str(x) for x in educs])]
    if countries and "Country" in dff.columns:
        dff = dff[dff["Country"].astype(str).isin([str(x) for x in countries])]
    return dff


def layout(df: pd.DataFrame) -> html.Div:
    # keep both sections in DOM; we just hide/show them => no missing-id callback errors
    return html.Div(
        [
            html.Div("Visualisation & Interaction", className="section-title"),

            dcc.Tabs(
                id="viz-subtabs",
                value="psych",
                children=[
                    dcc.Tab(label="Psychometrics Explorer", value="psych"),
                    dcc.Tab(label="Consumption Patterns", value="patterns"),
                ],
            ),

            html.Div(style={"height": "10px"}),

            # --- Psychometrics section ---
            html.Div(
                id="psych-section",
                children=[
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Psychometrics Controls", className="section-title"),
                                dbc.Row(
                                    [
                                        dbc.Col(dcc.Dropdown(id="psych-x", clearable=False), md=4),
                                        dbc.Col(dcc.Dropdown(id="psych-y", clearable=False), md=4),
                                        dbc.Col(dcc.Dropdown(id="psych-color", clearable=False), md=4),
                                    ],
                                    className="g-2",
                                ),
                                html.Div(style={"height": "10px"}),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.RadioItems(
                                                id="psych-scope",
                                                options=[
                                                    {"label": "All data", "value": "all"},
                                                    {"label": "Selected only", "value": "selected"},
                                                ],
                                                value="all",
                                                inline=True,
                                            ),
                                            md=12,
                                        )
                                    ]
                                ),
                            ]
                        ),
                        className="panel-card",
                    ),
                    html.Div(style={"height": "14px"}),

                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="psych-scatter"), md=6),
                            dbc.Col(dcc.Graph(id="psych-parcoords"), md=6),
                        ],
                        className="g-3",
                    ),

                    html.Div(style={"height": "14px"}),

                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="psych-violin"), md=6),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div("Selection summary", className="section-title"),
                                            html.Div(id="psych-summary", className="section-placeholder"),
                                        ]
                                    ),
                                    className="panel-card",
                                ),
                                md=6,
                            ),
                        ],
                        className="g-3",
                    ),
                ],
                style={"display": "block"},
            ),

            # --- Patterns section ---
            html.Div(
                id="patterns-section",
                children=[
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Patterns Controls", className="section-title"),
                                dbc.Row(
                                    [
                                        dbc.Col(dcc.Dropdown(id="pat-drug-a", clearable=False), md=4),
                                        dbc.Col(dcc.Dropdown(id="pat-drug-b", clearable=False), md=4),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="pat-metric",
                                                options=[
                                                    {"label": "Count", "value": "count"},
                                                    {"label": "Percent", "value": "percent"},
                                                ],
                                                value="count",
                                                clearable=False,
                                            ),
                                            md=4,
                                        ),
                                    ],
                                    className="g-2",
                                ),
                            ]
                        ),
                        className="panel-card",
                    ),
                    html.Div(style={"height": "14px"}),

                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="pat-sankey"), md=6),
                            dbc.Col(dcc.Graph(id="pat-heatmap"), md=6),
                        ],
                        className="g-3",
                    ),

                    html.Div(style={"height": "14px"}),

                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="pat-stacked"), md=6),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div("Pattern summary", className="section-title"),
                                            html.Div(id="pat-summary", className="section-placeholder"),
                                        ]
                                    ),
                                    className="panel-card",
                                ),
                                md=6,
                            ),
                        ],
                        className="g-3",
                    ),
                ],
                style={"display": "none"},
            ),
        ]
    )


def register_callbacks(app, df: pd.DataFrame) -> None:
    template = "plotly_dark"
    H = 360
    M = dict(l=40, r=20, t=60, b=40)

    # show/hide sub-sections (both exist in DOM)
    @app.callback(
        Output("psych-section", "style"),
        Output("patterns-section", "style"),
        Input("viz-subtabs", "value"),
    )
    def _toggle_sections(v):
        if v == "patterns":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    # init dropdown options + defaults when tab-viz is active
    @app.callback(
        Output("psych-x", "options"),
        Output("psych-y", "options"),
        Output("psych-color", "options"),
        Output("psych-x", "value"),
        Output("psych-y", "value"),
        Output("psych-color", "value"),
        Output("pat-drug-a", "options"),
        Output("pat-drug-b", "options"),
        Output("pat-drug-a", "value"),
        Output("pat-drug-b", "value"),
        Input("main-tabs", "value"),
    )
    def _init_viz_options(main_tab):
        if main_tab != "tab-viz":
            return [], [], [], None, None, None, [], [], None, None

        num_opts = [{"label": c, "value": c} for c in NUMERIC if c in df.columns]
        cat_opts = [{"label": c, "value": c} for c in CATEGORICAL if c in df.columns]

        x0 = "Nscore" if "Nscore" in df.columns else (num_opts[0]["value"] if num_opts else None)
        y0 = "Escore" if "Escore" in df.columns else (num_opts[0]["value"] if num_opts else None)
        c0 = "Gender" if "Gender" in df.columns else (cat_opts[0]["value"] if cat_opts else None)

        drug_opts = [{"label": d, "value": d} for d in DRUGS if d in df.columns]
        da = "Cannabis" if "Cannabis" in df.columns else (drug_opts[0]["value"] if drug_opts else None)
        db = "Nicotine" if "Nicotine" in df.columns else (drug_opts[1]["value"] if len(drug_opts) > 1 else da)

        return num_opts, num_opts, cat_opts, x0, y0, c0, drug_opts, drug_opts, da, db

    # --- Psychometrics: brushing + linking ---
    @app.callback(
        Output("psych-scatter", "figure"),
        Output("psych-parcoords", "figure"),
        Output("psych-violin", "figure"),
        Output("psych-summary", "children"),
        Input("eda-filter-age", "value"),
        Input("eda-filter-gender", "value"),
        Input("eda-filter-education", "value"),
        Input("eda-filter-country", "value"),
        Input("psych-x", "value"),
        Input("psych-y", "value"),
        Input("psych-color", "value"),
        Input("psych-scope", "value"),
        Input("psych-scatter", "selectedData"),
    )
    def _update_psych(ages, genders, educs, countries, xcol, ycol, color_by, scope, selectedData):
        dff = _filter_df(df, ages, genders, educs, countries)

        if dff.empty or not xcol or not ycol:
            fig = px.scatter(template=template, title="No data")
            fig.update_layout(height=H, margin=M)
            return fig, fig, fig, "No data after filters."

        # add stable row id for selection linking
        dff = dff.reset_index(drop=False).rename(columns={"index": "_rowid"})

        scatter = px.scatter(
            dff,
            x=xcol,
            y=ycol,
            color=color_by if color_by in dff.columns else None,
            template=template,
            title=f"{xcol} vs {ycol}",
            custom_data=["_rowid"],
        )
        scatter.update_layout(height=H, margin=M)

        # resolve selected rowids
        selected_rowids = set()
        if selectedData and "points" in selectedData:
            for p in selectedData["points"]:
                cd = p.get("customdata")
                if cd and len(cd) > 0:
                    selected_rowids.add(cd[0])

        sel = dff[dff["_rowid"].isin(selected_rowids)] if selected_rowids else dff.iloc[0:0]
        view_df = sel if (scope == "selected" and not sel.empty) else dff

        # parallel coordinates on the view_df
        par_cols = [c for c in NUMERIC if c in view_df.columns]
        if len(par_cols) >= 2 and not view_df.empty:
            par = px.parallel_coordinates(
                view_df,
                dimensions=par_cols,
                template=template,
                title="Parallel Coordinates",
            )
        else:
            par = px.scatter(template=template, title="Parallel Coordinates (need numeric columns)")
        par.update_layout(height=H, margin=M)

        # violin: compare distribution of xcol by a category (prefer Gender if exists, else color_by)
        group_col = "Gender" if "Gender" in dff.columns else (color_by if color_by in dff.columns else None)
        if group_col:
            vio = px.violin(
                view_df,
                x=group_col,
                y=xcol,
                box=True,
                points="outliers",
                template=template,
                title=f"Violin of {xcol} by {group_col} ({'selected' if scope=='selected' else 'all'})",
            )
        else:
            vio = px.violin(view_df, y=xcol, box=True, points="outliers", template=template, title=f"Violin of {xcol}")
        vio.update_layout(height=H, margin=M)

        # summary
        n_all = len(dff)
        n_sel = len(sel)
        s = f"Rows (filtered): {n_all:,}. Selected: {n_sel:,}."
        if not view_df.empty:
            s2 = []
            for c in [xcol, ycol]:
                if c in view_df.columns:
                    s2.append(f"{c}: mean={view_df[c].mean():.3f}, std={view_df[c].std():.3f}")
            if s2:
                s += " " + " | ".join(s2)

        return scatter, par, vio, s

    # --- Patterns: Sankey + heatmap + stacked bar (linked to filters) ---
    @app.callback(
        Output("pat-sankey", "figure"),
        Output("pat-heatmap", "figure"),
        Output("pat-stacked", "figure"),
        Output("pat-summary", "children"),
        Input("eda-filter-age", "value"),
        Input("eda-filter-gender", "value"),
        Input("eda-filter-education", "value"),
        Input("eda-filter-country", "value"),
        Input("pat-drug-a", "value"),
        Input("pat-drug-b", "value"),
        Input("pat-metric", "value"),
    )
    def _update_patterns(ages, genders, educs, countries, drug_a, drug_b, metric):
        dff = _filter_df(df, ages, genders, educs, countries)

        if dff.empty or not drug_a or not drug_b or drug_a not in dff.columns or drug_b not in dff.columns:
            fig = px.scatter(template=template, title="No data")
            fig.update_layout(height=H, margin=M)
            return fig, fig, fig, "No data after filters."

        # --- Sankey: status(drug_a) -> status(drug_b) ---
        a = dff[drug_a].astype(str).fillna("Missing")
        b = dff[drug_b].astype(str).fillna("Missing")
        flow = pd.crosstab(a, b)

        # order rows/cols (best-effort)
        def _ordered(idx):
            items = list(idx)
            ordered = [x for x in STATUS_ORDER if x in items]
            leftovers = sorted([x for x in items if x not in ordered])
            return ordered + leftovers

        row_labels = _ordered(flow.index)
        col_labels = _ordered(flow.columns)
        flow = flow.reindex(index=row_labels, columns=col_labels, fill_value=0)

        # build sankey nodes (left statuses + right statuses)
        left = [f"A: {x}" for x in flow.index.tolist()]
        right = [f"B: {x}" for x in flow.columns.tolist()]
        labels = left + right

        sources, targets, values = [], [], []
        for i, r in enumerate(flow.index):
            for j, c in enumerate(flow.columns):
                v = int(flow.loc[r, c])
                if v > 0:
                    sources.append(i)
                    targets.append(len(left) + j)
                    values.append(v)

        sankey = go.Figure(
            data=[
                go.Sankey(
                    node=dict(label=labels, pad=12, thickness=14),
                    link=dict(source=sources, target=targets, value=values),
                )
            ]
        )
        sankey.update_layout(template=template, height=H, margin=M, title=f"Sankey: {drug_a} → {drug_b}")

        # --- Heatmap: Age x status(drug_a) ---
        if "Age" in dff.columns:
            heat_df = pd.crosstab(dff["Age"].astype(str), dff[drug_a].astype(str))
            heat_df = heat_df.reindex(index=sorted(heat_df.index), columns=_ordered(heat_df.columns), fill_value=0)

            if metric == "percent":
                denom = heat_df.sum(axis=1).replace(0, np.nan)
                heat_show = (heat_df.T / denom).T * 100.0
                heat_title = f"Heatmap: Age × {drug_a} (%)"
            else:
                heat_show = heat_df
                heat_title = f"Heatmap: Age × {drug_a}"

            heat = px.imshow(heat_show, template=template, title=heat_title, aspect="auto")
        else:
            heat = px.scatter(template=template, title="Heatmap needs Age")
        heat.update_layout(height=H, margin=M)

        # --- Stacked bar: Country composition of status(drug_a) (Top 10 countries) ---
        if "Country" in dff.columns:
            top_c = dff["Country"].astype(str).value_counts().head(10).index.tolist()
            tmp = dff[dff["Country"].astype(str).isin(top_c)].copy()
            tmp["Country"] = tmp["Country"].astype(str)
            tmp["Status"] = tmp[drug_a].astype(str)

            if metric == "percent":
                g = tmp.groupby(["Country", "Status"]).size().reset_index(name="count")
                tot = g.groupby("Country")["count"].transform("sum").replace(0, np.nan)
                g["value"] = (g["count"] / tot) * 100.0
                bar_title = f"Top Countries: {drug_a} status (%)"
                ycol = "value"
            else:
                g = tmp.groupby(["Country", "Status"]).size().reset_index(name="value")
                bar_title = f"Top Countries: {drug_a} status "
                ycol = "value"

            stacked = px.bar(
                g,
                x="Country",
                y=ycol,
                color="Status",
                template=template,
                title=bar_title,
            )
            stacked.update_layout(xaxis_tickangle=-25)
        else:
            stacked = px.scatter(template=template, title="Stacked bar needs Country")
        stacked.update_layout(height=H, margin=M)

        # summary
        s = f"Rows (filtered): {len(dff):,}. Drug A={drug_a}, Drug B={drug_b}. Metric={metric}."
        return sankey, heat, stacked, s
