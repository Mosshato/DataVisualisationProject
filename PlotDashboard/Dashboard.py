from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

from tabs import data_tab, visualisation_tab
from tabs import model_tab


def build_layout() -> dbc.Container:
    sidebar = html.Div(
        [
            html.Div("Controls / Parameters", className="sidebar-title"),

            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div("EDA Filters", className="sidebar-section-title"),

                        html.Label("Age"),
                        dcc.Dropdown(id="eda-filter-age", multi=True, placeholder="All"),

                        html.Div(style={"height": "10px"}),

                        html.Label("Gender"),
                        dcc.Dropdown(id="eda-filter-gender", multi=True, placeholder="All"),

                        html.Div(style={"height": "10px"}),

                        html.Label("Education"),
                        dcc.Dropdown(id="eda-filter-education", multi=True, placeholder="All"),

                        html.Div(style={"height": "10px"}),

                        html.Label("Country"),
                        dcc.Dropdown(id="eda-filter-country", multi=True, placeholder="All"),

                        html.Div(style={"height": "10px"}),

                        html.Label("Numeric column"),
                        dcc.Dropdown(id="eda-numeric-col", clearable=False),
                    ]
                ),
                className="sidebar-card",
            ),
        ],
        className="sidebar",
    )

    tabs = dcc.Tabs(
        id="main-tabs",
        value="tab-data",
        children=[
            dcc.Tab(label="Data", value="tab-data", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="Visualisation & Interaction", value="tab-viz", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="Model (Predict)", value="tab-model", className="tab", selected_className="tab--selected"),
        ],
        className="tabs",
    )

    tab_content = html.Div(id="tab-content", className="tab-content")

    content = html.Div(
        [
            html.Div(
                [
                    html.Div("Drug Consumption Dashboard", className="app-title"),
                    html.Div("Tabs + sidebar", className="app-subtitle"),
                ],
                className="header",
            ),
            tabs,
            tab_content,
        ],
        className="content",
    )

    return dbc.Container([sidebar, content], fluid=True, className="app-root")


def register_callbacks(app: Dash) -> None:
    from dash import Output, Input

    df = data_tab.load_df()

    @app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
    def render_tab(tab_value: str):
        if tab_value == "tab-data":
            return data_tab.layout(df)

        if tab_value == "tab-viz":
            return visualisation_tab.layout(df)

        if tab_value == "tab-model":
            return model_tab.layout(df)

        return html.Div("Tab necunoscut.")
