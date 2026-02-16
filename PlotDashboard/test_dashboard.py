from dash import Dash
import dash_bootstrap_components as dbc
from tabs import model_tab

from Dashboard import build_layout, register_callbacks
from tabs import data_tab, visualisation_tab

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    assets_folder="assets"
)

app.layout = build_layout()

# callbacks globale (tabs navigation etc.)
register_callbacks(app)

# load once
df = data_tab.load_df()

# callbacks EDA (Data tab)
data_tab.register_callbacks(app, df)

# callbacks Viz (Visualisation & Interaction tab)
visualisation_tab.register_callbacks(app, df)


model_tab.register_callbacks(app, df)

if __name__ == "__main__":
    app.run(debug=True)
