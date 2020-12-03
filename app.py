import dash
from dash_bootstrap_components import themes

external_stylesheet = [themes.JOURNAL]

app = dash.Dash(__name__, assets_ignore='.*bootstrap-journal.css.*', external_stylesheets=external_stylesheet)
server = app.server
app.config.suppress_callback_exceptions = True