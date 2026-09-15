import os

import dash
from dash import Dash, dcc, html

app = Dash(__name__, use_pages=True, title="Car Price Classifier")
server = app.server

app.layout = html.Div([
    html.Nav([
        html.Span("ST127314 - A3", className="nav-brand"),
        html.Div([
            dcc.Link(page["name"], href=page["path"], className="nav-link")
            for page in dash.page_registry.values()
        ], className="nav-links"),
    ], className="navbar"),
    dash.page_container,
])

if __name__ == "__main__":
    app.run(debug=False, host=os.environ.get("HOST", "127.0.0.1"), port=8050, use_reloader=False)

