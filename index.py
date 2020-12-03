import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from app import server

from apps import home, explore, participants, data

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink('Home', href='/home')),
        dbc.NavItem(dbc.NavLink('Explore', href='/explore')),
        dbc.NavItem(dbc.NavLink('Participants', href='/participants')),
        dbc.NavItem(dbc.NavLink('Data', href='/data'))
    ],
    brand='Science and Society',
    brand_href='home',
    color="primary",
    dark=True,
    fluid=True
)

app.layout = dbc.Container([
    html.Div([
        dcc.Location(id='url', refresh=False),
        navbar,
        html.Div(id='page-content')
    ])
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def navigation(pathname):
    if pathname == '/home':
        return home.layout
    elif pathname == '/explore':
        return explore.layout
    elif pathname == '/participants':
        return participants.layout
    elif pathname == '/data':
        return data.layout
    else:
        return home.layout
    
    
if __name__ == '__main__':
    app.run_server(debug=True)