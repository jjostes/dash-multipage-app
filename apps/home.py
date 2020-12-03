import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

layout = html.Div([
    dbc.Container([
        
        html.Br(),
        
        html.H4(children=['Introduction'], style={'font-family':'sans-serif'}),
        html.Hr(),
        
        html.P('''
            In 2019, the Pew Research Center conducted a survey of 4,464 adults living within households 
            in the United States. Part of their American Trends Panel, the survey measured respondent 
            attitudes regarding a number of topics, from trust in researchers and the scientific process 
            to whether or not scientists should be involved with guiding public policy decisions. 
            This dashboard's purpose is to provide the user with the ability to examine theses trends for themselves.
            '''
              )],
        style={'background-color':'rgba(229, 237, 250, 0.5', 'padding': '5px'})
])