from app import app
from app import server
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
from scipy import stats
import pyreadstat
import re



# load data
fpath = 'data/ATP W42.sav'

df, meta = pyreadstat.read_sav(fpath)

df_copy = pyreadstat.pyreadstat.set_value_labels(df, meta)


""" 
-----------------------------
DATA CLEANING, TRANSFORMATION
-----------------------------
"""

# helper function used to sort survey items according to their thematic subject matter code (e.g. starts with 'RQ')
def list_helper(theme_code):
    return [i for i in df.columns if theme_code in i]


society = ['PAST_W42', 'FUTURE_W42', 'SC1_W42']
policy = list_helper('POLICY')
confidence = list_helper('CONF')
rq_form1 = list_helper('RQ')
pq_form2 = list_helper('PQ')
scm4 = list_helper('SCM4')
scm5 = list_helper('SCM5')
q = [i for i in df.columns if re.search("^Q[0-9]", i)] #regex to grab Q6, Q7, etc.
pop = list_helper('POP')
knowledge = list_helper('KNOW')
demographics = list_helper('F_')
weight = ['WEIGHT_W42']



# The dictionary meta.column_names_to_labels repeats the key at the start of the value string.
# e.g. key = 'PAST_W42'; value = 'PAST_W42. Compared with twenty years ago...'
# This for loop removes the substring 'PAST_W42' from the beginning of the value string. 

for key, value in meta.column_names_to_labels.items():
    meta.column_names_to_labels[key] = re.sub(pattern='.+\.\s?', string=value, repl='')

    
# within the same dictionary, the following string (saved as a regex pattern) repeats for each CONF item.
# this for loop removes 'pattern' in order to make for easier reading later on

pattern = '^How much confidence, if any, do you have in each of the following to act in the best interests of the public\?\s'

for key, value in meta.column_names_to_labels.items():
    if key in confidence:
        meta.column_names_to_labels[key] = re.sub(pattern=pattern, string=value, repl='')
        
        
        
# For certain columns, ordinal values didn't follow a spectrum of good to bad; agree to disagree
# Here we collect these columns, and use a for loop to switch 'Worse' from 2.0 to 3.0
# The values now read {1.0: 'Better', 3.0: 'Worse', 2.0: 'About the same', 99.0: 'Refused'}

rq_pq = rq_form1 + pq_form2

ordinals_to_switch = [i for i in rq_pq if re.search("^(P|R)Q(1)", i)]
ordinals_to_switch = ordinals_to_switch + society + q + ['POLICY3_W42']

for col_name in ordinals_to_switch:
    df[col_name] = df[col_name].map(lambda x: 2.0 if x == 3.0 else (3.0 if x == 2.0 else x))
    
    
    
# To directly edit the dictionary values of meta.variable_values_labels, it was copied as variable 'meta_vvl' to make more readable
# The dict object is still stored at the same memory location as the variable, so values _2, _3 are used to switch 2.0 to 3.0 and vice versa. Otherwise the elif statement wouldn't change due key 2.0 equalling key 3.0

meta_vvl = meta.variable_value_labels.copy()

for col_name in ordinals_to_switch:
    
    value_2 = meta_vvl[col_name][2.0]
    value_3 = meta_vvl[col_name][3.0]

    for k, v in meta_vvl[col_name].items():
        
        if k == 2.0:
            meta_vvl[col_name][2.0] = value_3
            
        elif k == 3.0:
            meta_vvl[col_name][3.0] = value_2
            
# dictionary of column names to be used with the dcc.Dropdown() property 'options'
demo_dropdown = [{'label': v, 'value': k} for k,v in meta.column_names_to_labels.items() if k in demographics]


# labels to be used with the theme selection dropdown, similar to demo. 
theme_categories = ['Social impact of scientific developments',
                    'Policy decisions on scientific issues',
                    'Confidence in public figures',
                    'Importance of scientific issues',
                    'Opinions on research scientists',
                    'Questions regarding scientific research',
                    'Solving the countires problems',
                    'General scientific knowledge']

theme_names = [society, policy, confidence, scm4, scm5, q, pop, knowledge]
theme_select_dropdown = dict(zip(theme_categories, theme_names))


# labels to be used with the researcher selection dropdown
researchers_cat = ['Medical Research Scientists', 
                   'Environmental Research Scientists', 
                   'Nutrition Research Scientists']

med_scientists = [i for i in rq_form1 if re.search("(_F1A)", i)]
env_scientists = [i for i in rq_form1 if re.search("(_F1B)", i)]
nutr_scientists = [i for i in rq_form1 if re.search("(_F1C)", i)]

research_names = [med_scientists, env_scientists, nutr_scientists]
res_dropdown = dict(zip(researchers_cat, research_names))


# labels to be used with the practitioner selection dropdown
practitioners_cat = ['Medical Doctors', 
                     'Environmental Health Specialists', 
                     'Dietician']

md = [i for i in pq_form2 if re.search("(_F2A)", i)]
env_specialists = [i for i in pq_form2 if re.search("(_F2B)", i)]
dieticians = [i for i in pq_form2 if re.search("(_F2C)", i)]

pract_names = [md, env_specialists, dieticians]
pract_dropdown = dict(zip(practitioners_cat, pract_names))

'''
---------
FUNCTIONS
---------
'''
# Rather than repeat the following code for the callbacks of tab1/tab2/tab3, they're saved as the following functions

def make_freq_distr(x,y):
    new_df = pd.crosstab(df_copy[x],
                     df_copy[y],
                     df_copy.WEIGHT_W42, aggfunc = sum, dropna=True,
                     normalize='index'). \
                     loc[meta.variable_value_labels[x].values()]. \
                     loc[:, meta.variable_value_labels[y].values()]*100
    
    new_df = new_df.applymap(lambda x: round(x, 2))

    fig = px.bar(data_frame=new_df,
                 x=new_df.columns,
                 y=new_df.index,
                 color_discrete_sequence=['#636efa', '#00cc96', '#ef553b', '#ab63fa'])
#                  hover_name=new_df.index,
#                  hover_data=(new_df.columns)*100),

    fig.update_layout(
        font={'size':15},
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Frequency (%)",
        yaxis_title=None,

        legend=dict(
            font=dict(size=16),
            title=None,
            yanchor="top",
            y=1.5,
            xanchor="left",
            x=0.01)
    )
    
    return fig


def unweighted_table(x,y):
    temp_groupby = df_copy.groupby([x, y]).WEIGHT_W42.count().reset_index()
    
    temp_pivot = temp_groupby.pivot(index=x, columns=y, values='WEIGHT_W42')\
                    .loc[meta.variable_value_labels[x].values()]\
                    .loc[:, meta.variable_value_labels[y].values()]
    
    temp_values = np.rot90(temp_pivot.values, k=3)
    temp_values = np.fliplr(temp_values)
    temp_values = np.vstack([temp_pivot.index, temp_values])

    fig = go.Figure(data=[go.Table(
        header=dict(values=['Index'] + list(temp_pivot.columns)),
        cells=dict(values=temp_values))
                         ])

    return fig
    
    

def weighted_table(x,y):
    temp_groupby = df_copy.groupby([x, y]).WEIGHT_W42.sum().reset_index()
    
    temp_groupby.WEIGHT_W42 = temp_groupby.WEIGHT_W42.map(lambda x: round(x, 0))
    
    temp_pivot = temp_groupby.pivot(index=x, columns=y, values='WEIGHT_W42')\
                    .loc[meta.variable_value_labels[x].values()]\
                    .loc[:, meta.variable_value_labels[y].values()]
    
    temp_values = np.rot90(temp_pivot.values, k=3)
    temp_values = np.fliplr(temp_values)
    temp_values = np.vstack([temp_pivot.index, temp_values])

    fig = go.Figure(data=[go.Table(
        header=dict(values=['Index'] + list(temp_pivot.columns)),
        cells=dict(values=temp_values))
                         ])

    return fig

def chi_squared(x,y):
    stats_df = pd.crosstab(index=df[x],
                       columns=df[y],
                       values=df.WEIGHT_W42,
                       aggfunc='sum',
                       dropna=True)
    
    stats_df = stats_df.drop(99.0, axis=0)
    stats_df = stats_df.drop(99.0, axis=1)
    
    observed_freq = stats_df.to_numpy()
    
    chi2, p, dof, expected = stats.chi2_contingency(observed_freq)
    
    return 'chi-squared: {} || p-value: {} || degrees of freedom: {}'.format(chi2, p, dof)



""" 
--------------------------------------
DASH APP: LAYOUT, TABS, CALLBACKS
--------------------------------------

------
LAYOUT
------
"""

app.layout = html.Div([
    dbc.Container([
        
# Navbar
        dbc.NavbarSimple(
            brand="Science and Society",
            brand_href="#",
            color="primary",
            dark=True,
            fluid=True
        ),
        html.Br(),

# Intro
        html.Div([
            html.H4(children=['Introduction'], style={'font-family':'sans-serif'}),
            html.Hr(),
            html.P("""\
            In 2019, the Pew Research Center conducted a survey of 4,464 adults living within households
            in the United States. Part of their American Trends Panel, the survey measured respondent
            attitudes regarding a number of topics, from trust in researchers and the scientific process
            to whether or not scientists should be involved with guiding public policy decisions.
            This dashboard's purpose is to provide the user with the ability to examine theses trends for themselves.
            """)
        ],
            style={'background-color':'rgba(229, 237, 250, 0.5', 'padding': '5px'}
        ),
        html.Br(),

        html.H4(children=['Exploring by demographic']),
        html.Hr(),

        html.Div([
            html.P('''\
            The following frequency distributions represent the proportion of answers given by a particular demographic.
            Age category is provided as the default. The themes covered by the survey were pre-grouped according to 
            general similarities determined by the researchers, and within each group specific survey items can be selected.
            '''),
            html.P(html.Em('''
            Survey items regarding researchers (medical, environmental, nutrition) and practitioners 
            (doctors, env. health specialists, dieticians) have been separated into their own tabs to
            help simplify the options menu.
            ''')),
            html.P('Note: DK/REF stands for didn\'t know / refused to respond.')
        ]),

# Tabs

        html.Div([
            dbc.Tabs(
                [
                    dbc.Tab(label='Main', tab_id='tab-1'),
                    dbc.Tab(label='Researchers', tab_id='tab-2'),
                    dbc.Tab(label='Practitioners', tab_id='tab-3')
                ],
                id="tabs",
                active_tab="tab-1",
                ),
            html.Div(id="content"),
        ]),
        html.Br(),
        
        html.Div([
            dbc.Col(
                [html.Em(children=['A note on the data'], style={'font-family':'sans-serif'}),
                 html.P("""\
                 Weighted values are used to better represent the distribution of sociodemographic characteristics in 
                 the U.S. population. If not taken into account, the following tables and charts could over- or underrepresent
                 a given demographic's response.
                 """)
                ],
                    lg=12,
            )
        ]),
        html.Br(),
    ])
],
style={'background-color:': 'rgba(197, 220, 235, 0.9)',
       'margin':'2rem'}
)


""" 
-----
TAB 1
-----
"""

tab1_content = html.Div([
        html.Br(),
       
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6(children=['Demographic'], style={'font-family':'sans-serif'}),
                    dcc.Dropdown(
                        id = 'xaxis-column1',
                        options = demo_dropdown,
                        value = 'F_AGECAT'
                    )
                ],
                    lg=8
                )
            ]),
            html.Br(),

            dbc.Row([
                dbc.Col([
                    html.H6(children=['Theme'], style={'font-family':'sans-serif'}),
                    dcc.Dropdown(
                        id = 'theme-selection',
                        options = [{'label': k, 'value': k} for k in theme_select_dropdown.keys()],
                        value = 'Social impact of scientific developments'
                    )
                ],
                    lg=8)
            ]),
            html.Br(),

            dbc.Row([
                 dbc.Col([
                    dcc.RadioItems(id='yaxis-column1',
                                  value = 'PAST_W42',
                                  inputStyle={'display-internal':'table-row'})
                ]),
            ]),

            dbc.Row([
                html.Br(),
                html.Br(),
                html.Br(),

                dbc.Col([
                    dcc.Graph(id='indicator-bar1',
                              config={'displayModeBar': False}
                    )
                ])
            ]),
            
            html.Br(),
            dbc.Row([
                html.P(id='chi-squared1')
            ]),
            html.H5('unweighted data'),
            dbc.Row([
                dcc.Graph(id='unweighted-table1')
            ]),
            html.H5('weighted data'),
            dbc.Row([
                dcc.Graph(id='weighted-table1')
            ])
        ])
])


""" 
-----
TAB 2
-----
"""

tab2_content = html.Div([
        html.Br(),

        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6(children=['Please choose a demographic'], style={'font-family':'sans-serif'}),
                    dcc.Dropdown(
                        id = 'xaxis-column2',
                        options = demo_dropdown,
                        value = 'F_AGECAT'
                    )
                ],
                    lg=8
                )
            ]),
            html.Br(),

            dbc.Row([
                dbc.Col([
                    html.H6(children=['Researcher'], style={'font-family':'sans-serif'}),
                    dcc.Dropdown(
                        id = 'researcher-selection',
                        options = [{'label': k, 'value': k} for k in res_dropdown.keys()],
                        value = 'Medical Research Scientists'
                    )
                ],
                    lg=8)
            ]),
            html.Br(),

            dbc.Row([
                 dbc.Col([
                    dcc.RadioItems(id='yaxis-column2',
                                  value = 'RQ1_F1A_W42',
                                  inputStyle={'display-inside':'flow'})
                ]),
            ]),

            dbc.Row([
                html.Br(),
                html.Br(),
                html.Br(),

                dbc.Col([
                    dcc.Graph(id='indicator-bar2',
                              config={'displayModeBar': False}
                    )
                ])
            ])
        ])
])


""" 
-----
TAB 3
-----
"""

tab3_content = html.Div([
        html.Br(),

        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6(children=['Please choose a demographic'], style={'font-family':'sans-serif'}),
                    dcc.Dropdown(
                        id = 'xaxis-column3',
                        options = demo_dropdown,
                        value = 'F_AGECAT'
                    )
                ],
                    lg=8
                )
            ]),
            html.Br(),

            dbc.Row([
                dbc.Col([
                    html.H6(children=['Practitioner'], style={'font-family':'sans-serif'}),
                    dcc.Dropdown(
                        id = 'practitioner-selection',
                        options = [{'label': k, 'value': k} for k in pract_dropdown.keys()],
                        value = 'Medical Doctors'
                    )
                ],
                    lg=8)
            ]),
            html.Br(),

            dbc.Row([
                 dbc.Col([
                    dcc.RadioItems(id='yaxis-column3',
                                  value = 'PQ1_F2A_W42',
                                  inputStyle={'display-inside':'flow'})
                ]),
            ]),

            dbc.Row([
                html.Br(),
                html.Br(),
                html.Br(),

                dbc.Col([
                    dcc.Graph(id='indicator-bar3',
                              config={'displayModeBar': False}
                    )
                ])
            ])
        ])
])


""" 
----------------
LAYOUT CALLBACKS
----------------
"""

# Switch tabs
@app.callback(
    Output('content', 'children'),
    [Input('tabs', 'active_tab')]
)
def switch_tab(at):
    if at == 'tab-1':
        return tab1_content
    elif at == 'tab-2':
        return tab2_content
    elif at == 'tab-3':
        return tab3_content
    return html.P("This shouldn't ever be displayed...")

""" 
---------------
TAB 1 CALLBACKS
---------------
"""
@app.callback(
    Output('yaxis-column1', 'options'),
    [Input('theme-selection', 'value')]
)
def set_theme_options(selected_theme):
        temp = [i for i in theme_select_dropdown[selected_theme]]
        temp_list = [{'label': meta.column_names_to_labels[i], 'value': i} for i in temp]
        
        return temp_list


@app.callback(
    Output('indicator-bar1', 'figure'),
    [Input('xaxis-column1', 'value'),
     Input('yaxis-column1', 'value')]
)
def update_graph(x_axis, y_axis):
    return make_freq_distr(x_axis, y_axis)

@app.callback(
    Output('unweighted-table1', 'figure'),
    [Input('xaxis-column1', 'value'),
     Input('yaxis-column1', 'value')]
)
def update_uw_table(x, y):
    return unweighted_table(x, y)


@app.callback(
    Output('weighted-table1', 'figure'),
    [Input('xaxis-column1', 'value'),
     Input('yaxis-column1', 'value')]
)
def update_uw_table(x, y):
    return weighted_table(x, y)


@app.callback(
    Output('chi-squared1', 'children'),
    [Input('xaxis-column1', 'value'),
     Input('yaxis-column1', 'value')]
)
def update_chi_squared(x, y):
    return chi_squared(x, y)
    
""" 
---------------
TAB 2 CALLBACKS
---------------
"""
@app.callback(
    Output('yaxis-column2', 'options'),
    [Input('researcher-selection', 'value')]
)
def set_theme_options(selected_theme):
        temp = [i for i in res_dropdown[selected_theme]]
        temp_list = [{'label': meta.column_names_to_labels[i], 'value': i} for i in temp]
        
        return temp_list

    
@app.callback(
    Output('indicator-bar2', 'figure'),
    [Input('xaxis-column2', 'value'),
     Input('yaxis-column2', 'value')]
)
def update_graph(x_axis, y_axis):
    return make_freq_distr(x_axis, y_axis)

""" 
---------------
TAB 3 CALLBACKS
---------------
"""
@app.callback(
    Output('yaxis-column3', 'options'),
    [Input('practitioner-selection', 'value')]
)
def set_theme_options(selected_theme):
        temp = [i for i in pract_dropdown[selected_theme]]
        temp_list = [{'label': meta.column_names_to_labels[i], 'value': i} for i in temp]
        
        return temp_list

    
@app.callback(
    Output('indicator-bar3', 'figure'),
    [Input('xaxis-column3', 'value'),
     Input('yaxis-column3', 'value')]
)
def update_graph(x_axis, y_axis):
    return make_freq_distr(x_axis, y_axis)




if __name__ == '__main__':
    app.run_server(debug=True)