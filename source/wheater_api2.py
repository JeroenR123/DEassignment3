import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express
import plotly.figure_factory

import pandas as pd

import datetime

def read_data():

    # Read predictions of Random Forest and Linear Regression models
    df_rf = pd.read_parquet('gs://bucket_de/model_store/rf/prediction.parquet')
    df_reg = pd.read_parquet('gs://bucket_de/model_store/model_store/prediction.parquet')

    # Read prediction of today
    pred_df = pd.read_parquet('gs://bucket_de/model_store/prediction/prediction.parquet')
    pred = str(round(pred_df['prediction'][0], 2))

    return df_rf, df_reg, pred

# Using the Darkly theme (https://bootswatch.com/darkly/)
app = dash.Dash(external_stylesheets = [dbc.themes.DARKLY])

# Retrieve the dataframes and the prediction of today
df_rf,df_reg, today_pred = read_data()
today_date = datetime.datetime.today()

# Add column with the right dates
df_rf['date'] = [today_date - datetime.timedelta(days = i) for i in range(len(df_rf))]
df_reg['date'] = [today_date - datetime.timedelta(days = i) for i in range(len(df_reg))]

# Plot the predictions of the Random Forest model with the true values
fig1 = plotly.express.line(data_frame=df_rf,
                           x='date',
                           y=['prediction', 'true value'],
                           title='Random Forest',
                           template='plotly_dark',
                           range_y=[0, 40])

# Plot the predictions of the Linear Regression model with the true values
fig2 = plotly.express.line(data_frame=df_reg,
                           x='date',
                           y=['prediction', 'true value'],
                           title='Linear Regression',
                           template='plotly_dark',
                           range_y=[0, 40])


# Webapp layout
app.layout = html.Div([dbc.Row([dbc.Col(html.Div(html.P('Weather radar')),
                                        style = {'fontSize':40, 'height': '25px'},
                                        width = {'size':4, 'offset' : 1}),
                                dbc.Col(html.Div(
                                    html.P('The predicted temperature of {:} is {:} °C'.format(today_date.date(),
                                                                                               today_pred)),
                                        style = {'fontSize':20}),
                                        width = {'size':4, 'offset' : 6})]),
                        dbc.Row([dbc.Col([dcc.Graph(id='graph_LR', figure = fig2, style = {'height': 400})]),
                                 dbc.Col([dcc.Graph(id='graph_RF', figure=fig1, style={'height': 400})])
                                 ])])


if __name__ == '__main__':
    app.run_server(debug=True)