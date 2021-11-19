import pandas as pd
import numpy as np
import warnings
import base64
import io
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from flask import Flask
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

from modules.errors import *
from modules.tables import table
from modules.utils import palette

# App set-up.
external_stylesheets = [
    'https://fonts.googleapis.com/css?family=Open+Sans:300,400,700',
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
]

server = Flask(__name__)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

app.title = 'Clustering'

app.css.config.serve_locally = True
app.config.suppress_callback_exceptions = True
app.scripts.config.serve_locally = True

application = app.server

# App layout.
app.layout = html.Div(children=[

    # First column.
    html.Div(

        children=[

            # Data upload.
            html.Label(
                children='Data',
                className='custom-label'
            ),

            html.Div(
                id='data_tooltip',
                children='?',
                className='custom-tooltip-icon',
            ),

            dbc.Tooltip(
                children=[
                    'The input file should contain the index in the first column and the features in the subsequent columns, '
                    'with the corresponding column headers. The accepted file formats are CSV, XLS, XLSX, and TXT. If "Sample" '
                    'is selected, the results will be generated using a sample data set.'],
                target='data_tooltip',
                placement='right',
                style={
                    'line-height': '0.7vw',
                    'font-size': '0.6vw',
                    'color': 'white',
                    'text-align': 'justify',
                    'background-color': '#5D69B1',
                    'border-radius': '5px',
                    'padding': '0.25vw 0.25vw 0.25vw 0.25vw',
                    'height': '4vw',
                    'width': '16vw',
                }
            ),

            dcc.Checklist(
                id='sample_data',
                options=[{'label': 'Sample', 'value': 'true'}],
                value=['true'],
                inputClassName='custom-select-input',
                style={
                    'margin': '1vw 0vw 0.25vw 8vw',
                    'display': 'inline-block',
                    'vertical-align': 'middle',
                    'color': '#5D69B1',
                    'font-size': '90%',
                }
            ),

            dcc.Upload(
                id='uploaded_file',
                children=html.Div(
                    children=[
                        html.P(
                            children='Drag and Drop or ',
                            style={
                                'display': 'inline',
                                'text-decoration': 'none'
                            }
                        ),
                        html.A(
                            children='Select File',
                            style={
                                'display': 'inline',
                                'text-decoration': 'underline'
                            }
                        ),
                    ],
                    style={'font-size': '90%'}
                ),
                className='custom-upload',
                multiple=False
            ),

            # Feature scaling.
            html.Label(
                children='Feature Scaling',
                className='custom-label'
            ),

            dcc.RadioItems(
                id='scaling',
                options=[
                    {'label': 'Standard Scaler', 'value': 'standard'},
                    {'label': 'Robust Scaler', 'value': 'robust'},
                    {'label': 'MinMax Scaler', 'value': 'minmax'},
                    {'label': 'MaxAbs Scaler', 'value': 'maxabs'},
                    {'label': 'None', 'value': 'none'}
                ],
                value='standard',
                inputClassName='custom-select-input',
                labelClassName='custom-select-label'
            ),

            # Principal components analysis.
            html.Label(
                children='Principal Component Decomposition',
                className='custom-label'
            ),

            dcc.RadioItems(
                id='decomposition',
                options=[
                    {'label': 'True', 'value': 'true'},
                    {'label': 'False', 'value': 'false'}
                ],
                value='true',
                inputClassName='custom-select-input',
                labelClassName='custom-select-label'
            ),

            html.Label(
                children='Number of Principal Components',
                className='custom-label'
            ),

            html.Div(
                id='decomposition_tooltip',
                children='?',
                className='custom-tooltip-icon'
            ),

            dbc.Tooltip(
                children='Must be greater than or equal to two, and less than the number of features.',
                target='decomposition_tooltip',
                placement='right',
                style={
                    'line-height': '0.7vw',
                    'font-size': '0.6vw',
                    'color': 'white',
                    'text-align': 'justify',
                    'background-color': '#5D69B1',
                    'border-radius': '5px',
                    'padding': '0.25vw 0.25vw 0.25vw 0.25vw',
                    'height': '1.4vw',
                    'width': '16vw',
                }
            ),

            dcc.Input(
                id='n_components',
                type='number',
                min=2,
                value=3,
                debounce=True,
                className='custom-numeric-input'
            ),

            # Clustering.
            html.Label(
                children='Clustering Algorithm',
                className='custom-label'
            ),

            dcc.RadioItems(
                id='algorithm',
                options=[
                    {'label': 'K-Means', 'value': 'k-means'},
                    {'label': 'K-Means++', 'value': 'k-means++'}
                ],
                value='k-means++',
                inputClassName='custom-select-input',
                labelClassName='custom-select-label'
            ),

            html.Label(
                children='Number of Clusters',
                className='custom-label'
            ),

            html.Div(
                id='clusters_tooltip',
                children='?',
                className='custom-tooltip-icon'
            ),

            dbc.Tooltip(
                children='Must be greater than or equal to two, and less than the number of samples.',
                target='clusters_tooltip',
                placement='right',
                style={
                    'line-height': '0.7vw',
                    'font-size': '0.6vw',
                    'color': 'white',
                    'text-align': 'justify',
                    'background-color': '#5D69B1',
                    'border-radius': '5px',
                    'padding': '0.25vw 0.25vw 0.25vw 0.25vw',
                    'height': '1.4vw',
                    'width': '16vw',
                }
            ),

            dcc.Input(
                id='n_clusters',
                type='number',
                min=2,
                value=3,
                debounce=True,
                className='custom-numeric-input'
            ),

            html.Label(
                children='Mini-Batch',
                className='custom-label'
            ),

            dcc.RadioItems(
                id='mini_batch',
                options=[
                    {'label': 'True', 'value': 'true'},
                    {'label': 'False', 'value': 'false'}
                ],
                value='false',
                inputClassName='custom-select-input',
                labelClassName='custom-select-label'
            ),

            html.Label(
                children='Batch Size',
                className='custom-label'
            ),

            html.Div(
                id='batch_tooltip',
                children='?',
                className='custom-tooltip-icon'
            ),

            dbc.Tooltip(
                children='Must be greater than or equal to two, and less than the number of samples.',
                target='batch_tooltip',
                placement='right',
                style={
                    'line-height': '0.7vw',
                    'font-size': '0.6vw',
                    'color': 'white',
                    'text-align': 'justify',
                    'background-color': '#5D69B1',
                    'border-radius': '5px',
                    'padding': '0.25vw 0.25vw 0.25vw 0.25vw',
                    'height': '1.4vw',
                    'width': '16vw',
                }
            ),

            dcc.Input(
                id='batch_size',
                type='number',
                min=2,
                value=100,
                debounce=True,
                className='custom-numeric-input'
            ),

        ],

        style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'width': '20vw',
            'margin': '1vw 0vw 1vw 2vw'
        }

    ),

    # Second column.
    html.Div(

        children=[

            html.Label(
                children='Clustering Performance Evaluation',
                className='custom-label'
            ),

            html.Div(
                id='clustering_performance',
                style={
                    'display': 'block',
                    'height': '6vw',
                    'width': '30vw'
                }
            ),

            html.Label(
                children='Clustering Results',
                className='custom-label'
            ),

            html.Div(
                id='clustering_results',
                style={
                    'display': 'block',
                    'height': '30vw',
                    'width': '30vw'
                }
            ),

        ],

        style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'width': '30vw',
            'margin': '1vw 0vw 1vw 0vw'
        }

    ),

    # Third column.
    html.Div(

        children=[

            html.Label(
                children='Cluster Visualization',
                className='custom-label'
            ),

            html.Div(
                id='plot_tooltip',
                children='?',
                className='custom-tooltip-icon'
            ),

            dbc.Tooltip(
                children='The plot below is obtained using t-SNE to reduce the data to two dimensions.',
                target='plot_tooltip',
                placement='right',
                style={
                    'line-height': '0.7vw',
                    'font-size': '0.6vw',
                    'color': 'white',
                    'text-align': 'justify',
                    'background-color': '#5D69B1',
                    'border-radius': '5px',
                    'padding': '0.25vw 0.25vw 0.25vw 0.25vw',
                    'height': '1.4vw',
                    'width': '16vw',
                }
            ),

            html.Div(
                id='cluster_visualization',
                style={
                    'display': 'block',
                    'height': '22vw',
                    'width': '44vw'
                }
            ),

            html.Label(
                children='Descriptive Statistics',
                className='custom-label'
            ),

            html.Div(
                id='stats_tooltip',
                children='?',
                className='custom-tooltip-icon'
            ),

            dbc.Tooltip(
                children='The table below reports the descriptive statistics of the raw data, prior to any scaling being applied.',
                target='stats_tooltip',
                placement='right',
                style={
                    'line-height': '0.7vw',
                    'font-size': '0.6vw',
                    'color': 'white',
                    'text-align': 'justify',
                    'background-color': '#5D69B1',
                    'border-radius': '5px',
                    'padding': '0.25vw 0.25vw 0.25vw 0.25vw',
                    'height': '1.4vw',
                    'width': '16vw',
                }
            ),

            html.Div(
                id='descriptive_statistics',
                style={
                    'display': 'block',
                    'height': '18vw',
                    'width': '44vw'
                }
            ),

        ],

        style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'width': '40vw',
            'margin': '1vw 0vw 1vw 2vw'
        }

    ),

])

# App callbacks.
@app.callback(
    [
        Output('descriptive_statistics', 'children'),
        Output('clustering_results', 'children'),
        Output('cluster_visualization', 'children'),
        Output('clustering_performance', 'children')
    ],
    [
        Input('sample_data', 'value'),
        Input('uploaded_file', 'contents'),
        Input('uploaded_file', 'filename'),
        Input('scaling', 'value'),
        Input('decomposition', 'value'),
        Input('n_components', 'value'),
        Input('algorithm', 'value'),
        Input('n_clusters', 'value'),
        Input('mini_batch', 'value'),
        Input('batch_size', 'value')
    ]
)
def update_results(sample_data,
                   uploaded_file,
                   filename,
                   scaling,
                   decomposition,
                   n_components,
                   algorithm,
                   n_clusters,
                   mini_batch,
                   batch_size):

    try:

        # Load the data.
        if sample_data == ['true']:
            df = pd.read_csv('data/sample_data.csv', index_col=0)

        elif uploaded_file is not None:

            try:
                content_type, content_string = uploaded_file.split(',')
                decoded = base64.b64decode(content_string)

                if 'csv' in filename:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0)

                elif 'xls' in filename:
                    df = pd.read_excel(io.BytesIO(decoded), index_col=0)

                elif 'txt' in filename:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+', index_col=0)

                else:
                    return 4 * [upload_error_message]

            except:
                return 4 * [upload_error_message]

        else:
            return 4 * [data_error_message]

        # Drop the missing values.
        df.dropna(inplace=True)

        # Extract the index.
        index = list(df.index.astype(str))

        # Extract the data.
        X = df.values

        # Rescale the data
        if scaling == 'standard':
            X = StandardScaler().fit_transform(X)

        elif scaling == 'robust':
            X = RobustScaler().fit_transform(X)

        elif scaling == 'minmax':
            X = MinMaxScaler().fit_transform(X)

        elif scaling == 'maxabs':
            X = MaxAbsScaler().fit_transform(X)

        # Decompose the data.
        if decomposition == 'true':
            if n_components < 2 or n_components > X.shape[1] - 1:
                return 4 * [pca_error_message]
            else:
                X = PCA(n_components=np.int(n_components), random_state=0).fit_transform(X)

        # Instantiate the model.
        if n_clusters < 2 or n_clusters > X.shape[0] - 1:
            return 4 * [cluster_error_message]

        else:

            if mini_batch == 'false':
                if algorithm == 'k-means':
                    kmeans = KMeans(init='random', n_clusters=n_clusters, random_state=0)
                elif algorithm == 'k-means++':
                    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, random_state=0)

            else:
                if batch_size < 2 or batch_size > X.shape[0] - 1:
                    return 4 * [batch_error_message]

                else:
                    if algorithm == 'k-means':
                        kmeans = MiniBatchKMeans(init='random', n_clusters=n_clusters, batch_size=batch_size, random_state=0)
                    elif algorithm == 'k-means++':
                        kmeans = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size, random_state=0)

            # Fit the model.
            model = kmeans.fit(np.double(X))

            # Extract the cluster labels.
            Y = model.labels_

            # Extract the number of clusters.
            N = len(np.unique(Y))

            # Generate the descriptive statistics table.
            stats = df.describe()
            stats.iloc[:1, :] = stats.iloc[:1, :].apply(lambda x: [format(z, ',.0f') for z in x], axis=0)
            stats.iloc[1:, :] = stats.iloc[1:, :].apply(lambda x: [format(z, ',.3f') for z in x], axis=0)
            stats.index = ['Count', 'Mean', 'Standard Deviation', 'Minimum', '25th Percentile', 'Median', '75th Percentile', 'Maximum']
            stats.reset_index(drop=False, inplace=True)
            stats.rename(columns={'index': ''}, inplace=True)
            stats = table(stats, height=18, width=44)

            # Generate the results table.
            results = df.copy()
            results = results.apply(lambda x: [format(z, ',.2f') for z in x], axis=0)
            results['Record ID'] = index
            results['Cluster Label'] = Y
            columns = ['Record ID', 'Cluster Label']
            columns.extend(df.columns)
            results = results[columns]
            results.reset_index(inplace=True, drop=True)
            results = table(results, height=28, width=30)

            # Generate the metrics table.
            metrics = pd.DataFrame({
                'Metric': [
                    'Silhouette Coefficient',
                    'Davies-Bouldin Index',
                    'Calinski-Harabasz Index'
                ],
                'Value': [
                    format(silhouette_score(X, Y), ',.3f'),
                    format(davies_bouldin_score(X, Y), ',.3f'),
                    format(calinski_harabasz_score(X, Y), ',.0f')
                ]
            })
            metrics = table(metrics, height=6, width=30)

            # Reduce the data to 2 dimensions using t-SNE.
            reduced_data = TSNE(n_components=2, random_state=0).fit_transform(X)
            reduced_model = kmeans.fit(np.double(reduced_data))
            Z = reduced_model.labels_

            # Define the meshgrid for the contour plot.
            x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
            y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

            x = np.linspace(x_min, x_max, 500)
            y = np.linspace(y_min, y_max, 500)
            xx, yy = np.meshgrid(x, y)

            z = reduced_model.predict(np.c_[xx.ravel(), yy.ravel()])
            z = z.reshape(xx.shape)

            # Get the color palette.
            p = palette(N)

            # Define the fill colors for the contour plot.
            colorscale = [[z / np.max(Z), p[z]] for z in np.unique(Z).tolist()]

            # Define the marker colors for the scatter plot.
            colors = [p[z] for z in Z.tolist()]

            # Define the hover text for the scatter plot.
            text = ['<b>Record No.: </b>' + index[i] + '<br><b>Cluster No.: </b>' + str(Z[i]) for i in range(len(Z))]

            # Generate the figure layout.
            layout = dict(
                paper_bgcolor='white',
                plot_bgcolor='white',
                showlegend=False,
                margin=dict(t=0, b=0, r=0, l=0),
                font=dict(family='Open Sans'),
                xaxis=dict(
                    range=[np.min(x), np.max(x)],
                    visible=False
                ),
                yaxis=dict(
                    range=[np.min(y), np.max(y)],
                    visible=False
                )
            )

            # Generate the figure traces.
            data = []

            data.append(
                go.Contour(
                    x=x,
                    y=y,
                    z=z,
                    colorscale=colorscale,
                    opacity=0.25,
                    showscale=False,
                    hoverinfo='none'
                )
            )

            data.append(
                go.Scatter(
                    x=reduced_data[:, 0],
                    y=reduced_data[:, 1],
                    mode='markers',
                    text=text,
                    marker=dict(
                        color=colors,
                        size=9,
                        line=dict(width=1)
                    ),
                    hovertemplate='%{text}<extra></extra>'
                )
            )

            # Generate the graph.
            plot = dcc.Graph(
                figure=go.Figure(
                    data=data,
                    layout=layout
                ),
                style={
                    'height': '22vw',
                    'width': '44vw'
                },
                config={
                    'responsive': True,
                    'autosizable': True
                }
            )

        return [stats, results, plot, metrics]

    except:

        raise dash.exceptions.PreventUpdate

if __name__ == '__main__':
    application.run(debug=False, host='127.0.0.1')
