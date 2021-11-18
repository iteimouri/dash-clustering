import dash_html_components as html

data_error_message = html.Div(
    children=[
        html.Span(
            children='Error: ',
            style={
                'font-weight': 'bold',
                'white-space': 'pre',
                'color': '#8d8d8d'
            }
        ),
        'No Data.'
    ], style={
        'color': '#9d9d9d',
        'font-size': '90%',
        'width': '25vw',
        'padding': '2vw 0vw 0vw 0vw'
    }
)

pca_error_message = html.Div(
    children=[
        html.Span(
            children='Error: ',
            style={
                'font-weight': 'bold',
                'white-space': 'pre',
                'color': '#8d8d8d'
            }
        ),
        'The number of components must be greater than or equal to two, and less than the number of features.'
    ], style={
        'color': '#9d9d9d',
        'font-size': '90%',
        'width': '25vw',
        'padding': '2vw 0vw 0vw 0vw'
    }
)

batch_error_message = html.Div(
    children=[
        html.Span(
            children='Error: ',
            style={
                'font-weight': 'bold',
                'white-space': 'pre',
                'color': '#8d8d8d'
            }
        ),
        'The batch size must be greater than or equal to two, and less than the number of samples.'
    ], style={
        'color': '#9d9d9d',
        'font-size': '90%',
        'width': '25vw',
        'padding': '2vw 0vw 0vw 0vw'
    }
)

cluster_error_message = html.Div(
    children=[
        html.Span(
            children='Error: ',
            style={
                'font-weight': 'bold',
                'white-space': 'pre',
                'color': '#8d8d8d'
            }
        ),
        'The number of clusters must be greater than or equal to two, and less than the number of samples.'
    ], style={
        'color': '#9d9d9d',
        'font-size': '90%',
        'width': '25vw',
        'padding': '2vw 0vw 0vw 0vw'
    }
)