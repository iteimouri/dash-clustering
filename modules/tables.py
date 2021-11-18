import dash_html_components as html

def table(data, height, width):

    # Table header.
    table_header = [

        html.Tr(
            children=[
                html.Div(
                    children=str(data.columns[j]),
                    style={
                        'display': 'inline-block',
                        'vertical-align': 'top',
                        'width': str(width / data.shape[1]) + 'vw',
                        'height': '1.5vw',
                        'line-height': '1.5vw',
                        'white-space': 'pre',
                        'font-size': '0.6vw',
                        'text-align': 'left' if j == 0 else 'center',
                        'background-color': '#5D69B1',
                        'color': '#ffffff',
                        'padding': '0',
                    }
                )
                for j in range(data.shape[1])
            ],
        )

    ]

    # Table data.
    table_data = []

    for i in range(data.shape[0]):

        table_data.append(

            html.Tr(
                children=[
                    html.Div(
                        children=data.iloc[i, j],
                        style={
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            'width': str(width / data.shape[1]) + 'vw',
                            'height': '1.5vw',
                            'line-height': '1.5vw',
                            'white-space': 'pre',
                            'font-size': '0.6vw',
                            'text-align': 'left' if j == 0 else 'center',
                            'color': 'inherit' if j == 0 else '#666666',
                        }
                    )
                    for j in range(data.shape[1])
                ],
            )

        )

    # Full table.
    return html.Div(

        children=[

            html.Table(
                children=table_header,
                style={
                    'display': 'block',
                    'width': str(width) + 'vw',
                    'height': 'auto',
                    'white-space': 'no-wrap',
                    'overflow-y': 'hidden',
                }
            ),

            html.Table(
                children=table_data,
                style={
                    'display': 'block',
                    'width': str(width) + 'vw',
                    'max-height': str(height) + 'vw',
                    'white-space': 'no-wrap',
                    'overflow-y': 'scroll',
                }
            ),

        ],

        style={
            'display': 'block',
            'width': str(width) + 'vw',
            'overflow-x': 'scroll',
            'white-space': 'no-wrap',
        }

    )
