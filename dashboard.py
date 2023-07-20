from dash import Dash, dcc, html, Input, Output, callback
import base64
import io

from car_detection import play

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Переместите или ',
            html.A('Выберите видео-файл')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin-top': '320px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.H6('Нажмите чтобы запустить детекцию', style={'margin-left': '590px'}),
    html.Button('Detect!', id='detect',
                style={
                    'font-size': '12px', 
                    'width': '140px', 
                    'display': 'inline-block', 
                    'margin-bottom': '10px', 
                    'margin-right': '5px', 
                    'height': '37px', 
                    'verticalAlign': 'top',
                    'margin-left': '690px'
                }, )
])


@callback(Output('detect', 'n_clicks'),
          Input('upload-data', 'contents'),
          Input('upload-data', 'filename'),
          Input('detect', 'n_clicks'),
            
          )
def out_video(contents, filename, n_clicks):
    try:
        if not n_clicks:
            return 0

        _, content_string = contents[0].split(',')

        decoded = base64.b64decode(content_string)

        play(io.BytesIO(decoded))

        return 0
    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    app.run(debug=True)
