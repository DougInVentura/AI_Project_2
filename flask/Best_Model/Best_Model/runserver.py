"""
This script runs the Best_Model application using a development server.
"""

from os import environ
from Best_Model import app


if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    # Define the upload folder
    UPLOAD_FOLDER = 'Resources/'
    PLOT_DISPLAY_FOLDER = 'plots/'
    PLOT_SAVE_FOLDER = '/Best_Model/static/plots/'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['PLOT_SAVE_FOLDER'] = PLOT_SAVE_FOLDER
    app.config['PLOT_DISPLAY_FOLDER'] = PLOT_DISPLAY_FOLDER
    app.secret_key = '^#UzJd4MZWAx@f%4m'
    
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
