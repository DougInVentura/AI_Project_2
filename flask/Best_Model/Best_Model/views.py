"""
Routes and views for the flask application.
"""

from fileinput import filename
import os
from datetime import datetime
from tkinter import SE
from flask import render_template, request, session, send_file, send_from_directory
from flask.helpers import redirect
from Best_Model import app
from Models.tpot import evaluate_tpot
from Utilities.utils import get_columns, load_settings, save_settings

@app.route('/')
@app.route('/home')
def home():
    # run the module1 function get_accuracy and display the accuracy in the web page
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )
    """Renders the home page."""

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )


@app.route('/select_file')
def select_file():
    """
    select_file.html
    Renders the select file page.
    """
    return render_template('select_file.html')


@app.route('/file_chosen', methods=['POST'])
def file_chosen():
    """
    select_file.html
    Renders the select file page, uploads the file and displays a list of coloumns
    for the user to then select the 'y'. This method stores the selected file in the session.
    """
   
    # Get the file information from the returned request
    selectedFile = request.files['file']    
       
    # If the user does not select a file, the browser submits an empty file without a filename
    # this displayes the 'No file selected' message
    if selectedFile.filename == '':
        return render_template(
            'select_file.html',
            title='Select File',
            year=datetime.now().year,
            upload_result=f'No file selected'
        )
    
    # Build fullly qualified file name and save in the session
    filename = os.path.join(app.config['UPLOAD_FOLDER'], selectedFile.filename)
    
    # Save the selected file to the session
    session['selectedFile'] = filename
    
    # Call the method that uploads the file (overwrites what is there)
    # TODO:
    # this could be expanded to create a temp file instead of overwriting
    # this temp file could be compared to the original file and if the temp file
    # is more up to date, then the original file could be overwritten.
    upload(request.files['file'])
    
    # Look for saved settings that provide the 'Y' value
    # if found then return the name of the column so it can be preselected
    y = load_settings(filename)
    
    # Get the columns from the file
    fields = get_columns(filename)                     
    fields = fields.split(', ')
    form_data = {'file': selectedFile.filename, 'fields': fields, 'selectedY':y }
    return render_template('select_file.html', data=form_data)

@app.route('/field_selected', methods=['POST'])
def field_selected():
    """
    select_file.html
    Collects the the selected 'y' value from the form and saves it in the session.
    """
    # Get the selected field from the form
    fieldSelected = request.form.get('dropdown')
    
    # Save the selected field in the session
    session['fieldSelected'] = fieldSelected
    
    fileName = session['selectedFile']
    save_settings(fileName, fieldSelected);
    
    # Package the data for sending to the index.html page
    data = {'fieldSelected': fieldSelected, 'selectedFile':session['selectedFile']}
    return render_template('index.html', data=data)
 

@app.route('/upload', methods=['POST'])
def upload(selectedFile):
    """ 
    select_file.html
    This method uploads the selected file to the server 
    """
      
    # Remember the filename even if not uploading   
    filename = os.path.join(app.config['UPLOAD_FOLDER'], selectedFile.filename)
    
    if selectedFile:
        # Save the file to the upload folder
        selectedFile.save(filename)

        return render_template(
            'select_file.html',
            title='Select File',
            year=datetime.now().year,
            upload_result=f'File "{selectedFile.filename}" uploaded successfully'
        )

@app.route('/tpot')
def tpot():
    return render_template('tpot.html')

@app.route('/tpot_process', methods=['POST'])
def tpot_process():
    """
    Renders the tpot page.
    """
    #Load results if available but give them the chance to reprocess if they want to.
    #   Speed
    #   Accuracy
    #   Interactive Plot
    #   Should save other results to a file for later review as well
    selectedFile = session['selectedFile']
    
    #retroeve the Y value from the session
    y = load_settings(selectedFile)
    if(y == ''):
         y = session['fieldSelected']
 
    # Evaluate the TPOT model
    results = evaluate_tpot(selectedFile, app.config['PLOT_SAVE_FOLDER'], y)
    
    # Save the results in the session
    session["MODEL_RESULTS"] = results
    
    # Get the accuracy and key features from the results
    accuracy = results['accuracy']
    key_features = results['key_features']
    
    # Filter features that contain 'uuid' as they are encoded fields
    key_features = [feature for feature in key_features if 'uuid' not in feature]
    
    # Format key features for display
    key_features = "\n\r".join(key_features)     
    # <br> doesn't work key_features = "<br>".join(key_features)   
   
    error = results['error']
       
    # Remember the name of the model before the user 
    # navigates to the graph page
    session['MODEL_PAGE'] = 'tpot.html'
    
    #Display the results in the tpot page
    form_data = {'selectedFile': selectedFile, 'accuracy':accuracy, 'selectedY':y, 'key_features_list':results['key_features'], 'key_features':results['key_features']}
    return render_template('tpot.html', title='TPOT', data=form_data)

@app.route('/display_graph', methods=['POST'])
def display_graph():
    
    # Set the plot type to 'bar' until the UI is changed to support 'line'
    plot_type = 'bar'
    

    # Get the selected field from the form
    selected_feature = request.form.get('dropdown')
    
    # Get the selected file from the session
    selectedFile = session['selectedFile']
    
    """
    # This code will open the html file but doesn't provide a way to navigate back to the tpot page
    # use this only if you cant get the graph to display in the a page using an iframe that provides a way to navigate back to the tpot page
    #build the plot file name from feature selected
    new_folder = f"D:/ASU/homework/Project 2/WebSite/Best Model/{os.path.split(selectedFile)[0]}/{app.config['PLOT_FOLDER']}"
    plot_filename = os.path.join(new_folder, os.path.basename(selectedFile))
    plot_filename = plot_filename.replace('.csv', f"_{selected_feature}.html")
   
    
    directory = os.path.dirname(plot_filename) 
    plot_filename = os.path.basename(plot_filename) 
    return send_from_directory(directory, plot_filename)
    """
    
    # Open html file
    # TODO: Change this to render the html on a template page using an iframe
    # this will allow the user to see the plot but the template page will provide
    # navigation back to the tpot page.
    
    #build the plot file name from feature selected
    #new_folder = f"{os.path.split(selectedFile)[0]}/{app.config['PLOT_FOLDER']}"
    #plot_filename = os.path.join(new_folder, os.path.basename(selectedFile))
    #plot_filename = plot_filename.replace('.csv', f"_{selected_feature}.html")
    

    
    # don't hardcode new_folder = f"file:///D:/ASU/homework/Project2/WebSite/Best_Model/{os.path.split(selectedFile)[0]}/{app.config['PLOT_FOLDER']}"
    # new_folder = f"file:///{current_directory}/{os.path.split(selectedFile)[0]}/{app.config['PLOT_FOLDER']}"
    # new_folder = f"{current_directory}/{os.path.split(selectedFile)[0]}/{app.config['PLOT_FOLDER']}"
    # new_folder = f"{current_directory}/{app.config['PLOT_FOLDER']}"
    new_folder = f"{app.config['PLOT_DISPLAY_FOLDER']}TPOT/"
    plot_filename = os.path.join(new_folder, os.path.basename(selectedFile))
    plot_filename = plot_filename.replace('.csv', f"_{selected_feature}_{plot_type}.png")

    form_data = {'image_file': plot_filename, 'model_name':'TPOT'}
    return render_template('graph.html', title='Graph', data=form_data)

    
"""
{{ url_for('static', filename=image_name) }}
This displays the image but it's not a web page.  Can't even hit back without confirming page refresh
    current_directory = os.getcwd()
    new_folder = f"{current_directory}/{os.path.split(selectedFile)[0]}/{app.config['PLOT_FOLDER']}"
    plot_filename = os.path.join(new_folder, os.path.basename(selectedFile))
    plot_filename = plot_filename.replace('.csv', f"_{selected_feature}.png")

    # form_data = {'image_file': plot_filename, 'model_name':'TPOT'}
    # return render_template('graph.html', title='Graph', data=form_data)
    return send_file(plot_filename, mimetype='image/png')
    
    HTMLPage code
    <img src="{{ url_for('display_image') }}" alt="Dynamic Image">
"""

@app.route('/return_to_model_page', methods=['POST'])
def return_to_model_page():
   model_page = session['MODEL_PAGE']
   
   # Need to retrieve data from session so you don't have to reprocess
   # when navigating back to the model processing page before selecting
   # a different feature to graph.

   #retroeve the Y value from the session
   y = session['fieldSelected']

   # Get the selected file from the session
   selectedFile = session['selectedFile']
   
   # retrieve model processing results from the session
   # Get the accuracy and key features from the results
   results = session["MODEL_RESULTS"]
   accuracy = results['accuracy']
   key_features = results['key_features']
    
   # Filter features that contain 'uuid' as they are encoded fields
   key_features = [feature for feature in key_features if 'uuid' not in feature]
    
   # Format key features for display
   key_features = "\n\r".join(key_features)     
  
   model_page = session['MODEL_PAGE']
   
   #Display the results in the tpot page
   form_data = {'selectedFile': selectedFile, 'accuracy':accuracy, 'selectedY':y, 'key_features_list':results['key_features'], 'key_features':results['key_features']}
   return render_template(model_page, title='TPOT', data=form_data)

@app.route('/automl')
def automl():
    """Renders the automl page."""
    #accuracy = evaluate_using_tpot("Resources/account_predict_data.csv")
    return render_template(
        'automl.html',
        title='H20 AutoML',
        year=datetime.now().year
        #message=f'Accuracy: {accuracy}'
    )

@app.route('/model3')
def model3():
    """Renders the page for the 3rd model."""
    accuracy = evaluate("Resources/account_predict_data.csv")
    return render_template(
        'model3.html',
        title='Third model',
        year=datetime.now().year,
        message=f'Accuracy: {accuracy}'
    )


