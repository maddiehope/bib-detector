'''
    BIB DETECTION FLASK APP
    https://bib-detector.dynv6.net
    -----------------------

    Oracle Cloud instance: instance-20231130-1131
    Public IP Address: 150.136.10.94

    Logging in:
        IP adress: ssh -i ~/Desktop/ssh-key-2023-11-30.key ubuntu@150.136.10.94
        Domain: ssh -i ~/Desktop/ssh-key-2023-11-30.key ubuntu@bib-detector.dynv6.net

'''

# IMPORTS: -------------------------------------------------------------------------------------------------------------------

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import requests
import pandas as pd
from flask_cors import CORS
import sqlite3
import os
import joblib

# ----------------------------------------------------------------------------------------------------------------------------

# FLASK CONFIGURATION/SETUP: -------------------------------------------------------------------------------------------------

# creating the Flask 
app = Flask(__name__, template_folder='templates')
CORS(app)

# "Home" page, route "/"
@app.route('/', methods=['GET'])
def home():

    ### TEST ###
    loaded_model = joblib.load('number_classifier_vgg16.pkl')
    print(type(loaded_model))
    ############

    return render_template("home.html",title="Home") 


# ----------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.debug = True
    app.run()
    
#if __name__ == "__main__":
    #app.run(debug=True, port=8080) # can be commented out when deployed