'''
    BIB DETECTION FLASK APP

    Oracle Cloud instance: instance-20231130-1131
    
'''

# IMPORTS: -------------------------------------------------------------------------------------------------------------------

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import requests
import pandas as pd
from flask_cors import CORS
import sqlite3
import os

# ----------------------------------------------------------------------------------------------------------------------------

# FLASK CONFIGURATION/SETUP: -------------------------------------------------------------------------------------------------

# creating the Flask 
app = Flask(__name__, template_folder='templates')
CORS(app)

# "Home" page, route "/"
@app.route('/', methods=['GET'])
def selection():
    return render_template("home.html",title="Home") 


# ----------------------------------------------------------------------------------------------------------------------------

app.run(debug=True, port=8080) # can be commented out when deployed