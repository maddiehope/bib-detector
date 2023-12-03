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
from flask_mail import Mail, Message
import sqlite3
import os
import joblib

# ----------------------------------------------------------------------------------------------------------------------------

# FLASK CONFIGURATION/SETUP: -------------------------------------------------------------------------------------------------

# creating the Flask 
app = Flask(__name__, template_folder='templates')
CORS(app)
mail = Mail(app)

# configuring email settings
import config # private file containing email & password information

app.config['MAIL_SERVER'] = 'https://bib-detector.dynv6.net'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = config.email
app.config['MAIL_PASSWORD'] = config.password
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

# function to send user email of the prediction results
def send_email(receiver_email, subject, body, attachment_path):
   
   message = Message(subject, sender=config.email, recipients=[receiver_email])
   message.body = body

   with app.open_resource(attachment_path) as attachment:
      message.attach(attachment_path, 'text/csv', attachment.read())

   mail.send(message)

# "Home" page, route "/"
@app.route('/', methods=['GET'])
def home():

    return render_template("home.html",title="Home") 

# "Upload" page, route "/upload"
@app.route('/upload', methods=['POST'])
def upload():

    '''
        This is where the code that makes predictions will go.
    '''

    f = request.files['file']
    email = request.form['email']
    f.save(os.path.join('uploads', f.filename))

    send_email(email, 'Your Bib Prediction Results', 'Please find the attached CSV file with your video prediction results.', 'test.csv') # test.csv is temporary

    return render_template("results.html", title="Results")


# ----------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.debug = True
    app.run()

#if __name__ == "__main__":
    #app.run(debug=True, port=8080) # can be commented out when deployed