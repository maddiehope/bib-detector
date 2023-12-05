'''
    BIB DETECTION FLASK APP
    https://app.bib-detector.dynv6.net
    -----------------------

    Oracle Cloud instance: instance-20231130-1131
    Public IP Address: 150.136.10.94

    Logging in:
        IP adress: ssh -i ~/Desktop/ssh-key-2023-11-30.key ubuntu@150.136.10.94
        Domain: ssh -i ~/Desktop/ssh-key-2023-11-30.key ubuntu@bib-detector.dynv6.net

'''

# IMPORTS: -------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append('/Users/maddiehope/Library/Python/3.8/lib/python/site-packages') #### change file path here

from flask import Flask, render_template, request
import pandas as pd
from flask_cors import CORS
from flask_mail import Mail, Message
import os
import joblib
import sys

import model_functions # most important import :)

# ----------------------------------------------------------------------------------------------------------------------------

# FLASK CONFIGURATION/SETUP: -------------------------------------------------------------------------------------------------

# creating the Flask 
app = Flask(__name__, template_folder='templates')
CORS(app)

# configuring email settings
sys.path.append('..')       # this is where my private file containing email & password information is located                  
import config                                      # if you want to run this application locally, you can conifgure a file with your own email info

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = config.email
app.config['MAIL_PASSWORD'] = config.password
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

# function to send user email of the prediction results
def send_email(receiver_email, subject, body, attachment_path):
   
   message = Message(subject, sender=config.email, recipients=[receiver_email])
   message.body = body
   
   if attachment_path != "":
    with app.open_resource(attachment_path) as attachment:
        message.attach(attachment_path, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', attachment.read())
   
   mail.send(message)
   return("Message sent successfully.")

# "Home" page, route "/"
@app.route('/', methods=['GET'])
def home():

    return render_template("home.html",title="Home") 

# "Upload" page, route "/upload"
@app.route('/upload', methods=['POST'])
def upload():

    f = request.files['file']
    email = str(request.form['email'])
    f.save(os.path.join('uploads', f.filename))

    # Checking file type
    if (f.filename[-4:].lower() != '.mp4') and (f.filename[-4:].lower() != '.mov'):
        # Error email
        send_email(email, 
                   'Error in Bib Prediction Results', 
                   'The file you submitted was not the correct type. Please try again with a .mp4 or .mov file at https://app.bib-detector.dynv6.net.', "") # add link

    else:
        send_email(email, 
                   'Your Bib Prediction Results', 
                   'Please find the attached CSV file with your video prediction results.',
                   f'results/{email}_results.xlsx') 

        # Passing file upload to prediction model


    #model_functions.master()


    return render_template("results.html", title="Results")


# ----------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.debug = True
    app.run()

#if __name__ == "__main__":
    #app.run(debug=True, port=8080) # can be commented out when deployed