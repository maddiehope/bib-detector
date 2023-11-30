'''
This file contains the functions created in bib-detection.ipynb. 
Necessary functions to the API are stored here, they were simply copy and pasted from my Jupyter notebook.

If you wish to deploy the app from your own computer, note that the models will have to be trained in the 
bib-detector.ipynb notebook and the designated file paths must be changed prior to accessing these functions.

'''

# IMPORTS: -------------------------------------------------------------------------------------------------------------------

import pandas as pd
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import torch
import joblib

# ----------------------------------------------------------------------------------------------------------------------------

# MODEL INITIALIZATION: ------------------------------------------------------------------------------------------------------

'''
    MODEL OBJECTS:

        - model_bib     - bib detection model trained on custom dataset 
        - model_people  - pretrained model focusing on 'person' class
        - model_gender  - gender classification model trained on custom dataset


    
    #### NOTE: you will have to change the file path to the model below to wherever the best weights saved on your computer. 
'''

# Load the YOLO bib detection model
model_bib = YOLO("/Users/maddiehope/runs/detect/train15/weights/best.pt") #### change file path here 

# Load the YOLO people model
model_people = YOLO('yolov8n.pt') # pretrained YOLOv8n model
# Make the only detection class be 'person'
model_people.classes = ['person']

# Load the Resnet50 gender classification model
model_gender = joblib.load('gender_classifier_resnet50.pkl')

# ----------------------------------------------------------------------------------------------------------------------------

# PREDICTION PIPELINES: ------------------------------------------------------------------------------------------------------

# predicition pipeline 
# (combines model_people and model_bib for a single image)
def prediction_pipeline(image):

    people_list = []
    bibs_list = []

    '''
        Takes a string image path and runs it through the person detection model & then the bib detection model.

        The lists above are where the cropped images of each will be saved (matched by index).

        This is so that when we use the gender classifer on the person & the number detector on the bib, both results will be able to be paired 
        efficiently in the final databases. 

    '''

    # Convert the input image path back to an actual image
    # (needed for person cropping and image detection in bib model)

    im = Image.open(image) # Open image using PIL

    # If the image has an alpha channel (transparency), convert it to RGB
    if im.mode == 'RGBA':
        im = im.convert('RGB')
    
    # Run the image through people model
    model_people_result = model_people(im)

    for i, people in enumerate(model_people_result):

        people_boxes = people.boxes.xyxy.tolist() # extracts the bounding box coordniates from the results attributes
                                                  # setting the torch tensor to list will result in a list of lists,
                                                  # where each list within the list is a row of the tensor
        
        for b in people_boxes: # for every row of the tensor, it is the coordinates of a detected person box

            x_min, y_min, x_max, y_max = map(int, b) # extracting these coordinates from the rows 

            person_roi = im.crop((x_min, y_min, x_max, y_max)) # crops the region of interest (ROI) containing the person from the original image

            r = model_bib(person_roi)  # runs the cropped ROI through the bib model

            bib_box = r[0].boxes.xyxy.tolist()              # takes the model results and gets the bounding box coordinates

            if (bib_box != []): # making sure there IS a bib for the person
                x_min, y_min, x_max, y_max = map(int, bib_box[0])  # extracting these coordinates from the rows
                                                                    # no loop is necessary for the bib cropping b/c there should only be 1 bib per person 'b'

                bib_roi = person_roi.crop((x_min, y_min, x_max, y_max)) # crops the new region of interest (ROI) from the exisiting crop of the person
                                                                    # new ROI crop should be of the bib

                # Adding the cropped image of the person (person_roi) and the cropped image of the bib (im)
                people_list.append(person_roi)
                bibs_list.append(bib_roi)

    return people_list, bibs_list

# multiple predicition pipeline 
# (combines model_people and model_bib for a list of images)
def multiple_predicition_pipeline(image_paths):

    '''
        Takes multiple string image paths and runs them through the prediction_pipeline() function.

        Resulting lists are the same as what prediction_pipeline() returns, they just contain all of the people
        in the many images in the input list.
    '''

    total_people_list = []
    total_bib_list = []

    for image_path in image_paths:
        people_list, bibs_list = prediction_pipeline(image_path)
        total_people_list += people_list
        total_bib_list += bibs_list

    return total_people_list, total_bib_list

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# gender predicition pipeline 
# (combines model_people results and model_gender)
def gender_predictions(loaded_model, people_list):

    '''
        Just like with bibs_list and people_list (returned by the prediction pipeline functions), 
        gender_list will share matching indicies with the person image it predicts a gender on.
    '''

    gender_list = [] 
    for people in people_list:
       
       # Apply the transformations
        input_tensor = preprocess(people)
        input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

        # Make the prediction
        with torch.no_grad():
            output = loaded_model(input_batch)

        # Get the predicted label
        class_names = ['man', 'woman']
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_names[predicted_idx.item()]

        gender_list.append(predicted_label)

    return gender_list

# ----------------------------------------------------------------------------------------------------------------------------