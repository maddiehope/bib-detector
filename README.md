# Bib Detector
[Visit this application's domain to try it out!](https://app.bib-detector.dynv6.net)

### Overview
**Bib Detector** addresses a common challenge faced by race managers – accurately **identifying and recording the bib numbers and genders of runners as they cross the finish line**. Manual recognition can lead to errors in race results and prolonged race management processes. This application leverages computer vision and deep learning to automate the classification of bib numbers and genders, providing a more efficient and accurate solution for race reporting. By combining **YOLOv8n**, **ResNet50**, and **OCR** for video analysis, the application can efficiently process video frames, ensuring precise recognition of bib numbers and genders.

### Process

##### Data Collection:
**Datasets used in this application:**
- **'bib-numbers'**- I combined two datasets: [OCR Racing Bib Number Recognition](https://www.kaggle.com/datasets/trainingdatapro/ocr-race-numbers/) and [RBNR](https://people.csail.mit.edu/talidekel/RBNR.html).
- **'gender'** - I used a [dataset downloaded from Kraggle](https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset). 
Using the linked datasets, I gathered two  diverse datasets of images. One contains runners with visible bib numbers, and the other contains images of faces and their sex. 
These datasets were used for the training and testing of the models used.

##### Object Detection: 
I utilized the object detection YOLO to localize the regions of the bib numbers in the images or video frames. 
This step involved training the object detection model to accurately identify and outline the areas where the bib numbers are located.

##### Bib Number Recognition:
I used the tesseract OCR to recognize the extracted bib numbers from the localized regions. 

##### Sex Classification:
I trained an image classification model (ResNet50) to classify the sex of the runners in the images or video frames. 

##### Integration: 
I integrated the object detection model for bib number localization,
the bib number OCR, and the sex classification model into a unified
pipeline for processing images or video frames. 
This integrated pipeline is capable of detecting bib numbers, recognizing the digits on the bibs, and
classifying the sex of the individuals.

##### Flask API Development & Deployment: 
I created a Flask application to serve as the API
endpoint for the model. This involves setting up routes to handle video uploads,
processing the video frames, and the results of bib numbers and genders to the user via email.

Video processing logic is within the Flask
application and extracts frames from the uploaded video and applies the integrated model
pipeline to each frame. This detects bib numbers and classifies genders, aggregating results into a Excel spreadsheet.

Oracle Cloud is the server for my application and it can be accessed via the link provided above. 

### Future Improvements

- My Oracle Cloud server has limited capabilies and thus large files are not accomadated as uploads. I already paid for increased storage so this issue is kind of a brick wall. All of my functions are tested in 'bib-detector.ipynb' and work perfectly, there is just red tape when trying to upload them through the flask app. I hope to resolve this so that users can actually use my application via the domain.
- One big issue is that the gender classification model tends to make a lot of incorrect predicitions on women. This probably has to do with the fact that the pictures the model is being used on are of athletes running, while it was trained on regular images of men and women (mostly snapshots from TV/smiling for pictures). In the future, a more specific dataset to this application could be curated.
- Another major improvement that can be made is the OCR accuracy. It was difficult for me to fine tune the image pre-processing, and I will look into more ways the recognition abilities can be improved.
- My webpage styling was not the best, but this is becuase I have not had a ton of experience with html/css. Hopefully one day I can go back and make everything look nicer on the front end. 
