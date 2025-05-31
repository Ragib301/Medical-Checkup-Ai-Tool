import streamlit as st
from keras.models import load_model
from PIL import Image
from utils import classify, set_background


# Load Class Names
def class_names(label):
    with open(label, 'r') as f:
        class_name = [a[2:].strip() for a in f.readlines()]
        f.close()
    return class_name

# Display Image & Classify
def image_classification(image_file, model, class_names):
    if image_file is not None:
        image  = Image.open(image_file).convert('RGB')
        st.image(image, use_container_width =True)

        # Classify Images
        class_name, conf_score = classify(image, model, class_names)
        
        # Write Classfications
        st.write("## {}".format(class_name))
        st.write("### Score: {}%".format(conf_score))

# Load Pneumonia Classifier
pneumonia_model = load_model('Models/pneumonia_classifier.h5')
pneumonia_label = f'Models/pneumonia_labels.txt'
pneumonia_classes = class_names(pneumonia_label)

# Load Eye Disease Classifier
eye_model = load_model('Models/eye_disease_classifier.h5')
eye_label = f'Models/eye_disease_labels.txt'
eye_classes = class_names(eye_label)

# Set Background
set_background('background.jpg')

# Set Title
st.title(f"Medical Checkup Ai Tool")

# Set Header
st.header(f"Pneumonia Classification using Chest X-Ray")

# Upload Chest X-Ray File
chest_x_ray = st.file_uploader(' ', type=['jpeg', 'jpg', 'png'], key='1')
image_classification(chest_x_ray, pneumonia_model, pneumonia_classes)

# Set Header
st.header(f"Eye Disease Classification using Eye Test Image")

# Upload Eye Test File
eye_test = st.file_uploader(' ', type=['jpeg', 'jpg', 'png'], key='2')
image_classification(eye_test, eye_model, eye_classes)
