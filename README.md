# Medical-Checkup-Ai-Tool
An intelligent and interactive medical image-based diagnostic tool built with **Python**, **Streamlit**, **Keras**, and **Google Teachable Machine**. This prototype is capable of identifying signs of Pneumonia and multiple eye-related diseases from X-ray and retinal images respectively. It serves as an early detection assistant by combining fast image classification with an intuitive GUI.

---

## 🩺 Features

* Upload and analyze medical images (X-ray images or Eye Retina scans)
* Predict conditions such as:
  * Pneumonia (from chest X-rays)
  * Eye diseases (e.g., Myopia, Glaucoma, Retinopathy)
* Uses pre-trained image classification models triained from medical datasets via Google Teachable Machine.
* Runs entirely on the web via Streamlit with no heavy computation required on User-end
* Responsive interface with diagnosis feedback and probability

---

## 🧠 How It Works

### 1. **Data Collection & Model Training**
* Medical datasets sourced from **Mendeley Data**
* Labeled data split into training, validation, and test sets
* Model trained using Google Teachable Machine

  * No-code model training platform
  * Output: downloadable `.h5`/TensorFlow Lite model or web exportable model

---

### 2. **Image Upload & Preprocessing**

```python
# Upload Chest X-Ray File
chest_x_ray = st.file_uploader(' ', type=['jpeg', 'jpg', 'png'], key='1')
image_classification(chest_x_ray, pneumonia_model, pneumonia_classes)

# Upload Eye Test File
eye_test = st.file_uploader(' ', type=['jpeg', 'jpg', 'png'], key='2')
image_classification(eye_test, eye_model, eye_classes)
```
```python
# Convert image to (224, 224)
image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
```
* Resize input image to expected model input size
* Normalize pixel values between 0 and 1

---

### 3. **Model Prediction**

```python
# Make prediction
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = round(prediction[0][index], 1)*100
```
---

### 4. **Output & Visualization**

```python
def image_classification(image_file, model, class_names):
    if image_file is not None:
        image  = Image.open(image_file).convert('RGB')
        st.image(image, use_container_width =True)

        # Classify Images
        class_name, conf_score = classify(image, model, class_names)
        
        # Write Classfications
        st.write("## {}".format(class_name))
        st.write("### Score: {}%".format(conf_score))
```

* Displays result with corresponding confidence score
* Allows quick visual feedback for potential diagnosis

---

## 🛠️ Tech Stack

* Python 3.10+
* Streamlit
* Teachable Machine (for model training)
* TensorFlow / Keras
* OpenCV / Pillow (for image handling)

---

## 💻 Run the App

```bash
streamlit run main.py
```

---

## 📁 Folder Structure

```
├── main.py                     # Streamlit application code
├── utils.py                    # Main funtions of image preprocessing and classification are stored here
├── requirements.txt            # Python dependencies
├── background.jpg              # Background image for the app
├── Models
    ├── eye_disease_classifier.h5   # Eye Retina Image Classification Model
    ├── eye_disease_labels.txt      # Labels for Eye Image Classification Model
    ├── pneumonia_classifier.h5     # Pneumonia Classification Model
    ├── pneumonia_labels.txt        # Labels forPneumonia Classification Model
└── README.md                   # Project documentation
```

---

## 📌 Limitations

* Model accuracy is limited by dataset scope (prototype only)
* Does not replace real medical diagnosis
* Requires higher-quality datasets and validation for production usage
---

## 📄 License
* MIT License. For research and educational use only.

> Built with a passion for merging healthcare and AI during free hours. Always room to grow with better data and collaborative feedback.
