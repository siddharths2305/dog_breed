<!-- 🐾 DARK MODE ANIME AESTHETIC README FOR DOG BREED CLASSIFICATION ML MODEL -->

# 🐶 Dog Breed Classification using CNN  
**Deep Learning Model 🧠 | Image Recognition 📸 | TensorFlow-Powered 🔥**

<p align="center">
  <img 
    src="https://github.com/siddharths2305/dog-breed-classification/blob/main/assets/dog.gif?raw=true" 
    width="300" 
    style="border-radius: 50%; box-shadow: 0px 0px 30px #00ffff; object-fit: cover;"
    alt="Dog Classification Anime GIF"
  />
</p>

> 🧩 **Classifies dog breeds** using a Convolutional Neural Network (CNN) trained with TensorFlow & Keras.  
> Supports **image upload prediction** and ready for **Streamlit or Flask deployment**.

---

## 🚀 Features
- 📂 Trains CNN model for dog breed classification  
- 🧠 Uses TensorFlow/Keras with data augmentation  
- 💾 Saves model in `.h5` and `.pkl` formats  
- 🐍 Predicts breeds directly from uploaded images  
- 🌐 Streamlit/Flask compatible for web deployment  

---

## ⚙️ Tech Stack

<div align="center">

<img src="https://img.shields.io/badge/Python-000000?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/TensorFlow-000000?style=for-the-badge&logo=tensorflow&logoColor=FF6F00">
<img src="https://img.shields.io/badge/Keras-000000?style=for-the-badge&logo=keras&logoColor=D00000">
<img src="https://img.shields.io/badge/OpenCV-000000?style=for-the-badge&logo=opencv&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-000000?style=for-the-badge&logo=streamlit&logoColor=FF4B4B">
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white">

</div>

---

## 🧩 Model Architecture

Input (224x224x3)
↓
Conv2D + ReLU
↓
MaxPooling
↓
Conv2D + ReLU
↓
MaxPooling
↓
Flatten
↓
Dense(128) + ReLU
↓
Dropout(0.5)
↓
Dense(output_classes, softmax)

---

## 🧠 Training Example

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('dog_breed_model.h5')

def predict_breed(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    breed = np.argmax(pred)
    print(f"Predicted Breed ID: {breed}")


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=20, validation_data=val_data)
model.save('dog_breed_model.h5')
