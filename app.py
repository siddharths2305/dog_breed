from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import os

# Load class labels mapping from pickle
with open("dog_breed_classifier.pkl", "rb") as f:
    model_info = pickle.load(f)

# Load the trained model
model = load_model("dog_breed_classifier.h5")
class_labels = model_info["class_labels"]

# Flask app
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    uploaded_image = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            uploaded_image = filepath

            # Preprocess uploaded image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict breed
            preds = model.predict(img_array)
            class_idx = np.argmax(preds)
            prediction = class_labels[class_idx]

    return render_template("index.html", prediction=prediction, uploaded_image=uploaded_image)

if __name__ == "__main__":
    app.run(debug=True)
