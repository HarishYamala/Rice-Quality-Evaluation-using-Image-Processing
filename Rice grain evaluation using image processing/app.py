from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import os

app = Flask(__name__)

# Load trained CNN model
model = load_model("models/cnn_model.h5")

# Rice type labels (must match training folders)
type_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# ------------------ Quality Classification ------------------
def classify_quality(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return "Unknown"

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    if min(w, h) == 0:
        return "Unknown"

    aspect_ratio = max(w, h) / min(w, h)

    if aspect_ratio >= 3.0:
        return "Slender"
    elif aspect_ratio >= 2.1:
        return "Medium"
    elif aspect_ratio >= 1.1:
        return "Bold"
    else:
        return "Round"

# ------------------ Routes ------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        os.makedirs("uploads", exist_ok=True)
        img_path = os.path.join("uploads", file.filename)
        file.save(img_path)

        # ---------- CNN Prediction (Rice Type) ----------
        img = Image.open(img_path).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        type_index = int(np.argmax(preds))
        rice_type = type_labels[type_index]

        # ---------- Quality Prediction ----------
        rice_quality = classify_quality(img_path)

        return jsonify({
            'quality': rice_quality,
            'type': rice_type
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
