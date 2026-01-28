from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

from utils.quality_classifier import classify_grade_by_aspect_ratio

model = load_model("../models/cnn_model.h5")

def predict_image_with_quality(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    type_index = int(np.argmax(preds))

    quality = classify_grade_by_aspect_ratio(img_path)

    print("Rice Type Index:", type_index)
    print("Quality:", quality)

predict_image_with_quality(
    r"C:\Users\DELL\Desktop\major_ps2\Rice_Image_Dataset\Basmati\Basmati (1001).jpg"
)
