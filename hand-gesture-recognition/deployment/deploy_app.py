import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import base64

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the trained model
model_path = r"E:\new yousef\hand-gesture-recognition\models\asl_gesture_recognition_model.h5"
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Define constants
labels = ['hello', 'yes', 'no', 'i love you', 'thank you']
img_height, img_width = 64, 64
CONFIDENCE_THRESHOLD = 85.0  # Increased from 40.0
MIN_HAND_AREA = 1000  # Minimum area for hand detection
MAX_HAND_AREA = 100000  # Maximum area for hand detection

# HSV range for skin detection (adjusted for better skin detection)
lower_skin = np.array([0, 30, 60], dtype=np.uint8)
upper_skin = np.array([20, 150, 255], dtype=np.uint8)

def preprocess_image(img):
    if img is None or img.size == 0:
        raise ValueError("Invalid image data")
    img = np.array(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No hand detected in image")
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)
    if not (MIN_HAND_AREA < area < MAX_HAND_AREA):
        raise ValueError("Hand area outside acceptable range")
    x, y, w, h = cv2.boundingRect(max_contour)
    padding = int(max(w, h) * 0.2)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    hand_region = img[y:y+h, x:x+w]
    hand_mask = mask[y:y+h, x:x+w]
    img_masked = cv2.bitwise_and(hand_region, hand_region, mask=hand_mask)
    if len(img_masked.shape) == 2:
        img_masked = cv2.cvtColor(img_masked, cv2.COLOR_GRAY2BGR)
    img_resized = cv2.resize(img_masked, (img_width, img_height))
    img_normalized = img_resized / 255.0
    return np.expand_dims(img_normalized, axis=0)

def predict_gesture(img):
    try:
        processed_img = preprocess_image(img)
        predictions = model.predict(processed_img, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx] * 100
        
        if confidence < CONFIDENCE_THRESHOLD:
            return "Low confidence", confidence
            
        predicted_label = labels[predicted_idx]
        return predicted_label, confidence
    except ValueError as e:
        return str(e), 0.0
    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'frame' in request.json:
            frame_data = request.json['frame'].split(',')[1]
            frame_bytes = base64.b64decode(frame_data)
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({'error': 'Invalid frame data'}), 400
            predicted_label, confidence = predict_gesture(frame)
            return jsonify({
                'prediction': predicted_label,
                'confidence': float(confidence)
            })

        return jsonify({'error': 'No valid input provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)