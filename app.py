import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import hashlib
import json

app = Flask(__name__, static_folder='static', template_folder='templates')

CORRECTIONS_FILE = "corrections.json"
known_hashes = {}

# Load corrections if available
if os.path.exists(CORRECTIONS_FILE):
    with open(CORRECTIONS_FILE, "r") as f:
        corrected_classifications = json.load(f)
else:
    corrected_classifications = {}

# Load trained ML model
model = tf.keras.models.load_model('model/waste_model.h5')

# Dataset directories
dataset_folder = "dataset"
for folder in ["metal", "organic", "plastic", "temp"]:
    os.makedirs(os.path.join(dataset_folder, folder), exist_ok=True)
app.config['DATASET_FOLDER'] = dataset_folder

# Image preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Route to serve dataset images
@app.route('/dataset/<category>/<filename>')
def dataset_file(category, filename):
    return send_from_directory(os.path.join(app.config['DATASET_FOLDER'], category), filename)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Classify uploaded or captured image
@app.route('/classify', methods=['POST'])
def classify_image_route():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    file_data = file.read()
    file_hash = hashlib.md5(file_data).hexdigest()

    # If user has corrected this image before, reuse that result
    if file_hash in corrected_classifications:
        stored_category, stored_filename = corrected_classifications[file_hash]
        return render_template('pass.html', classification=stored_category,
                               image_category=stored_category,
                               image_filename=stored_filename)

    # Otherwise, save image temporarily
    temp_folder = os.path.join(app.config['DATASET_FOLDER'], "temp")
    os.makedirs(temp_folder, exist_ok=True)
    temp_path = os.path.join(temp_folder, file.filename)
    with open(temp_path, 'wb') as f:
        f.write(file_data)

    # Classify image
    image = preprocess_image(temp_path)
    predictions = model.predict(image)
    class_names = ['metal', 'organic', 'plastic']
    predicted_class = class_names[np.argmax(predictions)]

    # Move to predicted category folder, avoid duplicates
    target_folder = os.path.join(app.config['DATASET_FOLDER'], predicted_class)
    os.makedirs(target_folder, exist_ok=True)
    base_name, ext = os.path.splitext(file.filename)
    new_path = os.path.join(target_folder, file.filename)
    counter = 1
    while os.path.exists(new_path):
        new_filename = f"{base_name}_{counter}{ext}"
        new_path = os.path.join(target_folder, new_filename)
        counter += 1
    os.rename(temp_path, new_path)

    # Cache the result
    known_hashes[file_hash] = (predicted_class, os.path.basename(new_path))

    return render_template('pass.html', classification=predicted_class,
                           image_category=predicted_class,
                           image_filename=os.path.basename(new_path))

# Update feedback from user
@app.route('/update_category', methods=['POST'])
def update_category():
    feedback = request.form.get('feedback')
    current_category = request.form.get('current_category')
    filename = request.form.get('filename')

    image_path = os.path.join(app.config['DATASET_FOLDER'], current_category, filename)
    with open(image_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    if feedback == 'yes':
        message = "Thank you for confirming the classification!"
    else:
        if feedback != current_category:
            current_path = os.path.join(app.config['DATASET_FOLDER'], current_category, filename)
            target_folder = os.path.join(app.config['DATASET_FOLDER'], feedback)
            os.makedirs(target_folder, exist_ok=True)
            new_path = os.path.join(target_folder, filename)
            counter = 1
            while os.path.exists(new_path):
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{counter}{ext}"
                new_path = os.path.join(target_folder, new_filename)
                counter += 1
            os.rename(current_path, new_path)

            corrected_classifications[file_hash] = (feedback, os.path.basename(new_path))
            with open(CORRECTIONS_FILE, "w") as f:
                json.dump(corrected_classifications, f)

            message = f"Thank you! The image has been updated to the correct category: {feedback.capitalize()}."
        else:
            message = "Thank you for confirming the classification!"

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Feedback Received</title>
      <link rel="stylesheet" href="{{{{ url_for('static', filename='pass.css') }}}}"> 
    </head>
    <body>
      <div class="pass-container">
        <h1>{message}</h1>
        <button class="scan-another" onclick="window.location.href='/'">Scan Another</button>
      </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
