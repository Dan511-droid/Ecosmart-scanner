🛠 Technology Stack
Backend: Python, Flask
Frontend: HTML5, CSS3, JavaScript (Vanilla)
Machine Learning: TensorFlow, Keras
Image Processing: Pillow (PIL)
Dataset Storage: Local file system (Metal / Organic / Plastic folders)

========================================

Clone the repository
git clone https://github.com/Dan511-droid/Ecosmart-scanner.git
cd ecosmart-waste-management

Create a virtual environment & install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Ensure model is available
Place your trained waste_model.h5 file in the model/ folder.

Run the Flask app
python app.py

==============================================

ecosmart-waste-management/
│
├── app.py                  # Main Flask application
├── model/
│   └── waste_model.h5      # Trained TensorFlow model
├── dataset/
│   ├── metal/
│   ├── organic/
│   ├── plastic/
│   └── temp/
├── static/
│   ├── styles.css          # Main CSS file
│   ├── pass.css            # CSS for results page
│   ├── favicon.ico
│   └── ...
├── templates/
│   ├── index.html          # Main upload/capture page
│   └── pass.html           # Classification result page
├── corrections.json        # Stores corrected classifications
├── requirements.txt
└── README.md

====================================================================

How It Works

User uploads or captures an image
Desktop: Choose between Upload or Capture
Mobile: Single button offering both camera and gallery options
Image is sent to Flask backend
Image is preprocessed to 150x150px and normalized
ML model predicts category: Metal / Organic / Plastic
Image is stored in the corresponding dataset folder
Result is displayed to the user with preview
Feedback option allows corrections, which are saved for retraining

