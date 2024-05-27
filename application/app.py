from flask import Flask, render_template, request, send_from_directory
import os
import whisper
from werkzeug.utils import secure_filename
import wave

model = whisper.load_model("tiny.en")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_data' not in request.files:
        return 'No file part', 400
    
    file = request.files['audio_data']

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    initial_prompt = ""
    
    result = model.transcribe(filepath,initial_prompt = initial_prompt, fp16=False)
    return result["text"], 200
