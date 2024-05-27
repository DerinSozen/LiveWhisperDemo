from flask import Flask, render_template, request
import os
import whisper
from werkzeug.utils import secure_filename
import wave
from datetime import datetime
import scipy.io
from scipy.io import wavfile

load_start_time = datetime.now()
model = whisper.load_model("tiny.en")
load_end_time = datetime.now()
time_difference = (load_end_time - load_start_time).total_seconds() * 10**3
print(f"Model loaded in {time_difference} ms")

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
    
    response_start_time = datetime.now()
    file = request.files['audio_data']

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    initial_prompt = ""
    
    result = model.transcribe(filepath,initial_prompt = initial_prompt, fp16=False)
    response_end_time = datetime.now()
    latency = (response_end_time - response_start_time).total_seconds() * 10**3
    print(f"latency {latency} ms")
    
    return result["text"], 200
