from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import torch
import librosa
import numpy as np
import cv2
from torchvision import transforms
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from PIL import Image
import timm  # Make sure to install timm: pip install timm

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

### -------------------- AUDIO DEEPFAKE DETECTION SETUP -------------------- ###
# Load a pre-trained/fine-tuned Wav2Vec2 model for Audio DeepFake Detection.
model_audio_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_audio_name)
model_audio = Wav2Vec2ForSequenceClassification.from_pretrained(model_audio_name, num_labels=2)
model_audio.eval()

def preprocess_audio(file_path):
    # Loads audio at 16 kHz and processes it for Wav2Vec2.
    audio, sr = librosa.load(file_path, sr=16000)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    return input_values

def detect_fake_voice(audio_path):
    input_values = preprocess_audio(audio_path)
    with torch.no_grad():
        logits = model_audio(input_values).logits
    # Assuming label 1 means FAKE and label 0 means REAL.
    prediction = torch.argmax(logits, dim=-1).item()
    return "FAKE VOICE" if prediction == 1 else "REAL VOICE"

### -------------------- VIDEO DEEPFAKE DETECTION SETUP -------------------- ###
# We now use a fine-tuned Xception model for deepfake video detection.
# The model is loaded via timm and outputs a single logit per input frame.
# A sigmoid activation converts the logit to a probability of being fake.
# Higher probability means higher chance of being fake.

# Load the Xception model. Make sure you have timm installed.
try:
    # Create the model. 'xception' is available in timm.
    model_video = timm.create_model('xception', pretrained=False, num_classes=1)
    model_video_path = "xception_deepfake.pth"  # Update this with your model's weight file.
    if os.path.exists(model_video_path):
        model_video.load_state_dict(torch.load(model_video_path, map_location=torch.device('cpu')))
    else:
        print("Warning: Xception deepfake model not found; using untrained model!")
    model_video.eval()
except Exception as e:
    print("Error loading video model:", e)
    model_video = None

# Define transform for video frame processing.
# Use the normalization values that were used during training.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def detect_fake_video_from_frames(frames, threshold=0.5):
    """
    Process a list of frames and average the predictions.
    Returns "FAKE VIDEO" if the average probability is above threshold.
    """
    if model_video is None:
        return "MODEL NOT LOADED"
        
    predictions = []
    for frame in frames:
        # Convert from BGR (OpenCV) to RGB and then to PIL Image.
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_frame).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            logit = model_video(input_tensor)
        # Convert logit to probability using sigmoid.
        prob = torch.sigmoid(logit).item()
        predictions.append(prob)
    
    # Average the probability over all sampled frames.
    avg_prob = np.mean(predictions)
    return "FAKE VIDEO" if avg_prob > threshold else "REAL VIDEO"

def sample_video_frames(video_path, num_frames=5):
    """
    Samples up to num_frames evenly spaced frames from the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return frames
    # Determine frame indices to sample.
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    current_idx = 0
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in indices:
            frames.append(frame)
        current_idx += 1
    cap.release()
    return frames

### -------------------- FLASK ENDPOINTS -------------------- ###

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No audio file selected"}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    result = detect_fake_voice(file_path)
    return jsonify({"result": result})

@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No video file selected"}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    # Sample multiple frames from the video.
    frames = sample_video_frames(file_path, num_frames=5)
    if not frames:
        return jsonify({"error": "Could not read video file or no frames found"}), 400
    result = detect_fake_video_from_frames(frames, threshold=0.5)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
