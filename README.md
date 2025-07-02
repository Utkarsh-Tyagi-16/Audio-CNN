# Audio CNN Visualizer

![Audio CNN 1](audio-cnn-1.png)
![Audio CNN 2](audio-cnn-2.png)

## Overview
Audio CNN Visualizer is a full-stack application that allows users to upload audio files (WAV format), sends them to a deep learning backend for inference, and visualizes the model's predictions and feature maps in an interactive, modern web UI. The frontend is built with Next.js and the backend is served via a Python API (e.g., FastAPI, Modal, etc.).

---

## Features
- Drag-and-drop or file upload for audio files
- Real-time visualization of model predictions with confidence scores and emojis
- Interactive display of feature maps and audio waveforms
- Responsive, modern UI with mobile and desktop support
- Error handling and user feedback
- Easy integration with any backend API endpoint

---

## Theory: How CNNs Work for Audio Classification

### 1. **Audio Preprocessing**
- **Raw audio** is first loaded and normalized.
- The waveform is converted into a **spectrogram** (a 2D representation of frequency vs. time) using Short-Time Fourier Transform (STFT) or Mel-Frequency Cepstral Coefficients (MFCCs).
- The spectrogram acts like an "image" input for the CNN.

### 2. **Convolutional Neural Networks (CNNs) for Audio**
- **Convolutional layers** slide learnable filters over the spectrogram to extract local patterns (e.g., frequency bands, temporal changes).
- **Activation functions** (like ReLU) introduce non-linearity.
- **Pooling layers** (e.g., max pooling) reduce dimensionality and help the network focus on the most important features.
- **Deeper layers** learn more abstract representations (e.g., phonemes, timbre, rhythm).

### 3. **Feature Extraction and Classification**
- The final layers of the CNN flatten the feature maps and pass them through one or more **fully connected (dense) layers**.
- The output layer uses **softmax** (for multi-class) or **sigmoid** (for binary) activation to produce class probabilities.
- The class with the highest probability is the predicted label.

### 4. **Visualization**
- **Feature maps** from intermediate CNN layers can be visualized to show what the network is "looking at" in the audio.
- **Waveform and spectrogram** visualizations help users understand the input and the model's focus.

---

## Architecture & Workflow

```mermaid
graph TD;
    User-->|Uploads WAV|Frontend(Next.js App)
    Frontend-->|POST audio (base64 or file)|Backend(Python API)
    Backend-->|Runs CNN inference|Model(CNN Model)
    Model-->|Returns predictions, feature maps, waveform|Backend
    Backend-->|JSON response|Frontend
    Frontend-->|Visualizes results|User
```

### How it Works
1. **User uploads a WAV file** via the web interface.
2. **Frontend** encodes the audio and sends it to the backend API endpoint (configurable via `NEXT_PUBLIC_API_URL`).
3. **Backend** receives the audio, processes it, and runs it through a trained CNN model for audio classification.
4. **Backend** returns a JSON response containing:
    - Top predictions (class, confidence)
    - Feature maps from CNN layers
    - Input spectrogram and waveform data
5. **Frontend** visualizes the predictions, feature maps, and waveform interactively for the user.

---

## Tech Stack
- **Frontend:** Next.js (React), TypeScript, Tailwind CSS, modern UI components
- **Backend:** Python (FastAPI, Flask, or Modal), PyTorch/TensorFlow for CNN model
- **Visualization:** Custom React components for feature maps, waveform, and color scales

---

## Project Structure
- `audio-cnn-visualisation/` — Next.js frontend
- `main.py`, `model.py` — Python backend for inference
- `chirpingbirds.wav` — Example audio file

---

## Getting Started

### Prerequisites
- Node.js (for frontend)
- Python 3.8+ (for backend)

### Frontend Setup
```bash
cd audio-cnn-visualisation
npm install
# Add your backend URL to .env.local
# Example:
# NEXT_PUBLIC_API_URL=https://your-backend-url
npm run dev
```

### Backend Setup
```bash
# (Example for FastAPI)
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## Deployment
- **Backend:** Deploy on Modal, Render, Heroku, or any cloud provider. Expose a POST endpoint for audio inference.
- **Frontend:** Deploy on Vercel or Netlify. Set the `NEXT_PUBLIC_API_URL` environment variable in your frontend deployment to point to your backend.

---

## API Example
**Request:**
```json
POST / (or /predict)
{
  "audio_data": "<base64-encoded-audio>"
}
```
**Response:**
```json
{
  "predictions": [
    {"class": "dog", "confidence": 0.92},
    {"class": "cat", "confidence": 0.05}
  ],
  "visualization": {"conv1": {"shape": [8, 8], "values": [[...], ...]}},
  "input_spectrogram": {"shape": [128, 128], "values": [[...], ...]},
  "waveform": {"values": [...], "sample_rate": 44100, "duration": 2.5}
}
```

---

## Usage
1. Open the frontend in your browser.
2. Upload a WAV file.
3. View predictions, feature maps, and waveform visualizations.
4. Explore the interactive UI for more details.

---

## Troubleshooting
- **Build or Lint Errors:** Ensure your code matches the latest pushed version and all lint errors are fixed before deploying.
- **API Errors:** Check that your backend is running and accessible from the frontend. Update `NEXT_PUBLIC_API_URL` as needed.
- **Large Files:** Do not commit `venv/` or large model files to git. Use `.gitignore` and requirements.txt.

---

