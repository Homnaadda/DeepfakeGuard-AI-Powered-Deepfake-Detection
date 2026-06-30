
import os
import numpy as np
import librosa
import tensorflow as tf
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.video.io.VideoFileClip import VideoFileClip
import tempfile
import matplotlib.pyplot as plt
import io

# Constants from the audio project
SAMPLE_RATE = 22050
DURATION = 4
SAMPLES = SAMPLE_RATE * DURATION
N_MELS = 128

def generate_audio_visualizations(audio, output_path):
    """
    Generate audio analysis plots and save to file.
    """
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Waveform
        librosa.display.waveshow(audio, sr=SAMPLE_RATE, ax=ax1, color='#2DD4BF')
        ax1.set_title('Waveform', color='#F8FAFC', fontsize=12)
        ax1.grid(True, alpha=0.2)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(mel_spec_db, sr=SAMPLE_RATE, ax=ax2, cmap='viridis')
        ax2.set_title('Mel Spectrogram', color='#F8FAFC', fontsize=12)
        ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error generating visualization: {e}")
        return False

def extract_audio_from_video(video_path, output_audio_path):
    """
    Extracts audio from a video file and saves it as a WAV file.
    """
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            return False
        video.audio.write_audiofile(output_audio_path, logger=None)
        video.close()
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False

def load_and_preprocess_audio(audio_path):
    """
    Load and preprocess audio file for the model.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Consistent padding or truncation
        if len(audio) < SAMPLES:
            audio = np.pad(audio, (0, SAMPLES - len(audio)))
        else:
            audio = audio[:SAMPLES]
        
        return audio
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

def extract_features(audio):
    """
    Extract Mel Spectrogram features matching the training logic.
    """
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            hop_length=512,
            n_fft=2048,
            fmin=20,
            fmax=8000
        )
        # Log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalization
        mean = np.mean(mel_spec_db)
        std = np.std(mel_spec_db)
        if std > 0:
            mel_spec_db = (mel_spec_db - mean) / std
            
        return mel_spec_db
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

_audio_model = None

def load_audio_model(model_path):
    global _audio_model
    if _audio_model is None:
        try:
            _audio_model = tf.keras.models.load_model(model_path)
            print(f"Audio Mode loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load audio model: {e}")
    return _audio_model

def predict_audio(model, audio_features):
    """
    Returns (label, confidence)
    label: "FAKE" or "REAL"
    confidence: float (0-100)
    """
    try:
        # Prepare shape: (1, 128, 173, 1) - approximate shape depending on time steps
        # The model expects (batch, height, width, channels) usually for 2D CNNs on spectrograms
        # Let's check the audio project again if needed, but core.py did: np.expand_dims(features, axis=(0, -1))
        
        features_model = np.expand_dims(audio_features, axis=(0, -1))
        
        prediction = model.predict(features_model, verbose=0)[0][0]
        prediction_float = float(prediction)
        
        # Audio Project Logic: label = "FAKE" if prediction_float >= 0.5 else "REAL"
        # Confidence logic from audio project: confidence = prediction_float if label == "FAKE" else 1 - prediction_float
        
        if prediction_float >= 0.5:
            label = "FAKE"
            confidence = prediction_float * 100
        else:
            label = "REAL"
            confidence = (1 - prediction_float) * 100
            
        return label, round(confidence, 2)
    except Exception as e:
        print(f"Error in audio prediction: {e}")
        return "ERROR", 0.0
