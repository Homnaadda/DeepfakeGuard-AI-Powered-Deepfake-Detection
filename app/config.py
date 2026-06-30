import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pointing to the local models folder
MODELS_DIR = os.path.join(BASE_DIR, "models")

UPLOADED_VIDEOS_DIR = os.path.join(BASE_DIR, "uploaded_videos")
UPLOADED_IMAGES_DIR = os.path.join(BASE_DIR, "uploaded_images")

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv'}

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, '.env'))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

