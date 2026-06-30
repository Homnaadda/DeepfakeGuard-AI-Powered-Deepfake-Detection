# DeepFake Video Detection

## Overview
A comprehensive digital forensics platform for detecting Deepfakes. It analyzes videos, images, and audio to detect manipulation using advanced Machine Learning models, and provides professional AI-generated forensic reports.

## Features
- **Video Analysis**: Extracts frames and detects manipulated faces with confidence scores.
- **Image Analysis**: Detects Deepfake manipulation in uploaded images.
- **Audio Analysis**: Analyzes audio tracks for voice synthesis and tampering.
- **AI Forensic Reports**: Generates professional PDF forensic reports detailing findings, technical assessment, threat implications, and recommendations (Powered by Gemini AI).
- **Interactive UI**: User-friendly web interface for file uploads, visualization, and results.

## Technology Stack
- **Frontend**: HTML, CSS, JavaScript (Bootstrap, jQuery)
- **Backend**: Python, FastAPI, Uvicorn, Jinja2
- **AI/ML**: PyTorch, TensorFlow/Keras, OpenCV, face_recognition, librosa
- **Libraries**: numpy, Pillow, ReportLab, python-dotenv
- **External APIs**: Google Gemini API (for report generation)

## Project Architecture
The application is structured as a FastAPI web application. The frontend communicates with the backend via REST endpoints.
When media is uploaded, it is saved locally, preprocessed (e.g., extracting frames, isolating faces, or extracting audio features), and fed into PyTorch (Vision) or TensorFlow (Audio) models. 
The system integrates Google Gemini to parse detection metrics into comprehensive forensic reports, which are generated as PDFs using ReportLab.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DeepFake-Video-detection
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up the Environment Variables (see below).

## Running the Project
To run the server locally, execute the provided batch script or use Uvicorn directly:
```bash
# Using batch file (Windows)
run_app.bat

# Or using uvicorn manually
uvicorn app.main:app --reload
```
The application will be accessible at `http://127.0.0.1:8000`.

## Environment Variables
Create a `.env` file in the root directory based on the `.env.example` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## Folder Structure
```
DeepFake-Video-detection/
├── app/
│   ├── static/           # CSS, JS, and Bootstrap files
│   ├── templates/        # Jinja2 HTML templates
│   ├── ai_report.py      # Gemini AI report generation & PDF logic
│   ├── audio_utils.py    # Audio extraction and preprocessing logic
│   ├── config.py         # App configuration and environment variables
│   ├── main.py           # FastAPI application & routes
│   ├── ml_models.py      # PyTorch model definitions
│   └── utils.py          # Video/Image preprocessing and model inference
├── models/               # Pre-trained .pt and .h5 model checkpoints
├── uploaded_images/      # Directory for temporary image files (Ignored in Git)
├── uploaded_videos/      # Directory for temporary video files (Ignored in Git)
├── .env.example          # Example environment variable file
├── .gitignore            # Git ignore rules for caches, envs, and large files
├── requirements.txt      # Python dependencies
└── run_app.bat           # Batch script to run the app
```

## Large Files Note
This repository relies on several pre-trained model checkpoints (e.g., `.pt` and `.h5` files) located in the `models/` directory. Some of these files exceed 50MB and are ignored by default in `.gitignore`. 
**Recommendation**: Use [Git Large File Storage (LFS)](https://git-lfs.com/) to track these files, or host them on a release page/HuggingFace model hub and download them during setup.

## Screenshots
*(Add your application screenshots here)*
- `[Home Page Screenshot]`
- `[Detection Dashboard Screenshot]`
- `[PDF Report Sample]`

## Future Improvements
- Implement asynchronous task queues (e.g., Celery) for long-running video processing.
- Add user authentication and history tracking.
- Expand support for more media formats.
- Dockerize the application for easier deployment.

## License
MIT License

## Author
*Your Name / Organization*
