from fastapi import FastAPI, Request, Form, UploadFile, File, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse, Response, FileResponse
import shutil
import os
import time
import cv2
import face_recognition
from PIL import Image as pImage
import torch
import numpy as np
from typing import List

from . import config, utils, ml_models, ai_report, audio_utils

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key="super-secret-key-change-this")

# Mount static directories
app.mount("/static", StaticFiles(directory=os.path.join(config.BASE_DIR, "app/static")), name="static")
app.mount("/uploaded_images", StaticFiles(directory=config.UPLOADED_IMAGES_DIR), name="uploaded_images")
app.mount("/uploaded_videos", StaticFiles(directory=config.UPLOADED_VIDEOS_DIR), name="uploaded_videos")

templates = Jinja2Templates(directory=os.path.join(config.BASE_DIR, "app/templates"))

@app.get("/", name="home")
async def index(request: Request):
    request.session.clear()
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video-detection", name="video_detection")
async def video_detection(request: Request):
    return templates.TemplateResponse("video_detection.html", {"request": request})

@app.post("/video-detection", name="video_detection_post")
async def video_detection_post(request: Request, upload_video_file: UploadFile = File(...), sequence_length: int = Form(...)):
    if not utils.allowed_video_file(upload_video_file.filename):
         return templates.TemplateResponse("video_detection.html", {"request": request, "error": "Only video files are allowed"})
    
    ext = upload_video_file.filename.split('.')[-1]
    saved_video_file = 'uploaded_file_' + str(int(time.time())) + "." + ext
    saved_path = os.path.join(config.UPLOADED_VIDEOS_DIR, saved_video_file)
    
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(upload_video_file.file, buffer)
        
    request.session['file_name'] = saved_path
    request.session['sequence_length'] = sequence_length
    
    return RedirectResponse(url=app.url_path_for("predict"), status_code=303)

@app.get("/image-detection", name="image_detection")
async def image_detection(request: Request):
    return templates.TemplateResponse("image_detection.html", {"request": request})

@app.get("/audio-detection", name="audio_detection")
async def audio_detection(request: Request):
    return templates.TemplateResponse("audio_detection.html", {"request": request})

@app.post("/image-detection", name="image_detection_post")
async def image_detection_post(request: Request, upload_image_file: UploadFile = File(...)):
    ext = upload_image_file.filename.split('.')[-1].lower()
    if ext not in ['jpg', 'jpeg', 'png']:
         return templates.TemplateResponse("image_detection.html", {"request": request, "error": "Only JPG or PNG images are allowed"})
    
    saved_image_file = 'uploaded_image_' + str(int(time.time())) + "." + ext
    saved_path = os.path.join(config.UPLOADED_IMAGES_DIR, saved_image_file)
    
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(upload_image_file.file, buffer)
        
    request.session['image_file_name'] = saved_path
    
    return RedirectResponse(url=app.url_path_for("predict_image_page"), status_code=303)

@app.post("/audio-detection", name="audio_detection_post")
async def audio_detection_post(request: Request, upload_audio_file: UploadFile = File(...)):
    if not (upload_audio_file.filename.lower().endswith('.wav') or upload_audio_file.filename.lower().endswith('.mp3')):
         return templates.TemplateResponse("audio_detection.html", {"request": request, "error": "Only WAV or MP3 files are allowed"})
    
    # Save the uploaded audio
    saved_audio_file = 'uploaded_audio_' + str(int(time.time())) + os.path.splitext(upload_audio_file.filename)[1]
    saved_path = os.path.join(config.UPLOADED_VIDEOS_DIR, saved_audio_file)
    
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(upload_audio_file.file, buffer)
        
    request.session['audio_file_name'] = saved_path
    return RedirectResponse(url=app.url_path_for("predict_audio_page"), status_code=303)

@app.get("/predict", name="predict")
async def predict_page(request: Request):
    if 'file_name' not in request.session:
        return RedirectResponse(url=app.url_path_for("home"))
    
    video_file = request.session['file_name']
    sequence_length = request.session['sequence_length']
    
    video_file_name = os.path.basename(video_file)
    video_file_name_only = os.path.splitext(video_file_name)[0]
    
    # URL for the video to be displayed in the template (relative to bound mount)
    # /uploaded_videos/filename.mp4
    original_video_url = app.url_path_for("uploaded_videos", path=video_file_name)

    # 1. Load Model
    if utils.device == "cuda":
        model = ml_models.Model(2).cuda()
    else:
        model = ml_models.Model(2).cpu()
    
    model_name_path = utils.get_accurate_model(sequence_length)
    if not model_name_path or not os.path.exists(model_name_path):
         return templates.TemplateResponse("index.html", {"request": request, "error": "No model found for this sequence length"})

    model.load_state_dict(torch.load(model_name_path, map_location=torch.device(utils.device)))
    model.eval()
    
    # 2. Preprocessing & Extraction (Unified for UI and Inference)
    start_time = time.time()
    preprocessed_images = []
    faces_cropped_images = []
    
    cap = cv2.VideoCapture(video_file)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    print(f"Number of frames: {len(frames)}")
    
    padding = 40
    faces_found = 0
    valid_face_tensors = []
    
    # Uniformly space out sampling across the video if possible
    step = max(1, len(frames) // int(sequence_length * 1.5)) if len(frames) > 0 else 1
    
    for i in range(0, len(frames), step):
        if faces_found >= sequence_length:
            break
            
        frame = frames[i]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 0:
            continue
            
        top, right, bottom, left = face_locations[0]
        try:
            # Add padding just like utils.predict_image
            frame_face = frame[max(0, top - padding):min(frame.shape[0], bottom + padding), 
                               max(0, left - padding):min(frame.shape[1], right + padding)]
            rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
            img_face_rgb = pImage.fromarray(rgb_face, 'RGB')
        except:
            # Fallback to strict bounds if padding exceeds image
            frame_face = frame[top:bottom, left:right]
            rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
            img_face_rgb = pImage.fromarray(rgb_face, 'RGB')
            
        transformed_face = utils.train_transforms(rgb_face)
        
        # Save preprocessed image (Full frame)
        image_name_pre = f"{video_file_name_only}_preprocessed_{faces_found+1}.png"
        image_path_pre = os.path.join(config.UPLOADED_IMAGES_DIR, image_name_pre)
        img_rgb = pImage.fromarray(rgb_frame, 'RGB')
        img_rgb.save(image_path_pre)
        preprocessed_images.append(app.url_path_for("uploaded_images", path=image_name_pre))

        # Save cropped face
        image_name_crop = f"{video_file_name_only}_cropped_faces_{faces_found+1}.png"
        image_path_crop = os.path.join(config.UPLOADED_IMAGES_DIR, image_name_crop)
        img_face_rgb.save(image_path_crop)
        faces_cropped_images.append(app.url_path_for("uploaded_images", path=image_name_crop))
        
        valid_face_tensors.append(transformed_face)
        faces_found += 1

    if faces_found == 0:
        return templates.TemplateResponse("predict.html", {"request": request, "no_faces": True})

    # Duplicate last frame if video ended before we got sequence_length faces
    while len(valid_face_tensors) < sequence_length:
        valid_face_tensors.append(valid_face_tensors[-1])
        
    # Stack into expected shape: (1, seq_length, c, h, w)
    frames_tensor = torch.stack(valid_face_tensors).unsqueeze(0)

    # 3. Prediction
    heatmap_images = []
    output = ""
    confidence = 0.0

    try:
        prediction = utils.predict(model, frames_tensor, video_file_name_only)
        confidence = round(prediction[1], 1)
        output = "REAL" if prediction[0] == 1 else "FAKE"
        
        # --- Audio Prediction Integration (Video Flow) ---
        audio_output = "N/A"
        audio_confidence = 0.0
        
        try:
            audio_file_path = os.path.splitext(video_file)[0] + ".wav"
            has_audio = audio_utils.extract_audio_from_video(video_file, audio_file_path)
            
            if has_audio:
                processed_audio = audio_utils.load_and_preprocess_audio(audio_file_path)
                if processed_audio is not None:
                     # Generate standard prediction
                    audio_features = audio_utils.extract_features(processed_audio)
                    if audio_features is not None:
                        audio_model_path = os.path.join(config.MODELS_DIR, "best_model_v2.h5")
                        tf_model = audio_utils.load_audio_model(audio_model_path)
                        if tf_model:
                             audio_output, audio_confidence = audio_utils.predict_audio(tf_model, audio_features)
        except Exception as e:
            print(f"Audio analysis failed in video flow: {e}")
        # ------------------------

        # Save session data for report
        request.session['prediction_result'] = output
        request.session['confidence_score'] = confidence
        request.session['audio_prediction_result'] = audio_output
        request.session['audio_confidence_score'] = audio_confidence
        # distinct from 'file_name' which is the video path
        
        context = {
            "request": request,
            "preprocessed_images": preprocessed_images,
            "faces_cropped_images": faces_cropped_images,
            "heatmap_images": heatmap_images,
            "original_video": original_video_url,
            "output": output,
            "output": output,
            "confidence": confidence,
            "audio_output": audio_output,
            "audio_confidence": audio_confidence
        }
        return templates.TemplateResponse("predict.html", context)
        
    except Exception as e:
        print(f"Exception detected: {e}")
        return templates.TemplateResponse("base.html", {"request": request}) # Fallback or error page

@app.get("/predict_audio_page", name="predict_audio_page")
async def predict_audio_page(request: Request):
    if 'audio_file_name' not in request.session:
        return RedirectResponse(url=app.url_path_for("home"))
        
    audio_path = request.session['audio_file_name']
    
    audio_output = "N/A"
    audio_confidence = 0.0
    visualization_url = ""
    
    try:
        processed_audio = audio_utils.load_and_preprocess_audio(audio_path)
        if processed_audio is not None:
            # 1. Prediction
            audio_features = audio_utils.extract_features(processed_audio)
            audio_model_path = os.path.join(config.MODELS_DIR, "best_model_v2.h5")
            tf_model = audio_utils.load_audio_model(audio_model_path)
            
            if tf_model and audio_features is not None:
                 audio_output, audio_confidence = audio_utils.predict_audio(tf_model, audio_features)
            
            # 2. Visualizations
            vis_filename = os.path.splitext(os.path.basename(audio_path))[0] + "_vis.png"
            vis_path = os.path.join(config.UPLOADED_IMAGES_DIR, vis_filename)
            if audio_utils.generate_audio_visualizations(processed_audio, vis_path):
                visualization_url = app.url_path_for("uploaded_images", path=vis_filename)
                
    except Exception as e:
        print(f"Audio Flow Error: {e}")

    # Store for standalone report
    request.session['audio_prediction_result_standalone'] = audio_output
    request.session['audio_confidence_score_standalone'] = audio_confidence
    request.session['audio_vis_url'] = visualization_url
    
    return templates.TemplateResponse("predict_audio.html", {
        "request": request,
        "audio_output": audio_output,
        "audio_confidence": audio_confidence,
        "visualization_url": visualization_url,
        "audio_filename": os.path.basename(audio_path)
    })

@app.get("/predict_image_page", name="predict_image_page")
async def predict_image_page(request: Request):
    if 'image_file_name' not in request.session:
        return RedirectResponse(url=app.url_path_for("home"))
    
    image_file = request.session['image_file_name']
    image_file_name = os.path.basename(image_file)
    original_image_url = app.url_path_for("uploaded_images", path=image_file_name)
    
    try:
        sequence_length = 10
        model_name_path = utils.get_accurate_model(sequence_length)
        if not model_name_path or not os.path.exists(model_name_path):
             return templates.TemplateResponse("index.html", {"request": request, "error": "No model found for this sequence length"})

        if utils.device == "cuda":
            model = ml_models.Model(2).cuda()
        else:
            model = ml_models.Model(2).cpu()
            
        model.load_state_dict(torch.load(model_name_path, map_location=torch.device(utils.device)))
        model.eval()

        prediction_result, confidence, cropped_face_name = utils.predict_image(model, image_file, image_file_name)
        
        if prediction_result == "NO_FACE":
             return templates.TemplateResponse("predict_image.html", {"request": request, "no_faces": True})

        cropped_face_url = app.url_path_for("uploaded_images", path=cropped_face_name) if cropped_face_name else ""
        output = "REAL" if prediction_result == 1 else "FAKE"
        confidence_val = round(confidence, 1)

        request.session['image_prediction_result'] = output
        request.session['image_confidence_score'] = confidence_val

        context = {
            "request": request,
            "original_image": original_image_url,
            "cropped_face": cropped_face_url,
            "output": output,
            "confidence": confidence_val
        }
        return templates.TemplateResponse("predict_image.html", context)
        
    except Exception as e:
        print(f"Exception detected in image flow: {e}")
        return templates.TemplateResponse("base.html", {"request": request})

@app.get("/download_report", name="download_report")
async def download_report(request: Request):
    
    # CASE 1: Full Video + Audio Report
    if 'file_name' in request.session and 'prediction_result' in request.session:
        video_file = request.session['file_name']
        video_filename = os.path.basename(video_file)
        video_file_name_only = os.path.splitext(video_filename)[0]
        
        result = request.session['prediction_result']
        confidence = request.session['confidence_score']
        
        audio_result = request.session.get('audio_prediction_result', 'N/A')
        audio_confidence = request.session.get('audio_confidence_score', 0.0)
        
        # Retrieve preprocessed images
        import glob
        pattern = os.path.join(config.UPLOADED_IMAGES_DIR, f"{video_file_name_only}_preprocessed_*.png")
        image_paths = glob.glob(pattern)
        try:
            image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
        except:
            pass
        
        ai_text = ai_report.generate_ai_analysis(video_filename, result, confidence, audio_result, audio_confidence)
        report_filename = f"Report_{video_filename}_{int(time.time())}.pdf"
        report_path = os.path.join(config.UPLOADED_IMAGES_DIR, report_filename)
        
        success = ai_report.create_pdf_report(video_filename, result, confidence, audio_result, audio_confidence, ai_text, image_paths, report_path)
        
        if success:
             return FileResponse(report_path, media_type='application/pdf', filename=report_filename)
        else:
             return Response("Failed to generate report", status_code=500)

    # CASE 2: Audio Only Report
    elif 'audio_file_name' in request.session:
        audio_path = request.session['audio_file_name']
        audio_filename = os.path.basename(audio_path)
        
        audio_result = request.session.get('audio_prediction_result_standalone', 'N/A')
        audio_confidence = request.session.get('audio_confidence_score_standalone', 0.0)
        vis_url = request.session.get('audio_vis_url', "")
        
        # Mocking video result as N/A
        result = "N/A"
        confidence = 0.0
        
        # If we have a visualization image, we can include it as evidence
        image_paths = []
        if vis_url:
            # Convert URL back to path
            vis_filename = vis_url.split('/')[-1]
            vis_path = os.path.join(config.UPLOADED_IMAGES_DIR, vis_filename)
            if os.path.exists(vis_path):
                image_paths.append(vis_path)

        ai_text = ai_report.generate_ai_analysis(audio_filename, result, confidence, audio_result, audio_confidence)
        
        report_filename = f"Report_{audio_filename}_{int(time.time())}.pdf"
        report_path = os.path.join(config.UPLOADED_IMAGES_DIR, report_filename)
        
        success = ai_report.create_pdf_report(audio_filename, result, confidence, audio_result, audio_confidence, ai_text, image_paths, report_path)
        
        if success:
             return FileResponse(report_path, media_type='application/pdf', filename=report_filename)
        else:
             return Response("Failed to generate report", status_code=500)

    # CASE 3: Image Only Report
    elif 'image_file_name' in request.session and 'image_prediction_result' in request.session:
        image_path = request.session['image_file_name']
        image_filename = os.path.basename(image_path)
        
        result = request.session['image_prediction_result']
        confidence = request.session['image_confidence_score']
        
        audio_result = "N/A"
        audio_confidence = 0.0
        
        image_paths = []
        if os.path.exists(image_path):
            image_paths.append(image_path)
            
        ai_text = ai_report.generate_ai_analysis(image_filename, result, confidence, audio_result, audio_confidence)
        
        report_filename = f"Report_{image_filename}_{int(time.time())}.pdf"
        report_path = os.path.join(config.UPLOADED_IMAGES_DIR, report_filename)
        
        success = ai_report.create_pdf_report(image_filename, result, confidence, audio_result, audio_confidence, ai_text, image_paths, report_path)
        
        if success:
             return FileResponse(report_path, media_type='application/pdf', filename=report_filename)
        else:
             return Response("Failed to generate report", status_code=500)

    else:
        return RedirectResponse(url=app.url_path_for("home"))


@app.get("/detection", name="detection")
async def detection(request: Request):
    return templates.TemplateResponse("detection.html", {"request": request})

@app.get("/introduction", name="introduction")
async def introduction(request: Request):
    return templates.TemplateResponse("introduction.html", {"request": request})
