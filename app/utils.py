import torch
from torchvision import transforms
from torch import nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
from . import config

# Globals
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean, std), std=np.divide([1, 1, 1], std))

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def allowed_video_file(filename):
    if '.' in filename:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in config.ALLOWED_VIDEO_EXTENSIONS:
            return True
    return False

def get_accurate_model(sequence_length):
    model_name = []
    sequence_model = []
    final_model = ""
    list_models = glob.glob(os.path.join(config.MODELS_DIR, "*.pt"))

    for model_path in list_models:
        model_name.append(os.path.basename(model_path))

    for model_filename in model_name:
        try:
            seq = model_filename.split("_")[3]
            if int(seq) == sequence_length:
                sequence_model.append(model_filename)
        except (IndexError, ValueError):
            pass

    if len(sequence_model) > 1:
        accuracy = []
        for filename in sequence_model:
            acc = filename.split("_")[1]
            accuracy.append(float(acc)) # Ensure this is treated as number
        max_index = accuracy.index(max(accuracy))
        final_model = os.path.join(config.MODELS_DIR, sequence_model[max_index])
    elif len(sequence_model) == 1:
        final_model = os.path.join(config.MODELS_DIR, sequence_model[0])
    else:
        print("No model found for the specified sequence length.")
        return None

    return final_model

def im_convert(tensor, video_file_name):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

def predict(model, img, video_file_name=""):
    fmap, logits = model(img.to(device))
    params = list(model.parameters())
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print('confidence of prediction:', confidence)
    return [int(prediction.item()), confidence]

def plot_heat_map(i, model, img, video_file_name=''):
    fmap, logits = model(img.to(device))
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    idx = np.argmax(logits.detach().cpu().numpy())
    bz, nc, h, w = fmap.shape
    
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    out = np.dot(fmap[i].detach().cpu().numpy().reshape((nc, h*w)).T, weight_softmax[idx, :].T)
    predict_heatmap = out.reshape(h, w)
    predict_heatmap = predict_heatmap - np.min(predict_heatmap)
    predict_img = predict_heatmap / np.max(predict_heatmap)
    predict_img = np.uint8(255 * predict_img)
    out = cv2.resize(predict_img, (im_size, im_size))
    heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    
    img_converted = im_convert(img[:, -1, :, :, :], video_file_name)
    
    # img_converted is 0-1 float, heatmap is 0-255 uint8
    result = heatmap * 0.5 + img_converted * 0.8 * 255
    
    heatmap_name = video_file_name + "_heatmap_" + str(i) + ".png"
    image_name = os.path.join(config.UPLOADED_IMAGES_DIR, heatmap_name)
    cv2.imwrite(image_name, result)
    return heatmap_name

import face_recognition
from PIL import Image as pImage
import time

def predict_image(model, image_path, original_filename):
    """
    Predicts whether a single image is REAL or FAKE using the sequence model.
    It extracts the face, applies transforms, and duplicates the frame.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise Exception("Could not read image")
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) == 0:
        return "NO_FACE", 0.0, None
        
    top, right, bottom, left = face_locations[0]
    padding = 40
    try:
        frame_face = frame[max(0, top - padding):min(frame.shape[0], bottom + padding), 
                           max(0, left - padding):min(frame.shape[1], right + padding)]
        rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
        img_face_rgb = pImage.fromarray(rgb_face, 'RGB')
    except Exception as e:
        frame_face = frame[top:bottom, left:right]
        rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
        img_face_rgb = pImage.fromarray(rgb_face, 'RGB')
        
    cropped_face_name = f"cropped_face_{int(time.time())}_{original_filename}"
    cropped_face_path = os.path.join(config.UPLOADED_IMAGES_DIR, cropped_face_name)
    img_face_rgb.save(cropped_face_path)
    
    transformed_frame = train_transforms(rgb_face)
    
    sequence_length = 10
    frames = [transformed_frame] * sequence_length
    frames_tensor = torch.stack(frames).unsqueeze(0)
    
    prediction_result = predict(model, frames_tensor, original_filename)
    
    return prediction_result[0], prediction_result[1], cropped_face_name
