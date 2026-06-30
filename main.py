import cv2
import numpy as np
import math
import time
import os
import sys
import urllib.request
import ctypes
import mediapipe as mp

try:
    try:
        msvcrt = ctypes.CDLL('msvcrt')
        dummy_free = msvcrt.free
    except Exception:
        dummy_free = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda p: None)

    orig_cdll_getattr = ctypes.CDLL.__getattr__
    def patched_cdll_getattr(self, name):
        if name == 'free':
            return dummy_free
        return orig_cdll_getattr(self, name)
    
    ctypes.CDLL.__getattr__ = patched_cdll_getattr
except Exception:
    pass

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
SEG_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"

HAND_MODEL_PATH = "hand_landmarker.task"
SEG_MODEL_PATH = "selfie_segmenter.tflite"

def download_file(path, url):
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            pass

download_file(HAND_MODEL_PATH, HAND_MODEL_URL)
download_file(SEG_MODEL_PATH, SEG_MODEL_URL)

base_hand = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
options_hand = vision.HandLandmarkerOptions(base_options=base_hand, num_hands=2, min_hand_detection_confidence=0.7)
landmarker = vision.HandLandmarker.create_from_options(options_hand)

base_seg = mp_python.BaseOptions(model_asset_path=SEG_MODEL_PATH)
options_seg = vision.ImageSegmenterOptions(base_options=base_seg, output_category_mask=False, output_confidence_masks=True)
segmenter = vision.ImageSegmenter.create_from_options(options_seg)

def calculate_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def is_peace_sign(landmarks):
    """
    Mendeteksi gestur jari 'peace' / V-sign dengan AI Landmarks presisi tinggi.
    """
    wrist = landmarks[0]
    
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    
    index_extended = index_tip.y < index_pip.y and calculate_distance(wrist, index_tip) > calculate_distance(wrist, index_mcp) * 1.2
    middle_extended = middle_tip.y < middle_pip.y and calculate_distance(wrist, middle_tip) > calculate_distance(wrist, middle_mcp) * 1.2
    
    ring_folded = ring_tip.y > ring_pip.y or calculate_distance(wrist, ring_tip) < calculate_distance(wrist, index_mcp) * 1.1
    pinky_folded = pinky_tip.y > pinky_pip.y or calculate_distance(wrist, pinky_tip) < calculate_distance(wrist, index_mcp) * 1.1
    
    return index_extended and middle_extended and ring_folded and pinky_folded

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    window_name = "Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    current_blur_factor = 0.0
    fade_speed = 0.08

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        hand_result = landmarker.detect(mp_image)
        seg_result = segmenter.segment(mp_image)

        peace_detected = False
        if hand_result.hand_landmarks:
            for landmarks in hand_result.hand_landmarks:
                if is_peace_sign(landmarks):
                    peace_detected = True
                    break

        target_blur_factor = 1.0 if peace_detected else 0.0
        if current_blur_factor < target_blur_factor:
            current_blur_factor = min(target_blur_factor, current_blur_factor + fade_speed)
        elif current_blur_factor > target_blur_factor:
            current_blur_factor = max(target_blur_factor, current_blur_factor - fade_speed)

        if current_blur_factor > 0.001:
            if seg_result.confidence_masks:
                seg_mask = seg_result.confidence_masks[0].numpy_view()
                seg_mask_blurred = cv2.GaussianBlur(seg_mask, (21, 21), 0)
                seg_mask_3d = np.stack((seg_mask_blurred,) * 3, axis=-1)

                blurred_full_frame = cv2.GaussianBlur(frame, (99, 99), 0)

                effective_mask = seg_mask_3d * current_blur_factor
                frame = (frame * (1.0 - effective_mask) + blurred_full_frame * effective_mask).astype(np.uint8)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
