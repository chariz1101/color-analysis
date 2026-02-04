import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

PALETTES = {
    "Deep Winter": ["#111111", "#191970", "#800020", "#FFFFFF"],
    "Cool Winter": ["#000080", "#C0C0C0", "#FF007F", "#000000"],
    "Clear Winter": ["#0000FF", "#00FF00", "#FF0000", "#FFFFFF"],
    "Deep Autumn": ["#4E3524", "#654321", "#800000", "#FFD700"],
    "Warm Autumn": ["#D2691E", "#8B4513", "#556B2F", "#DAA520"],
    "Soft Autumn": ["#BC8F8F", "#8FBC8F", "#BDB76B", "#CD853F"],
    "Warm Spring": ["#FF8C00", "#FFD700", "#32CD32", "#FF69B4"],
    "Light Spring": ["#FFFACD", "#FFB6C1", "#AFEEEE", "#98FB98"],
    "Clear Spring": ["#FFFF00", "#00FF7F", "#FF4500", "#EE82EE"],
    "Light Summer": ["#F0F8FF", "#FFB6C1", "#E0FFFF", "#F5F5DC"],
    "Cool Summer": ["#4682B4", "#D8BFD8", "#B0C4DE", "#708090"],
    "Soft Summer": ["#778899", "#BC8F8F", "#A9A9A9", "#B0E0E6"]
}

AVOID_PALETTES = {
    "Deep Winter": ["#F5F5DC", "#FFDAB9", "#E6E6FA", "#FFFACD"],
    "Cool Winter": ["#D2691E", "#DAA520", "#8B4513", "#556B2F"],
    "Clear Winter": ["#BC8F8F", "#A9A9A9", "#C0C0C0", "#F5F5DC"],
    "Deep Autumn": ["#E0FFFF", "#F0F8FF", "#FFB6C1", "#FFFFFF"],
    "Warm Autumn": ["#0000FF", "#FF00FF", "#C0C0C0", "#4682B4"],
    "Soft Autumn": ["#000000", "#FF0000", "#0000FF", "#FFFF00"],
    "Warm Spring": ["#4B0082", "#191970", "#2F4F4F", "#000000"],
    "Light Spring": ["#000000", "#800020", "#191970", "#3E4E23"],
    "Clear Spring": ["#BC8F8F", "#BDB76B", "#A9A9A9", "#F5F5DC"],
    "Light Summer": ["#8B4513", "#DAA520", "#000000", "#FF8C00"],
    "Cool Summer": ["#FF8C00", "#FFD700", "#D2691E", "#800000"],
    "Soft Summer": ["#FF0000", "#000000", "#00FF00", "#FFFFFF"]
}

def robust_white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0))
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def get_face_data(image):
    processed_img = robust_white_balance(image)
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    if not results.multi_face_landmarks:
        return None, None, None
    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape
    face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    points = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in face_oval]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points)], 255)
    skin_pixels = rgb_img[mask == 255]
    avg_rgb = np.mean(skin_pixels, axis=0).astype(int)
    return avg_rgb, processed_img, landmarks

def analyze_12_seasons(rgb_color):
    rgb_unit = np.uint8([[rgb_color]])
    lab_unit = cv2.cvtColor(rgb_unit, cv2.COLOR_RGB2LAB)[0][0]
    l, a, b = lab_unit 
    chroma = np.sqrt((a-128)**2 + (b-128)**2)
    is_warm = b > 128
    is_light = l > 150
    is_deep = l < 115
    is_clear = chroma > 25
    is_soft = chroma < 18
    if is_warm:
        if is_deep: season = "Deep Autumn"
        elif is_light: season = "Light Spring"
        elif is_clear: season = "Clear Spring"
        elif is_soft: season = "Soft Autumn"
        else: season = "Warm Autumn"
    else:
        if is_deep: season = "Deep Winter"
        elif is_light: season = "Light Summer"
        elif is_clear: season = "Clear Winter"
        elif is_soft: season = "Soft Summer"
        else: season = "Cool Winter"
    return season

def apply_drape(image, color_hex):
    color_hex = color_hex.lstrip('#')
    rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
    bgr = (rgb[2], rgb[1], rgb[0])
    h, w, _ = image.shape
    thickness = int(w * 0.08)
    draped_img = image.copy()
    cv2.rectangle(draped_img, (0, 0), (w, h), bgr, thickness)
    return draped_img