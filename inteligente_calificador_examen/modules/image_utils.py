# modules/image_utils.py
import cv2
import numpy as np
import easyocr

# Inicializa OCR en español e inglés
reader = easyocr.Reader(['es', 'en'], gpu=True)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen desde: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def extract_text(image_path):
    """
    Aplica OCR directamente a la imagen en escala de grises (sin efectos).
    """
    processed_img = preprocess_image(image_path)
    results = reader.readtext(processed_img, detail=0, paragraph=True)
    return "\n".join(results)
