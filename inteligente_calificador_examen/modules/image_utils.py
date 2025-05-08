import cv2
import numpy as np
import easyocr

# Inicializa OCR en español e inglés
reader = easyocr.Reader(['es', 'en'], gpu=True)  # Usa GPU si tienes la RTX 3050 activa

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen desde: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarización adaptativa
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )

    # Eliminación de ruido
    kernel = np.ones((1, 1), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return clean

def extract_text(image_path):
    """
    Aplica OCR a la imagen procesada y retorna el texto detectado.
    """
    processed_img = preprocess_image(image_path)
    results = reader.readtext(processed_img, detail=0, paragraph=True)
    return "\n".join(results)
