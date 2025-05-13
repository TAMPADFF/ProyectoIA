import cv2
import numpy as np
import os

def mejorar_contraste_para_llava(image_path, output_path="temp_llava.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = cv2.equalizeHist(gray)

    height, width = contrast.shape
    if width < 800:
        contrast = cv2.resize(contrast, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(output_path, contrast)
    return output_path

def preprocesamiento_dinamico(image_path):
    """
    Ajusta el preprocesamiento dependiendo de las condiciones de la imagen.
    Retorna la mejorada para OCR.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    contraste = np.std(gray)

    if contraste < 30:
        # Imagen muy opaca o tenue: ecualización fuerte + adaptativa + resize
        enhanced = cv2.equalizeHist(gray)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 8)
        processed = cv2.resize(binary, (gray.shape[1] * 2, gray.shape[0] * 2))

    elif contraste < 60:
        # Contraste medio: realzar + suavizar ligeramente
        eq = cv2.equalizeHist(gray)
        blur = cv2.bilateralFilter(eq, 9, 75, 75)
        _, processed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    else:
        # Imagen ya con buen contraste: binarizar suave
        _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    temp_path = "temp_dinamica.jpg"
    cv2.imwrite(temp_path, processed)
    return temp_path
