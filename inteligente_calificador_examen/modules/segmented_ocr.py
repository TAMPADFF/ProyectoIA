import cv2
import numpy as np
import easyocr
import os

reader = easyocr.Reader(['es', 'en'], gpu=True)

def segment_and_ocr(image_path):
    """
    Segmenta la imagen en bloques de texto y aplica OCR a cada uno.
    Retorna el texto concatenado de todos los bloques legibles.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 15, 8)

    # Encontrar contornos de bloques de texto
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    resultados = []
    idx = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 30:  # Filtro para evitar ruido
            bloque = img[y:y+h, x:x+w]
            bloque = cv2.resize(bloque, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
            texto = reader.readtext(bloque, detail=0, paragraph=True)
            resultado = " ".join(texto)
            if resultado.strip():
                resultados.append(resultado)
                idx += 1

    return "\n".join(reversed(resultados))  # Revertir para lectura de arriba abajo
