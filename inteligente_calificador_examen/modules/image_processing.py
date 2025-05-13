import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['es', 'en'], gpu=True)

def extract_text(image_path):
    """Versión simple (por compatibilidad)"""
    return extract_text_multi_ocr(image_path)

def extract_text_multi_ocr(image_path):
    """
    Intenta múltiples preprocesamientos sobre una imagen para mejorar la lectura OCR.
    Devuelve el texto más largo encontrado (más palabras).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    preprocesamientos = []

    # Variante 1: escala de grises + binarización Otsu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocesamientos.append(bin_otsu)

    # Variante 2: filtro bilateral + ecualización
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    equalized = cv2.equalizeHist(smooth)
    preprocesamientos.append(equalized)

    # Variante 3: inversión + binarización adaptativa
    inv = cv2.bitwise_not(gray)
    adap_bin = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 10)
    preprocesamientos.append(adap_bin)

    # Variante 4: resize + histograma
    resized = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2), interpolation=cv2.INTER_LINEAR)
    eq_resized = cv2.equalizeHist(resized)
    preprocesamientos.append(eq_resized)

    # Aplicar OCR a cada variante
    resultados = []
    for i, variante in enumerate(preprocesamientos):
        texto = reader.readtext(variante, detail=0, paragraph=True)
        resultado = "\n".join(texto)
        resultados.append((len(resultado.split()), resultado))

    # Seleccionar el resultado más largo (más palabras reconocidas)
    mejor_resultado = max(resultados, key=lambda x: x[0])[1]
    return mejor_resultado
