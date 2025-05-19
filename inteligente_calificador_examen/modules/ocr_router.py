# modules/ocr_router.py
from modules.image_utils import extract_text as ocr_simple
from modules.image_processing import extract_text_multi_ocr as ocr_multi
from modules.segmented_ocr import segment_and_ocr as ocr_segmentado
import os


def ocr_inteligente(image_path):
    """
    Prueba diferentes métodos de OCR y retorna el resultado más completo.
    """
    resultados = []

    try:
        texto1 = ocr_simple(image_path)
        resultados.append((len(texto1.split()), texto1))
    except Exception:
        pass

    try:
        texto2 = ocr_multi(image_path)
        resultados.append((len(texto2.split()), texto2))
    except Exception:
        pass

    try:
        texto3 = ocr_segmentado(image_path)
        resultados.append((len(texto3.split()), texto3))
    except Exception:
        pass

    if resultados:
        mejor = max(resultados, key=lambda x: x[0])  # Más palabras detectadas
        return mejor[1]
    else:
        return ""
