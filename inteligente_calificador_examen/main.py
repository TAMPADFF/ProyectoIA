from modules.image_enhancer import mejorar_contraste_para_llava
from modules.image_processing import extract_text, extract_text_multi_ocr  # OCR tradicional mejorado
from modules.grader import evaluar_respuesta
import os


def procesar_examen_con_ocr(image_path, modelo_llm="phi3:latest", dificultad=5):
    """
    Flujo automático: mejora imagen, extrae texto con OCR (incluye variantes) y evalúa con un LLM.
    """
    print("🔧 Mejorando imagen...")
    imagen_mejorada = mejorar_contraste_para_llava(image_path)

    print("🧠 Extrayendo texto con OCR...")
    texto_extraido = extract_text_multi_ocr(imagen_mejorada)
    print("\n📝 Texto extraído:\n", texto_extraido)

    if not texto_extraido.strip():
        print("⚠️ No se pudo extraer texto legible de la imagen.")
        return

    print("📊 Evaluando respuesta...")
    resultado = evaluar_respuesta(texto_extraido, modelo=modelo_llm, nivel_dificultad=dificultad)
    print("\n✅ Resultado de la evaluación:\n")
    print(resultado)


# Ejemplo de uso
if __name__ == "__main__":
    ruta = "../examenes/ExamenesCarlos/examen3/hoja1.jpeg"
    procesar_examen_con_ocr(ruta, modelo_llm="phi3:latest", dificultad=6)
