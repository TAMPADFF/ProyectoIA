# main.py
from modules.image_enhancer import mejorar_contraste_para_llava
from modules.ocr_router import ocr_inteligente  # OCR inteligente combinado
from modules.grader import evaluar_respuesta
from modules.curso_selector import seleccionar_curso
import os


def procesar_examen_con_ocr(image_path, modelo_llm="phi3:latest", dificultad=5):
    """
    Flujo automático: mejora imagen, extrae texto con OCR inteligente y evalúa con un LLM.
    """
    print("🔧 Mejorando imagen...")
    imagen_mejorada = mejorar_contraste_para_llava(image_path)

    print("🧠 Extrayendo texto con OCR inteligente...")
    texto_extraido = ocr_inteligente(imagen_mejorada)
    print("\n📝 Texto extraído:\n", texto_extraido)

    if not texto_extraido.strip():
        print("⚠️ No se pudo extraer texto legible de la imagen.")
        return

    curso = seleccionar_curso()
    if not curso:
        print("❌ No se seleccionó un curso válido. Proceso cancelado.")
        return

    print("📊 Evaluando respuesta...")
    resultado = evaluar_respuesta(texto_extraido, modelo=modelo_llm, dificultad=dificultad, curso=curso)
    print("\n✅ Resultado de la evaluación:\n")
    print(resultado)


# Ejemplo de uso
if __name__ == "__main__":
    ruta = "../examenes/EXAMENES_MAY/EXAMEN1/2.jpg"
    procesar_examen_con_ocr(ruta, modelo_llm="phi3:latest", dificultad=6)
