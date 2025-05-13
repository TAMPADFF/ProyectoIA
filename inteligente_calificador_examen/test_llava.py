import requests
import base64
from modules.image_enhancer import mejorar_contraste_para_llava

# Ruta original de la imagen del examen manuscrito
imagen_original = "../examenes/EXAMENES_MAY/EXAMEN1/2.jpg"

# 🔧 Paso 1: Mejorar la imagen
imagen_mejorada = mejorar_contraste_para_llava(imagen_original)

# 🔄 Paso 2: Codificar la imagen mejorada a base64
with open(imagen_mejorada, "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

# 🧠 Paso 3: Prompt mejorado para forzar la transcripción
prompt = (
    "Transcribe exactamente todo el texto manuscrito que aparece en esta imagen. "
    "Aunque esté borroso o difícil de leer, haz tu mejor esfuerzo por descifrar cada palabra escrita por el estudiante. "
    "No describas la imagen ni digas si puedes o no puedes ayudar. Solo transcribe lo que esté escrito."
)

# 🚀 Paso 4: Llamada a Ollama usando llava:7b
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llava:7b",
        "prompt": prompt,
        "images": [img_base64],
        "stream": False
    }
)

# 🖨️ Paso 5: Mostrar respuesta del modelo
print("Texto transcrito con LLaVA:\n")
if response.status_code == 200:
    print(response.json()["response"])
else:
    print(f"❌ Error al comunicarse con Ollama: {response.status_code} - {response.text}")
