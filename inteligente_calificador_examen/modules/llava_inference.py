import requests
import base64
import os

def encode_image_to_base64(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def transcribe_with_llava(image_path, model="llava:7b"):
    """
    Envía una imagen al modelo llava:7b y transcribe el texto manuscrito de la respuesta.
    """
    image_base64 = encode_image_to_base64(image_path)

    prompt = (
        "Transcribe con precisión el texto manuscrito que aparece en esta imagen de un examen. "
        "Devuelve solo lo que escribió el estudiante, sin explicaciones ni juicios."
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }
    )

    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error al comunicarse con Ollama: {response.status_code} - {response.text}"
