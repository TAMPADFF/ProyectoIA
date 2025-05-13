from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Carga el modelo preentrenado de Microsoft
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Usa GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_text_trocr(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

# Prueba con una imagen
if __name__ == "__main__":
    path = "../examenes/EXAMENES_MAY/EXAMEN1/2.jpg"  # Cambia esto
    texto = extract_text_trocr(path)
    print("\nTexto detectado:\n")
    print(texto)
