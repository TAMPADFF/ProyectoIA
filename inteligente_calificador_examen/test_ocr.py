from modules.image_utils import extract_text


ruta_imagen = "../examenes/examen2/Examen1.jpeg"  # Usa tu propia imagen
texto = extract_text(ruta_imagen)
print("Texto extraído:")
print(texto)
