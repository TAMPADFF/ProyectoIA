from modules.image_utils import extract_text


ruta_imagen = "../examenes/EXAMENES_MAY/EXAMEN1/2.jpg"  # Usa tu propia imagen
texto = extract_text(ruta_imagen)
print("Texto extraído:")
print(texto)
