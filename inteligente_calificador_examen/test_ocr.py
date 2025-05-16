from modules.image_utils import extract_text


ruta_imagen = "../examenes/ExamenesCarlos/examen3/hoja1.jpeg" # Usa tu propia imagen
texto = extract_text(ruta_imagen)
print("Texto extraído:")
print(texto)
