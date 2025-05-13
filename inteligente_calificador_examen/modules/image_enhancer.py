import cv2

def mejorar_contraste_para_llava(image_path, output_path="temp_llava.jpg"):
    """
    Mejora la imagen para aumentar la legibilidad del texto manuscrito.
    Aplica escala de grises, ecualización de histograma y redimensionamiento si es necesario.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = cv2.equalizeHist(gray)

    height, width = contrast.shape
    if width < 800:
        contrast = cv2.resize(contrast, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(output_path, contrast)
    return output_path
