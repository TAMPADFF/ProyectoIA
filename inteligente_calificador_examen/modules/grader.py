import requests


def evaluar_respuesta(texto_respuesta, modelo="phi3:latest", nivel_dificultad=5):
    """
    Evalúa una respuesta escrita por un estudiante usando un modelo LLM local (ej: phi3).
    """
    prompt = (
        f"Evalúa la siguiente respuesta de un estudiante sobre un tema académico. "
        f"Proporciona una puntuación del 1 al 10 basada en el criterio de dificultad {nivel_dificultad}. "
        f"También justifica tu evaluación de forma clara y breve.\n\n"
        f"Respuesta del estudiante:\n{texto_respuesta}"
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": modelo,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"❌ Error al comunicarse con el modelo: {response.status_code} - {response.text}"


# Ejemplo de uso individual:
if __name__ == "__main__":
    texto = (
        "Kant decía que el conocimiento empieza con la experiencia, "
        "pero también necesitamos conceptos racionales para organizar esa experiencia."
    )
    resultado = evaluar_respuesta(texto, nivel_dificultad=7)
    print("\nEvaluación generada por el modelo:\n")
    print(resultado)
