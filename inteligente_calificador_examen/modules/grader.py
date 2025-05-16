# modules/grader.py
import requests
from modules.rag_engine import construir_rag_desde_docs, consultar_con_rag

# Función para construir el índice según el curso o carpeta específica
indices_cargados = {}

def evaluar_respuesta(texto_respuesta, modelo="phi3:latest", dificultad=5, curso="documentos"):
    """
    Evalúa una respuesta escrita por un estudiante usando un modelo LLM local (ej: phi3),
    apoyado en conocimiento recuperado con RAG correspondiente al curso.
    """
    if curso not in indices_cargados:
        indices_cargados[curso] = construir_rag_desde_docs(carpeta_docs=curso)
    indice = indices_cargados[curso]

    # Recuperar contexto relevante con RAG
    contexto = consultar_con_rag(indice, texto_respuesta)

    prompt = (
        f"Evalúa la siguiente respuesta de un estudiante sobre un tema académico. "
        f"Proporciona una puntuación del 1 al 10 basada en el criterio de dificultad {dificultad}. "
        f"También justifica tu evaluación de forma clara y breve, utilizando el contexto proporcionado.\n\n"
        f"Respuesta del estudiante:\n{texto_respuesta}\n\n"
        f"Contexto del curso extraído de los documentos:\n{contexto}"
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
    resultado = evaluar_respuesta(texto, dificultad=7, curso="documentos/filosofia")
    print("\nEvaluación generada por el modelo:\n")
    print(resultado)
