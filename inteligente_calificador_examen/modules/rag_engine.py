# modules/rag_engine.py
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Inicializar el LLM (phi3 local desde Ollama)
llm = Ollama(model="phi3")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

def construir_rag_desde_docs(carpeta_docs="documentos"):
    """
    Construye un índice RAG desde documentos en una carpeta.
    """
    documentos = SimpleDirectoryReader(carpeta_docs).load_data()
    indice = VectorStoreIndex.from_documents(documentos, service_context=service_context)
    return indice

def consultar_con_rag(indice, pregunta):
    """
    Consulta el índice con una pregunta textual y devuelve la respuesta generada.
    """
    query_engine = indice.as_query_engine(similarity_top_k=5)
    respuesta = query_engine.query(pregunta)
    return str(respuesta)
