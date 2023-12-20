import PyPDF2
import gdown
import os
import shutil
import chromadb
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
from IPython.display import Markdown, display

# Importación de bibliotecas relacionadas con OpenAI
import getpass
os.environ["OPENAI_API_KEY"] = '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'
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

# Importación de bibliotecas relacionadas con embeddings y NLP
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from decouple import config
from llama_index import ServiceContext, VectorStoreIndex, download_loader, SimpleDirectoryReader
from jinja2 import Template
import requests
from decouple import config
import nltk
import pandas as pd
import ssl

# Manejo especial para contextos SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Descarga de datos de lenguaje natural (NLP)
nltk.download('punkt')


# Función para generar una plantilla de instrucciones para Zephyr
def zephyr_instruct_template(messages, add_generation_prompt=True):
    # Definir la plantilla Jinja
    template_str = "{% for message in messages %}"
    template_str += "{% if message['role'] == 'user' %}"
    template_str += "{{ message['content'] }}</s>\n"
    template_str += "{% elif message['role'] == 'assistant' %}"
    template_str += "{{ message['content'] }}</s>\n"
    template_str += "{% elif message['role'] == 'system' %}"
    template_str += "{{ message['content'] }}</s>\n"
    template_str += "{% else %}"
    template_str += "{{ message['content'] }}</s>\n"
    template_str += "{% endif %}"
    template_str += "{% endfor %}"
    template_str += "{% if add_generation_prompt %}"
    template_str += "\n"
    template_str += "{% endif %}"
    # Crear un objeto de plantilla con la cadena de plantilla
    template = Template(template_str)
    # Renderizar la plantilla con los mensajes proporcionados
    return template.render(messages=messages, add_generation_prompt=add_generation_prompt)

# Función para generar una respuesta utilizando la API de Hugging Face
def generate_answer(prompt: str, max_new_tokens: int = 768) -> str:
    try:
        # Tu clave API de Hugging Face
        api_key = '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'

        # URL de la API de Hugging Face para la generación de texto
        api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        # Cabeceras para la solicitud
        headers = {"Authorization": f"Bearer {api_key}"}
        # Datos para enviar en la solicitud POST
        # Sobre los parámetros: https://huggingface.co/docs/transformers/main_classes/text_generation
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95
            }
        }
        # Realizamos la solicitud POST
        response = requests.post(api_url, headers=headers, json=data)
        # Extraer respuesta
        respuesta = response.json()[0]["generated_text"][len(prompt):]
        return respuesta
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

# Función para preparar el prompt de la pregunta con información de contexto
def prepare_prompt(query_str: str, nodes: list):
    TEXT_QA_PROMPT_TMPL = (
        "La información de contexto es la siguiente:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Dada la información de contexto anterior, y sin utilizar conocimiento previo, responde la siguiente pregunta.\n"
        "Pregunta: {query_str}\n"
        "Respuesta: "
    )

    # Construimos el contexto de la pregunta
    context_str = ''
    for node in nodes:
        # Usamos get para obtener la clave, y si no está presente, proporcionamos un valor predeterminado
        page_label = node.metadata.get("page_label", "No Page Label")
        file_path = node.metadata.get("file_path", "No File Path")
        context_str += f"\npage_label: {page_label}\n"
        context_str += f"file_path: {file_path}\n\n"
        context_str += f"{node.text}\n"

    messages = [
        {
            "role": "system",
            "content": "Eres un asistente útil que siempre responde con respuestas veraces, útiles y basadas en hechos.",
        },
        {"role": "user", "content": TEXT_QA_PROMPT_TMPL.format(context_str=context_str, query_str=query_str)},
    ]
    final_prompt = zephyr_instruct_template(messages)
    return final_prompt

# Cargamos nuestro modelo de embeddings
print('Cargando modelo de embeddings...')
embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


# Construimos un índice de documentos a partir del archivo CSV con delimitador "|"
print('Indexando documentos...')

# Crear cliente y una nueva colección
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("seguros_collections5")

# Definir función de embedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Cargar documentos
documents = SimpleDirectoryReader("data").load_data()

# Configurar ChromaVectorStore y cargar datos
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)

# Crear un objeto 'retriever' a partir del índice para realizar recuperaciones de documentos
# con un parámetro de similitud superior (similarity_top_k) establecido en 2.
retriever = index.as_retriever(similarity_top_k=2)

def respuesta(pregunta):
    nodes = retriever.retrieve(pregunta)
    final_prompt = prepare_prompt(pregunta, nodes)
    respuesta = generate_answer(final_prompt)
    return respuesta