# Para crear el requirements.txt ejecutamos 
# pipreqs --encoding=utf8 --force

# Primera Carga a Github
# git init
# git add .
# git remote add origin https://github.com/nicoig/ProductGPT.git
# git commit -m "Initial commit"
# git push -u origin master

# Actualizar Repo de Github
# git add .
# git commit -m "Se actualizan las variables de entorno"
# git push origin master

# En Render
# agregar en variables de entorno
# PYTHON_VERSION = 3.9.12

# git remote set-url origin https://github.com/nicoig/ProductGPT.git
# git remote -v
# git push -u origin main


################################################
##


import streamlit as st
import base64
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

# Cargar las variables de entorno para las claves API
load_dotenv(find_dotenv())

# Función para codificar imágenes en base64
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# Configura el título y subtítulo de la aplicación en Streamlit
st.title("EcoGPT")
st.markdown("""
    <style>
    .small-font {
        font-size:18px !important;
    }
    </style>
    <p class="small-font">Soy EcoGPT, carga la imagen del producto que deseas reciclar y te daré algunos consejos</p>
    """, unsafe_allow_html=True)

# Imagen de cabecera
st.image('img/robot.png', width=250)

# Carga de imagen por el usuario
uploaded_file = st.file_uploader("Carga una imagen del producto que deseas reciclar", type=["jpg", "png", "jpeg"])

# Restablecer el contenido generado al cargar una nueva imagen
if 'last_uploaded_file' not in st.session_state or (uploaded_file is not None and uploaded_file != st.session_state['last_uploaded_file']):
    st.session_state['last_uploaded_file'] = uploaded_file
    st.session_state['generated_content'] = False

# Botón de enviar y proceso principal
if st.button("Analizar Producto") and uploaded_file is not None:
    with st.spinner('Identificando el producto y material...'):
        image = encode_image(uploaded_file)
        st.session_state['generated_content'] = True

        # Identificar el producto y el material con la IA
        chain = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)
        msg = chain.invoke(
            [AIMessage(content="Basándose en la imagen, identifique el producto y el tipo de material."),
             HumanMessage(content=[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}])
            ]
        )

        st.session_state['identificacion'] = msg.content
        st.markdown("**Identificación del producto y material:**")
        st.write(st.session_state['identificacion'])

        # Generar recomendaciones de reciclaje
        prompt_reciclaje = PromptTemplate.from_template(
            """
            Dado el siguiente producto y material identificado:
            {identification}
            ¿Qué consejos de reciclaje se pueden dar para este producto?

            Output:
            """
        )
        runnable = prompt_reciclaje | chain | StrOutputParser()
        st.session_state['consejos_reciclaje'] = runnable.invoke({"identification": st.session_state['identificacion']})
        st.markdown("**Consejos para reciclar este producto:**")
        st.write(st.session_state['consejos_reciclaje'])
        
        # Generar información sobre el impacto ecológico
        prompt_impacto_ecologico = PromptTemplate.from_template(
            """
            Dado el siguiente producto y material identificado:
            {identification}
            Proporcione información sobre el tiempo de biodegradación del material y el impacto positivo de reciclar este material.

            Output:
            """
        )
        runnable = prompt_impacto_ecologico | chain | StrOutputParser()
        st.session_state['impacto_ecologico'] = runnable.invoke({"identification": st.session_state['identificacion']})
        st.markdown("**Impacto Ecológico del Material:**")
        st.write(st.session_state['impacto_ecologico'])

# Función para compilar la información en un string
def compile_information():
    info = ""
    if st.session_state.get('generated_content', False):
        info += "Identificación del producto y material:\n" + st.session_state.get('identificacion', '') + "\n\n"
        info += "Consejos para reciclar este producto:\n" + st.session_state.get('consejos_reciclaje', '') + "\n\n"
        info += "Impacto Ecológico del Material:\n" + st.session_state.get('impacto_ecologico', '') + "\n\n"
    return info

# Botón para descargar la información
if st.session_state.get('generated_content', False):
    info_to_download = compile_information()
    st.download_button(label="Descargar Información", data=info_to_download, file_name="ecoGPT_info.txt", mime="text/plain")
