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
from io import BytesIO

# Cargar las variables de entorno para las claves API
load_dotenv(find_dotenv())

# Función para codificar imágenes en base64
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# Configura el título y subtítulo de la aplicación en Streamlit
st.title("ProductGPT")

st.markdown("""
    <style>
    .small-font {
        font-size:18px !important;
    }
    </style>
    <p class="small-font">Te ayudaré con la descripción ideal para que promociones tu producto, sólo carga tu la imágen de tu producto, añade características adicionales como contexto y listo</p>
    """, unsafe_allow_html=True)
# Imagen
st.image('img/robot.png', width=250)

# Inicializar la variable descripcion_producto
descripcion_producto = ""


# Carga de imagen y texto por el usuario
uploaded_file = st.file_uploader("Carga una imagen de tu producto", type=["jpg", "png", "jpeg"])
input_text = st.text_input("Añade características adicionales como el Precio, Marca, etc")

# Botón de enviar y proceso principal
if st.button("Enviar Consulta") and uploaded_file is not None and input_text:
    with st.spinner('Analizando tu consulta...'):
        image = encode_image(uploaded_file)

        # Analizar la imagen y el texto con la IA
        chain = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)
        msg = chain.invoke(
            [AIMessage(content="Por favor, realice un análisis detallado de la imagen proporcionada junto con la descripción del texto. Identifique elementos clave tales como el producto, la marca, el color, y otros detalles relevantes que puedan ser discernidos. Proporcione una descripción pormenorizada de todos los aspectos de la imagen que pueda evaluar, incluyendo posibles usos o el público objetivo del producto si esto pudiera ser inferido."),
             HumanMessage(content=[{"type": "text", "text": input_text},
                                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}])
            ]
        )

        producto = msg.content
        #st.markdown("**Información general del producto:**")
        #st.write(diagnostico)

        # Generar recomendaciones de tratamiento
        chain = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024)
        prompt_producto = PromptTemplate.from_template(
            """
            Dada la siguiente descripción de producto: "{product_vision}", se solicita crear un copy promocional atractivo y convincente. El copy debe ser breve, ingenioso y diseñado para captar la atención y despertar interés en redes sociales, debe ser completo e incluir las características más relevantes en párrafos.

            Por favor, sigue estas directrices:
            - Usa un lenguaje claro, directo y amigable.
            - Incluye al menos dos llamados a la acción.
            - Emplea emojis de manera efectiva para realzar el mensaje.
            - Asegúrate de que el tono sea optimista y empoderador.
            Output:
            """
        )
        runnable = prompt_producto | chain | StrOutputParser()
        descripcion_producto = runnable.invoke({"product_vision": producto})
        st.markdown("**Descripción promocional para tu producto:**")
        st.write(descripcion_producto)
        
        
        prompt_nombre_producto = PromptTemplate.from_template(
            """
            Dada la siguiente descripción de producto: "{product_vision}", se solicita establecer un nombre corto al producto, que describa en una o dos palabras,
            por ejemplo: botella_agua_negra
            Output:
            """
        )
        runnable = prompt_nombre_producto | chain | StrOutputParser()
        nombre_producto = runnable.invoke({"product_vision": producto})
        #st.markdown("**Descripción promocional para tu producto:**")
        #st.write(nombre_producto)
        


# Descarga de información al final del código
if descripcion_producto and nombre_producto:
    nombre_producto_str = nombre_producto.strip().replace(" ", "_")  # Asegura que el nombre no tenga espacios
    # Convertir la descripción a un archivo descargable
    output_data = descripcion_producto.encode("utf-8")
    b64 = base64.b64encode(output_data).decode()
    button_label = 'Descargar Descripción'

    # Generar el botón de descarga con el nombre del producto en el archivo
    st.download_button(
        label=button_label,
        data=output_data,
        file_name=f'descripcion_{nombre_producto_str}.txt',
        mime='text/plain'
    )  


# Nota: Asegúrate de tener configuradas tus claves API y cualquier otro ajuste específico necesario para tu entorno y APIs.
