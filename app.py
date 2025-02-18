import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from langchain_groq import ChatGroq  
@st.cache_resource
def initialize_llm():
    try:
        return ChatGroq(
            temperature=0.7,
            groq_api_key="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",  
            model_name="llama3-8b-8192"
        )
    except Exception as e:
        st.error(f" Erreur lors de l'initialisation de Groq LLM: {e}")
        return None

llm = initialize_llm()

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("my_model1.h5")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

model = load_model()

try:
    with open("class_indices.json", "r", encoding="utf-8") as file:
        class_indices = json.load(file)
except Exception as e:
    st.error(f"Erreur lors du chargement des classes: {e}")
    class_indices = {}

def load_and_preprocess_image(image):
    try:
        img = image.resize((244, 244))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f" Erreur lors du traitement de l'image: {e}")
        return None

def predict_disease(image, model):
    if model is None:
        st.error("Modèle CNN non chargé. Vérifiez votre fichier .h5")
        return None
    preprocessed_img = load_and_preprocess_image(image)
    if preprocessed_img is None:
        return None
    try:
        prediction = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_indices[str(predicted_class_index)]
        return predicted_class_name
    except Exception as e:
        st.error(f" Erreur lors de la prédiction: {e}")
        return None

def generate_treatment_with_llm(image, model, class_indices, llm):
    try:
        preprocessed_img = load_and_preprocess_image(image)
        if preprocessed_img is None:
            return None, "Erreur lors du traitement de l'image."

        st.write(f"Image chargée avec succès ! Forme : {preprocessed_img.shape}")

        prediction = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_indices[str(predicted_class_index)]

        prompt = f"Quels sont les traitements pour la maladie {predicted_class_name} sur cette plante ?"


        treatment = llm.invoke(prompt)

        if hasattr(treatment, "content"):
            treatment_text = treatment.content
        else:
            treatment_text = "Réponse invalide reçue de Groq LLM."


        # ✅ Affichage en Markdown (supporte les retours à la ligne)
        st.markdown(f"### 🩺 **Traitement Recommandé :**\n{treatment_text}")

        return predicted_class_name, treatment_text

    except Exception as e:
        st.error(f"❌ Erreur lors de la génération du traitement : {e}")
        return None, f"⚠ Erreur lors de la génération du traitement : {e}"


    


st.markdown(
    "<h1 style='text-align: center; color:green'>🌱 Save Plant 🌱</h1>",
    unsafe_allow_html=True
)
import requests
from io import BytesIO

st.write("📷 **Choisissez une option pour charger une image :**")
option = st.radio("🌐 **Image en ligne ou fichier local ?**", ("📡 URL d'image", "📂 Fichier local"))

image = None  

if option == "📡 URL d'image":
    image_url = st.text_input("🌍 **Entrez l'URL de l'image :**", "")
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="🌱 Image chargée depuis Internet", use_column_width=True)
        except Exception as e:
            st.error(f"⚠ Erreur lors du chargement de l'image en ligne : {e}")

elif option == "📂 Fichier local":
    uploaded_image = st.file_uploader("📤 **Téléchargez une image (JPG, PNG)**", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="🌱 Image téléchargée", use_column_width=True)

if image:
    predicted_class_name = predict_disease(image, model)

    if predicted_class_name:
        st.write(f"🟢 **Plante détectée:** {predicted_class_name.split('___')[0]}")
        st.write(f"🦠 **Maladie détectée:** {predicted_class_name.split('___')[1]}")

        # 🩺 Traitement basé sur LLM
        if st.button("🔬 Obtenir un traitement"):
            disease, treatment = generate_treatment_with_llm(image, model, class_indices, llm)
    else:
        st.error("Impossible d'analyser l'image. Essayez une autre image.")
