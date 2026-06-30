import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient


#  API KULCSOK BETÖLTÉSE DOTENV-VEL

load_dotenv()  # Ez olvassa be
GROQ_KEY = os.getenv("GROQ_API_KEY")
HF_KEY = os.getenv("HF_API_KEY")


# FRONTEND BEÁLLÍTÁSOK ÉS PARAMÉTEREZÉS

st.set_page_config(page_title="Esti Mese Generátor", page_icon="🌙", layout="wide")
st.title("Esti Mese Generátor")

# Oldalsáv: Paraméterek
st.sidebar.header("MI Paraméterek")
temperature = st.sidebar.slider("Kreativitás (Temperature)", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
max_tokens = st.sidebar.slider("Maximális hossz (Tokens)", min_value=500, max_value=2000, value=1000, step=100)

col1, col2 = st.columns(2)
with col1:
    child_name = st.text_input("Gyermek neve:", placeholder="pl. Peti")
    child_age = st.number_input("Életkor (év):", min_value=1, max_value=12, value=5)
with col2:
    theme = st.text_input("Mese témája:", placeholder="pl. rémálom egy küklopszról")
    moral = st.text_input("Tanulság:", placeholder="pl. a barátság fontossága")



# LANGCHAIN + GROQ SZÖVEGGENERÁLÁS

def generate_story_with_groq(api_key, name, age, theme, moral, temp, tokens):
    # LangChain ChatGroq inicializálása
    llm = ChatGroq(
        temperature=temp,
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        max_tokens=tokens
    )

    system_prompt = "Te egy díjnyertes magyar gyermekkönyv-író vagy."
    human_prompt = """
A feladatod egy esti mese írása a következő paraméterek alapján:
- Név: {name}
- Életkor: {age}
- Téma: {theme}
- Tanulság: {moral}

Gondolkodj lépésről lépésre:
1. Tervezd meg a cselekményt.
2. Írd meg a mesét magyarul.
3. Írj egy angol promptot a képgenerátornak, ami illusztrálja a mesét.

KÖTELEZŐ formátum, amit szigorúan be kell tartanod:
[TERV]
(Ide jön a cselekmény)
[MESE]
(Ide jön a mese szövege)
[KÉP_PROMPT]
(Ide jön az angol nyelvű képleírás)
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "name": name,
        "age": age,
        "theme": theme,
        "moral": moral
    })

    return response.content



# GOMB ÉS HIBAKEZELÉS

if st.button("Mese és kép generálása", use_container_width=True):
    # Ellenőrizzük, hogy sikerült-e beolvasni a kulcsokat a .env-ből
    if not GROQ_KEY or not HF_KEY:
        st.error("Hiba: Hiányzó API kulcsok! Ellenőrizd a .env fájlt.")
    elif not child_name or not theme:
        st.warning("Kérlek, töltsd ki a nevet és a témát!")
    else:
        try:
            # SZÖVEG GENERÁLÁSA (GROQ + LANGCHAIN)
            with st.spinner("Mese megírása a Groq-val"):
                raw_output = generate_story_with_groq(
                    GROQ_KEY, child_name, child_age, theme, moral, temperature, max_tokens
                )

                try:
                    plan_part = raw_output.split("[TERV]")[1].split("[MESE]")[0].strip()
                    story_part = raw_output.split("[MESE]")[1].split("[KÉP_PROMPT]")[0].strip()
                    image_prompt_part = raw_output.split("[KÉP_PROMPT]")[1].strip()
                except IndexError:
                    st.error(f"Az MI nem tartotta be a formátumot. Nyers válasz:\n\n{raw_output}")
                    st.stop()

            # KÉP GENERÁLÁSA (HUGGING FACE)
            with st.spinner("🎨 Illusztráció megrajzolása a Stable Diffusion segítségével..."):
                hf_client = InferenceClient(api_key=HF_KEY)
                image = hf_client.text_to_image(
                    image_prompt_part + ", cute children's book illustration style, colorful, high quality",
                    model="stabilityai/stable-diffusion-xl-base-1.0"
                )

            st.success("Elkészült a mese!")
            st.image(image, caption="Stable Diffusion által generált illusztráció")
            st.subheader(f"Mese {child_name} számára")
            st.write(story_part)

            with st.expander("Így gondolkodott az MI (Terv & Prompt)"):
                st.write("**Terv:**", plan_part)
                st.write("**Kép Prompt:**", image_prompt_part)

        except Exception as e:
            st.error(f"Hiba történt a generálás során: {str(e)}")