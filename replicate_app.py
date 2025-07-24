import streamlit as st
import replicate
import os
import json

from dotenv import load_dotenv

# --- Seitenkonfiguration ---
st.set_page_config(
    page_title="AI Bild-Generator",
    page_icon="🎨",
    layout="centered",
)

# --- Benutzerdefiniertes CSS für die Google-ähnliche Suchleiste ---
st.markdown("""
<style>
/* Hauptcontainer für die Texteingabe */
div[data-testid="stTextInput"] {
    position: relative;
    margin-bottom: 1rem; /* Etwas Platz nach unten */
}

/* Das eigentliche Eingabefeld */
div[data-testid="stTextInput"] input {
    border-radius: 24px;
    border: 1px solid #dfe1e5;
    padding: 12px 20px; /* Oben/Unten, Rechts/Links */
    width: 100%;
    transition: box-shadow 0.3s, border-color 0.3s;
    font-size: 16px;
}

/* Stil für das Eingabefeld bei Fokus (wenn man hineinklickt) */
div[data-testid="stTextInput"] input:focus {
    box-shadow: 0 1px 6px rgb(32 33 36 / 28%);
    border-color: rgba(223,225,229,0);
    outline: none;
}

/* Stil für den Platzhaltertext */
div[data-testid="stTextInput"] input::placeholder {
    color: #80868b;
    opacity: 1; /* Für Firefox */
}

/* Versteckt das Standard-Streamlit-Label */
div[data-testid="stTextInput"] label {
    display: none;
}
</style>
""", unsafe_allow_html=True)


# Lade Umgebungsvariablen aus .env-Datei (für lokale Entwicklung)
load_dotenv()

# --- App-Logik ---

# Titel und Einleitung
st.title("🎨 AI Bild-Generator")
st.markdown("Erwecke deine Ideen zum Leben! Gib einfach eine Beschreibung ein und die KI zeichnet es für dich.")

# --- Modell-Konfigurationen ---
# Wir definieren das komplexe ComfyUI-Workflow-JSON hier als Python-Dictionary.
# Das macht es einfacher, den Benutzer-Prompt dynamisch einzufügen.
COMFY_WORKFLOW_TEMPLATE = {
  "3": {
    "inputs": {
      "seed": 156680208700286,
      "steps": 10,
      "cfg": 2.5,
      "sampler_name": "dpmpp_2m_sde",
      "scheduler": "karras",
      "denoise": 1,
      "model": ["4", 0],
      "positive": ["6", 0],
      "negative": ["7", 0],
      "latent_image": ["5", 0]
    },
    "class_type": "KSampler"
  },
  "4": {
    "inputs": { "ckpt_name": "SDXL-Flash.safetensors" },
    "class_type": "CheckpointLoaderSimple"
  },
  "5": {
    "inputs": { "width": 1024, "height": 1024, "batch_size": 1 },
    "class_type": "EmptyLatentImage"
  },
  "6": {
    "inputs": {
      "text": "placeholder for the prompt", # Dieser Text wird ersetzt
      "clip": ["4", 1]
    },
    "class_type": "CLIPTextEncode"
  },
  "7": {
    "inputs": { "text": "text, watermark", "clip": ["4", 1] },
    "class_type": "CLIPTextEncode"
  },
  "8": {
    "inputs": { "samples": ["3", 0], "vae": ["4", 2] },
    "class_type": "VAEDecode"
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": ["8", 0]
    },
    "class_type": "SaveImage"
  }
}


# Replicate API Token prüfen
# Für Streamlit Cloud Deployment wird der Token aus den Secrets gelesen
if 'REPLICATE_API_TOKEN' in st.secrets:
    replicate_api_token = st.secrets['REPLICATE_API_TOKEN']
    os.environ['REPLICATE_API_TOKEN'] = replicate_api_token
else:
    # Für die lokale Entwicklung wird die Umgebungsvariable geprüft
    if not os.getenv('REPLICATE_API_TOKEN'):
        st.error("Replicate API Token nicht gefunden.")
        st.markdown("Bitte setze die Umgebungsvariable `REPLICATE_API_TOKEN`. Deinen Token erhältst du hier: replicate.com/account/api-tokens")
        st.stop()

# Das Formular für die Eingabe
with st.form(key='image_form'):
    prompt = st.text_input(
        "Bildbeschreibung eingeben",
        placeholder="z.B. Ein Foto eines Astronauten, der auf einem Pferd reitet",
        key="prompt_input"
    )
    aspect_ratio = st.selectbox(
        "Seitenverhältnis (nur für Google Imagen)",
        ("4:3", "3:4", "16:9", "9:16", "1:1"),
        index=0, # Standardwert ist 4:3
    )
    submit_button = st.form_submit_button(label='✨ Bilder erzeugen')

# Wenn der Button geklickt und eine Beschreibung eingegeben wurde
if submit_button and prompt:
    st.markdown("---")
    with st.spinner("KI-Magie wird gewirkt... das kann einen Moment dauern..."):
        try:
            # --- Modell 1: Google Imagen ---
            st.write("1. Erzeuge Bild mit Google Imagen...")
            imagen_input = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": "jpg",
                "safety_filter_level": "block_only_high"
            }
            imagen_output = replicate.run("google/imagen-4-fast", input=imagen_input)

            # --- Modell 2: ComfyUI Workflow ---
            st.write("2. Erzeuge Bild mit ComfyUI Workflow...")
            # Erstelle eine Kopie des Templates und füge den User-Prompt ein
            comfy_workflow = COMFY_WORKFLOW_TEMPLATE.copy()
            comfy_workflow['6']['inputs']['text'] = prompt

            comfy_ui_output = replicate.run(
                "fofr/any-comfyui-workflow:f552cf6bb263b2c7c547c3c7fb158aa4309794934bedc16c9aa395bee407744d",
                input={
                    "workflow_json": json.dumps(comfy_workflow),
                    "output_format": "webp",
                    "randomise_seeds": True,
                }
            )

            # --- Ergebnisse anzeigen ---
            st.markdown("---")
            st.subheader("Ihre generierten Bilder")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Google Imagen")
                if imagen_output:
                    # Die robuste str()-Konvertierung funktioniert auch hier.
                    image_url_1 = str(imagen_output)
                    st.image(image_url_1, use_container_width=True)
                else:
                    st.warning("Kein Ergebnis von Google Imagen erhalten.")
            
            with col2:
                st.markdown("#### ComfyUI Workflow")
                # ComfyUI gibt eine Liste zurück, wir nehmen das erste Element.
                if comfy_ui_output and isinstance(comfy_ui_output, list) and comfy_ui_output[0]:
                    result_item = comfy_ui_output[0]
                    # Die robuste str()-Konvertierung, die das Problem gelöst hat.
                    image_url_2 = str(result_item)
                    st.image(image_url_2, use_container_width=True)
                else:
                    st.warning("Kein Ergebnis vom ComfyUI Workflow erhalten.")

        except replicate.exceptions.ReplicateError as e:
            st.error(f"Fehler bei der Kommunikation mit Replicate: {e}")
        except Exception as e:
            st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
else:
    # Warnung, wenn der Button ohne Eingabe geklickt wird
    if submit_button and not prompt:
        st.warning("Bitte gib eine Beschreibung für das Bild ein, das du erstellen möchtest.")

st.markdown("---")
st.info("Diese App verwendet die Modelle `google/imagen-4-fast` und `fofr/any-comfyui-workflow` über die Replicate API.")
