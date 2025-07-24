import streamlit as st
import replicate
import os

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
        "Seitenverhältnis auswählen",
        ("4:3", "3:4", "16:9", "9:16", "1:1"),
        index=0, # Standardwert ist 4:3
    )
    submit_button = st.form_submit_button(label='✨ Bild erzeugen')

# Wenn der Button geklickt und eine Beschreibung eingegeben wurde
if submit_button and prompt:
    st.markdown("---")
    with st.spinner("KI-Magie wird gewirkt... das kann einen Moment dauern..."):
        try:
            # Das zu verwendende Modell: Google Imagen 4 Fast
            model_id = "google/imagen-4-fast"

            # Das Modell ausführen
            output = replicate.run(
                model_id,
                input={
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "output_format": "jpg",
                    "safety_filter_level": "block_only_high"
                }
            )

            # Das Bild anzeigen
            if output:
                # Für dieses Modell gibt replicate.run() die URL direkt als String zurück.
                image_url = output
                st.image(image_url, caption=f"Dein generiertes Bild für: '{prompt}'", use_column_width=True)
            else:
                st.error("Es tut mir leid, ich konnte kein Bild erzeugen. Bitte versuche es mit einer anderen Beschreibung.")

        except replicate.exceptions.ReplicateError as e:
            st.error(f"Fehler bei der Kommunikation mit Replicate: {e}")
        except Exception as e:
            st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
else:
    # Warnung, wenn der Button ohne Eingabe geklickt wird
    if submit_button and not prompt:
        st.warning("Bitte gib eine Beschreibung für das Bild ein, das du erstellen möchtest.")

st.markdown("---")
st.info("Diese App verwendet das `google/imagen-4-fast` Modell über die Replicate API.")
