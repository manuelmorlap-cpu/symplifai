# In-memory history of summaries (persists until server restart)
summaries_history = []
import os
import tempfile
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

# Cuando se usa Flask con templates en la misma carpeta, es necesario este cambio.
# Si el index.html está en la carpeta "templates", volver a usar la #1 de abajo
#1 app = Flask(__name__)
app = Flask(__name__, template_folder='.')

# Cargar variables de entorno
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY or not API_KEY.startswith("sk-"):
    raise RuntimeError("Falta OPENAI_API_KEY en .env o tiene formato inválido (debe empezar con 'sk-').")

client = OpenAI(api_key=API_KEY)

SOAP_SYSTEM_PROMPT = """Eres un asistente clínico que redacta resúmenes en formato SOAP de consultas médicas.
- Idioma: español rioplatense neutral.
- Estilo: claro, conciso y profesional.
- Diferencia claramente lo subjetivo de lo objetivo. Por ejemplo: El dolor es subjetivo, la temperatura corporal es objetiva.
- No inventes datos: si algo no se menciona, indícalo como 'no referido'.
- No incluyas nombres de pacientes ni datos sensibles.
- NO incluyas información irrelevante al resúmen clínico.
- Agrega un descargo académico al final.

Formato:
# Resumen clínico (formato SOAP)
**S - Subjetivo:** ...
**O - Objetivo:** ...
**A - Análisis/Impresión clínica:** ...
**P - Plan (académico):** ...

_Descargo: Este resumen es solo con fines académicos y NO constituye diagnóstico médico._
"""

# Cuando se usa Flask con templates en la misma carpeta, es necesario este cambio.
# Si el index.html está en la carpeta "templates", volver a usar la #1 de abajo
#1 app = Flask(__name__)
app = Flask(__name__, template_folder='.')

def _transcribir(audio_file_path: str) -> str:
    with open(audio_file_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
            language="es"
        )
    return (getattr(tr, "text", "") or "").strip()

def _resumir(texto: str) -> str:
    mensajes = [
        {"role": "system", "content": SOAP_SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Transcripción de la consulta (texto literal):\n\n"
            f"{texto}\n\n"
            "Redacta el resumen clínico en formato SOAP siguiendo estrictamente las instrucciones del sistema."
        )}
    ]
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=mensajes,
        temperature=0.2,
        max_tokens=800,
    )
    return (chat.choices[0].message.content or "").strip()

def beautify_resumen(text: str) -> str:
    """
    Convierte un texto con formato SOAP (con o sin ** **) a HTML limpio,
    sin mostrar las etiquetas 'S/O/A/P' explícitas.
    """
    import re
    if not text:
        return ""

    # quitar marcas markdown básicas
    text = text.replace("**", "").replace("_", "")

    # encabezado opcional (si viene)
    text = re.sub(r"^#\s*Resumen.*?\n", "", text, flags=re.IGNORECASE)

    # secciones → HTML
    text = re.sub(r"S\s*-\s*Subjetivo:\s*", "<h4>Subjetivo</h4><p>", text, flags=re.IGNORECASE)
    text = re.sub(r"O\s*-\s*Objetivo:\s*", "</p><h4>Objetivo</h4><p>", text, flags=re.IGNORECASE)
    text = re.sub(r"A\s*-\s*An(á|a)lisis/?Impresi(ó|o)n cl(í|i)nica:\s*", "</p><h4>Análisis / Impresión clínica</h4><p>", text, flags=re.IGNORECASE)
    text = re.sub(r"P\s*-\s*Plan.*?:\s*", "</p><h4>Plan</h4><p>", text, flags=re.IGNORECASE)

    # cerrar último párrafo si hicimos alguno
    if "<p>" in text and not text.strip().endswith("</p>"):
        text += "</p>"

    # descargo → caja aparte
    text = re.sub(
        r"Descargo:\s*(.*)$",
        r"</p><div class='descargo'><em>\1</em></div>",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # limpieza de dobles cierres
    text = text.replace("</p></p>", "</p>")
    return text

def transcribir_y_resumir(audio_file_path: str):
    # 1) Transcripción
    texto = _transcribir(audio_file_path)
    if not texto:
        return None, "No se pudo transcribir el audio (texto vacío)."

    # 2) Resumen SOAP
    resumen = _resumir(texto)

    # 3) HTML lindo
    resumen_html = beautify_resumen(resumen)

    return {"texto": texto, "resumen": resumen, "resumen_html": resumen_html}, None

@app.get("/")
def home():
    # Pass the history to the template
    return render_template("index.html", summaries_history=summaries_history)

@app.post("/process")
def process():
    # Acepta un archivo 'audio' por multipart/form-data
    if "audio" not in request.files:
        return jsonify({"ok": False, "error": "No se recibió archivo (campo 'audio')."}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "Archivo vacío."}), 400

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        file.save(tmp_path)
        data, err = transcribir_y_resumir(tmp_path)
        if err:
            return jsonify({"ok": False, "error": err}), 400
        # Append summary to history
        summaries_history.append(data["resumen_html"])
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Error procesando el audio: {e}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

#Para hacer launch de forma local con visualstudiocode
#if __name__ == "__main__":
    #app.run(host="127.0.0.1", port=7860, debug=False)

#Para hacer launch desde railway o un servidor
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)

