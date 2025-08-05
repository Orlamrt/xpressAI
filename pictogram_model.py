import json
import os
import spacy
from ctransformers import AutoModelForCausalLM
import re

# Archivo de memoria de patrones
PATTERNS_FILE = "user_patterns.json"

# Cargar memoria de patrones
if os.path.exists(PATTERNS_FILE):
    with open(PATTERNS_FILE, "r", encoding="utf-8") as f:
        USER_PATTERNS = json.load(f)
else:
    USER_PATTERNS = {}

def save_patterns():
    with open(PATTERNS_FILE, "w", encoding="utf-8") as f:
        json.dump(USER_PATTERNS, f, ensure_ascii=False, indent=4)

# Cargar Spacy para español
nlp = spacy.load("es_core_news_sm")

# Cargar LLaMA 2 local en GGUF
MODEL_PATH = r"models\llama-2-7b-chat.Q4_K_M.gguf"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, model_type="llama", gpu_layers=0)  # CPU


# Archivo de categorías
CATEGORIES_FILE = "categories.json"
if os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
        CATEGORIES = json.load(f)
else:
    CATEGORIES = {}

# Lugares base
PLACES = ["sala", "cocina", "baño", "escuela", "parque", "jardín", "habitación", "cuarto"]

def save_categories():
    with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(CATEGORIES, f, ensure_ascii=False, indent=4)

def auto_classify_word(word: str) -> str:
    """Clasificación básica automática"""
    word_lower = word.lower()
    if word_lower in PLACES:
        return "lugar"
    doc = nlp(word_lower)
    pos = doc[0].pos_
    if pos == "VERB":
        return "verbo"
    elif pos in ["NOUN", "PROPN"]:
        if word_lower in ["mamá","papa","papá","yo","abuela","abuelo","hermano","hermana"]:
            return "sujeto"
        return "objeto"
    return "otro"

def classify_word(word: str) -> str:
    """Clasifica y aprende nuevas palabras"""
    word_lower = word.lower()
    category = CATEGORIES.get(word_lower)
    if not category:
        category = auto_classify_word(word_lower)
        CATEGORIES[word_lower] = category
        save_categories()
    return category

def generate_sentence(words: list, user_id: str = "default") -> str:
    """
    Genera oración optimizada en primera persona usando LLaMA con memoria de patrones.
    """
    # Normalizar las palabras para usar como clave de patrón
    key = "|".join([w.lower() for w in words])

    # Crear historial para usuario si no existe
    if user_id not in USER_PATTERNS:
        USER_PATTERNS[user_id] = {}

    # 1️⃣ Revisar si el patrón ya fue generado
    if key in USER_PATTERNS[user_id]:
        return USER_PATTERNS[user_id][key]

    # 2️⃣ Prompt ultra restrictivo optimizado
    prompt = (
        "Eres un generador de oraciones para una aplicación de comunicación aumentativa para personas con discapacidad del habla.\n"
        "Debes generar una sola oración corta, afirmativa, natural y clara en español usando únicamente las palabras que te doy.\n"
        "No agregues palabras extra, no agregues adjetivos, no inventes información, no hagas preguntas, "
        "no repitas palabras, no des explicaciones.\n"
        "Usa las palabras exactamente una vez, ordénalas de forma natural y mantén la oración lo más simple posible.\n\n"
        f"Palabras: {', '.join(words)}\n\n"
        "Oración:"
    )

    # 3️⃣ Generar con LLaMA
    output = model(prompt, max_new_tokens=30, temperature=0.2, top_k=30)
    sentence = output.strip()

    # 4️⃣ Limpieza automática
    if "Oración" in sentence:
        sentence = sentence.split("Oración")[-1].strip(" :")

    sentence = sentence.replace("\n", " ").strip()

    match = re.search(r'([A-ZÁÉÍÓÚÑ][^.?!]*[.])', sentence)
    if match:
        sentence = match.group(1).strip()
    else:
        sentence = sentence.strip()

    if sentence and not sentence[0].isupper():
        sentence = sentence[0].upper() + sentence[1:]

    # 5️⃣ Guardar patrón en memoria
    USER_PATTERNS[user_id][key] = sentence
    save_patterns()

    return sentence