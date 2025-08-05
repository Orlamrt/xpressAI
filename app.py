from flask import Flask, request, jsonify
from flask_cors import CORS
from pictogram_model import classify_word, generate_sentence

app = Flask(__name__)
CORS(app)  # 🔹 Habilitar CORS para todos los orígenes

@app.route('/generate_sentence', methods=['POST'])
def generate_sentence_api():
    data = request.json
    sequence = data.get("sequence")

    if not sequence:
        return jsonify({"error": "No se recibió la secuencia de palabras"}), 400

    # Dividir la secuencia en palabras
    words = sequence.split()

    # Clasificar cada palabra automáticamente
    processed_sequence = [{"word": w, "category": classify_word(w)} for w in words]

    # Generar oración con LLaMA
    sentence = generate_sentence(words)

    return jsonify({
        "sequence_processed": processed_sequence,
        "generated_sentence": sentence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
