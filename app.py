from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transliterate import translit, get_available_language_codes
import torch

app = Flask(__name__)

# Define models for each language pair
model_names = {
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
    "en-ar": "Helsinki-NLP/opus-mt-en-ar",
    "ar-en": "Helsinki-NLP/opus-mt-ar-en",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "de-en": "Helsinki-NLP/opus-mt-de-en",
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "es-en": "Helsinki-NLP/opus-mt-es-en",
    "en-it": "Helsinki-NLP/opus-mt-en-it",
    "it-en": "Helsinki-NLP/opus-mt-it-en",
    "en-ru": "Helsinki-NLP/opus-mt-en-ru",
    "ru-en": "Helsinki-NLP/opus-mt-ru-en"
}

tokenizers = {pair: AutoTokenizer.from_pretrained(model_name) for pair, model_name in model_names.items()}
models = {pair: AutoModelForSeq2SeqLM.from_pretrained(model_name) for pair, model_name in model_names.items()}

def translate_text(text, source_lang, target_lang):
    pair = f"{source_lang}-{target_lang}"
    if pair not in models:
        return f"Translation model for {pair} not available."

    tokenizer = tokenizers[pair]
    model = models[pair]

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=100)  # Adjust max_length as needed
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def transliterate_text(text, source_lang):
    available_languages = get_available_language_codes()
    if source_lang not in available_languages:
        return f"Transliteration for {source_lang} not available."

    transliterated_text = translit(text, source_lang, reversed=True)
    return transliterated_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    mode = data['mode']
    text = data['text']
    source_language = data['source_language']
    target_language = data['target_language']

    if mode == 'translate':
        result = translate_text(text, source_language, target_language)
    elif mode == 'transliterate':
        result = transliterate_text(text, source_language)
    else:
        result = 'Invalid mode selected.'

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
