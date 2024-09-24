from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model('next_word_mode555l.h5')
with open('tokenizer (4).pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(text):
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
    pos = np.argmax(model.predict(padded_token_text), axis=-1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == pos:
            return word
    return ""

# Main route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    next_word = predict_next_word(text)
    return jsonify({"next_word": next_word})

if __name__ == '__main__':
    app.run(debug=True)
