from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and tokenizer
try:
    model = load_model('next_word_mode555l.h5')
    with open('tokenizer (4).pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

# Function to predict the next word
def predict_next_word(text):
    try:
        token_text = tokenizer.texts_to_sequences([text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
        pos = np.argmax(model.predict(padded_token_text), axis=-1)[0]

        for word, index in tokenizer.word_index.items():
            if index == pos:
                return word
    except Exception as e:
        print(f"Error during prediction: {e}")
    return ""

# Main route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        if text:
            next_word = predict_next_word(text)
            return jsonify({"next_word": next_word})
        else:
            return jsonify({"error": "No text provided!"}), 400
    except Exception as e:
        print(f"Error in predict route: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
