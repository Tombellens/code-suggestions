from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from flask_cors import cross_origin
app = Flask(__name__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sector_model = TFBertForSequenceClassification.from_pretrained('bert-fine-tuned-sector')
role_model = TFBertForSequenceClassification.from_pretrained('bert-fine-tuned-role')

sector_labels = ['academia', 'board', 'civilsoc', 'internatOrg', 'judiciary', 'media', 'other', 'politics(centr)',
 'politics(federalgov)', 'politics(gov)', 'politics(leg)', 'politics(local)', 'private', 'public', 'semiPrivate',
 'unknown']
role_labels = ['CoS', 'adviser', 'media', 'minister', 'nonMC', 'other', 'secretary', 'viceCoS']

def predict_sector(text):
    # Tokenize and prepare inputs for the model
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    # Get model outputs
    outputs = sector_model(**inputs)
    # Apply softmax to obtain probabilities
    probabilities = tf.nn.softmax(outputs.logits, axis=-1)
    probabilities = probabilities.numpy().tolist()[0]

    # Build the formatted response with label names and probabilities
    response = [{"code-value": label, "probability": prob} for label, prob in zip(sector_labels, probabilities)]
    return response

def predict_role(text):
    # Tokenize and prepare inputs for the model
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    # Get model outputs
    outputs = role_model(**inputs)
    # Apply softmax to obtain probabilities
    probabilities = tf.nn.softmax(outputs.logits, axis=-1)
    probabilities = probabilities.numpy().tolist()[0]

    # Build the formatted response with label names and probabilities
    response = [{"code-value": label, "probability": prob} for label, prob in zip(role_labels, probabilities)]
    return response

@app.route('/')
def hello_world():
    return 'Hello Sammy!'

@app.route('/code-probabilities-positions', methods=['POST'])
@cross_origin('*')
def predict_positions():
    try:
        # Extract input text from the JSON payload
        data = request.get_json()
        code_name = request.args.get('codeName')
        text = data['title'] + ' at ' + data['workplace'] + ": " + data['description']
        if code_name == 'sector':
            return(jsonify(
                predict_sector(text)
            ), 200)
        if code_name == "duration":
            return(jsonify(
                predict_role(text)
            ), 200)

        return jsonify("Invalid codeName"), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 400
