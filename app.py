from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from flask_cors import cross_origin
app = Flask(__name__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sector_model = TFBertForSequenceClassification.from_pretrained('bert-fine-tuned-sector')
role_model = TFBertForSequenceClassification.from_pretrained('bert-fine-tuned-role')
degree_model = TFBertForSequenceClassification.from_pretrained('bert-fine-tuned-degree')
field_model = TFBertForSequenceClassification.from_pretrained('bert-fine-tuned-field')

sector_labels = ['academia', 'board', 'civilsoc', 'internatOrg', 'judiciary', 'media', 'other', 'politics(centr)',
 'politics(gov)', 'politics(other)', 'politics(leg)', 'politics(local)', 'private', 'public', 'semiPrivate',
 'unknown']
role_labels = ['CoS', 'adviser', 'media', 'minister', 'nonMC', 'other', 'secretary', 'viceCoS']
degree_labels = ['PhD', 'bachelor(acad)', 'bachelor(prof)', 'highschool', 'master', 'other', 'unknown', 'nan']
field_labels = ['artshum', 'busecon', 'law', 'lifesci', 'na', 'natural', 'socialScience', 'unknown', 'nan']


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

def predict_degree(text):
    # Tokenize and prepare inputs for the model
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    # Get model outputs
    outputs = degree_model(**inputs)
    # Apply softmax to obtain probabilities
    probabilities = tf.nn.softmax(outputs.logits, axis=-1)
    probabilities = probabilities.numpy().tolist()[0]

    # Build the formatted response with label names and probabilities
    response = [{"code-value": label, "probability": prob} for label, prob in zip(degree_labels, probabilities)]
    return response

def predict_field(text):
    # Tokenize and prepare inputs for the model
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    # Get model outputs
    outputs = field_model(**inputs)
    # Apply softmax to obtain probabilities
    probabilities = tf.nn.softmax(outputs.logits, axis=-1)
    probabilities = probabilities.numpy().tolist()[0]

    # Build the formatted response with label names and probabilities
    response = [{"code-value": label, "probability": prob} for label, prob in zip(field_labels, probabilities)]
    return response


@app.route('/code-probabilities-positions', methods=['POST'])
@cross_origin('*')
def predict_positions():
    try:
        # Extract input text from the JSON payload
        positions = request.json['positions']
        code_name = request.args.get('codeName')
        code_values_objs = []
        for data in positions:
            text = data['title'] + ' at ' + data['workplace'] + ": " + data['description']
            if code_name == 'sector':
                code_values_objs.append(predict_sector(text))
            if code_name == "duration":
                code_values_objs.append(predict_role(text))
        return (jsonify(
            code_values_objs
        ), 200)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/code-probabilities-educations', methods=['POST'])
@cross_origin('*')
def predict_educations():
    try:
        # Extract input text from the JSON payload
        educations = request.json['educations']
        code_name = request.args.get('codeName')
        code_values_objs = []
        for data in educations:
            text = data['degree'] + ' in ' + data['field-of-study'] + " at " + data['institution']
            if code_name == 'field':
                code_values_objs.append(predict_field(text))
            if code_name == "degree":
                code_values_objs.append(predict_degree(text))
        return (jsonify(
            code_values_objs
        ), 200)

    except Exception as e:
        return jsonify({'error': str(e)}), 400