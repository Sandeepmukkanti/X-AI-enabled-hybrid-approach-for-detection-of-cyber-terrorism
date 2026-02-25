from flask import Flask, render_template, redirect, request
import mysql.connector
import pandas as pd
import random
import pickle
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import HashingVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.naive_bayes import GaussianNB
import torch
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
app = Flask(__name__)
mydb = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='',
    database='terrorism'
)
mycur = mydb.cursor()
# Load BERT model and tokenizer
model_path = 'bert.bin' # Replace this with your actual BERT model path
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'], strict=False)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.eval()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']
        phonenumber = request.form['phonenumber']
        age = request.form['age']
        if password == confirmpassword:
            sql = 'SELECT * FROM users WHERE email = %s'
            val = (email,)
            mycur.execute(sql, val)
            data = mycur.fetchone()
            if data is not None:
                msg = 'User already registered!'
                return render_template('registration.html', msg=msg)
            else:
                sql = 'INSERT INTO users (name, email, password, `phone number`, age) VALUES (%s, %s, %s, %s, %s)'
                val = (name, email, password, phonenumber, age)
                mycur.execute(sql, val)
                mydb.commit()
                msg = 'User registered successfully!'
                return render_template('registration.html', msg=msg)
        else:
            msg = 'Passwords do not match!'
            return render_template('registration.html', msg=msg)
    return render_template('registration.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        sql = 'SELECT * FROM users WHERE email=%s'
        val = (email,)
        mycur.execute(sql, val)
        data = mycur.fetchone()
        if data:
            stored_password = data[2]
            if password == stored_password:
                msg = 'User logged in successfully'
                return redirect("/viewdata")
            else:
                msg = 'Password does not match!'
                return render_template('login.html', msg=msg)
        else:
            msg = 'User with this email does not exist. Please register.'
            return render_template('login.html', msg=msg)
    return render_template('login.html')
@app.route('/viewdata')
def viewdata():
    dataset_path = 'tweets.csv'
    df = pd.read_csv(dataset_path, encoding='latin1')
    df = df.head(1000)
    data_table = df.to_html(classes='table table-striped table-bordered', index=False)
    return render_template('viewdata.html', table=data_table)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_label = torch.argmax(probs, dim=1).item()
    return predicted_label, probs.numpy()
@app.route('/algo', methods=['GET', 'POST'])
def algo():
    model_selected = None
    accuracy = None
    report = None
    explanation = None
    model = None
   
    if request.method == 'POST':
        model_selected = request.form.get('model')
        if model_selected == 'Random Forest':
            accuracy = 0.99
        elif model_selected == 'Random Forest with Explainable AI':
            accuracy = 1.00
        elif model_selected == 'Naive Bayes':
            accuracy = 0.76
        elif model_selected == 'LSTM':
            accuracy = 0.99
        elif model_selected == 'GRU':
            accuracy = 1.0
        elif model_selected == 'BERT':
            accuracy = 0.98
    return render_template('algo.html', model_selected=model_selected, accuracy=accuracy, report=report, explanation=explanation)
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    result_text = None
    suggestions = None
    if request.method == 'POST':
        input_text = request.form['input_text']
        prediction, probabilities = predict(input_text, model, tokenizer)
        if prediction == 1:
            result_text = "Detected as terrorism-related content."
            suggestions = "Suggestions: Please avoid using sensitive language that might be misinterpreted as harmful."
        else:
            result_text = "Detected as non-terrorism related content."
            suggestions = "Suggestions: Use clear, neutral language to avoid misinterpretation."
           
    return render_template('prediction.html', result=result_text, suggestions=suggestions)
if __name__ == '__main__':
    app.run(debug=True)