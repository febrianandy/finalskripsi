from flask import Flask, request, render_template, json, jsonify
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC
import joblib


# config = {
#     "DEBUG": True,          # some Flask specific configs
#     "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
#     "CACHE_DEFAULT_TIMEOUT": 86400
# }

app = Flask(__name__, template_folder='templates', static_folder='static')
# app.config.from_mapping(config)
# cache = Cache(app)
# cache.init_app(app)


@app.route("/")
def main():

    return render_template("index.html")


@app.route("/svm", methods=['POST', 'GET'])
# @cache.cached(timeout=86400)
def svm():
    data_tweet = pd.read_csv(
        'https://raw.githubusercontent.com/febrianandy/hate_speech/main/data.csv', encoding='latin-1')
    alay_dict = pd.read_csv(
        'https://raw.githubusercontent.com/febrianandy/hate_speech/main/new_kamusalay.csv', encoding='latin-1', header=None)
    stopwords = pd.read_csv(
        'https://raw.githubusercontent.com/febrianandy/hate_speech/main/stopwords.csv', encoding='latin-1')
    alay_dict = alay_dict.rename(columns={0: 'original',
                                          1: 'replacement'})
    stopwords = stopwords.rename(columns={0: 'stopword'})
    alay_dict_map = dict(
        zip(alay_dict['original'], alay_dict['replacement']))

    # Preprocessing Data

    def lowercase(text):
        return text.lower()

    def remove_unnecessary_char(text):
        text = re.sub('\n', ' ', text)  # Remove every '\n'
        text = re.sub('rt', ' ', text)  # Remove every retweet symbol
        text = re.sub('user', ' ', text)  # Remove every username
        # Remove every URL
        text = re.sub(
            '((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)
        text = re.sub('  +', ' ', text)  # Remove extra spaces
        text = re.sub(r"\d+", "", text)  # Remove number
        text = text.encode('ascii', 'replace').decode(
            'ascii')  # Remove non ASCII
        # Remove hastag, mention
        text = ' '.join(
            re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
        return text

    def remove_nonaplhanumeric(text):
        text = re.sub('[^0-9a-zA-Z]+', ' ', text)
        return text

    def normalize_alay(text):
        return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

    def remove_stopword(text):
        text = ' '.join(
            ['' if word in stopwords.stopword.values else word for word in text.split(' ')])
        text = re.sub('  +', ' ', text)  # Remove extra spaces
        text = text.strip()
        return text

    def preprocess(text):
        text = lowercase(text)  # 1
        text = remove_nonaplhanumeric(text)  # 2
        text = remove_unnecessary_char(text)  # 3
        text = normalize_alay(text)  # 4
        text = remove_stopword(text)  # 5
        return text

    data_tweet = data_tweet[['Tweet', 'HS']]
    data_tweet['Tweet'] = data_tweet['Tweet'].apply(preprocess)

    X = data_tweet['Tweet']
    label = data_tweet['HS']
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vector = tfidf_vectorizer.fit_transform(X)
    tfidf_vector.shape

    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_vector, label, test_size=0.2, shuffle=True, random_state=42)

    import joblib
    csv = joblib.load('svm_model.pkl')
    inp = request.form['hate_speech']
    result = tfidf_vectorizer.transform([inp]).toarray()

    pred = csv.predict(result)
    accuracy = csv.score(X_test, y_test)
    if pred == 1:
        return jsonify({'hate_speech': 'Hate Speech', 'accuracy': accuracy})
    else:
        return jsonify({'hate_speech': 'Not Hate Speech', 'accuracy': accuracy})


@app.route("/decisiontree", methods=['POST'])
def decisiontree():
    if request.method == 'POST' or request.method == 'GET':
        data_tweet = pd.read_csv(
            'https://raw.githubusercontent.com/febrianandy/hate_speech/main/data.csv', encoding='latin-1')
        alay_dict = pd.read_csv(
            'https://raw.githubusercontent.com/febrianandy/hate_speech/main/new_kamusalay.csv', encoding='latin-1', header=None)
        stopwords = pd.read_csv(
            'https://raw.githubusercontent.com/febrianandy/hate_speech/main/stopwords.csv', encoding='latin-1')
        alay_dict = alay_dict.rename(columns={0: 'original',
                                              1: 'replacement'})
        stopwords = stopwords.rename(columns={0: 'stopword'})
        alay_dict_map = dict(
            zip(alay_dict['original'], alay_dict['replacement']))
        # Preprocessing Data

        def lowercase(text):
            return text.lower()

        def remove_unnecessary_char(text):
            text = re.sub('\n', ' ', text)  # Remove every '\n'
            text = re.sub('rt', ' ', text)  # Remove every retweet symbol
            text = re.sub('user', ' ', text)  # Remove every username
            # Remove every URL
            text = re.sub(
                '((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)
            text = re.sub('  +', ' ', text)  # Remove extra spaces
            text = re.sub(r"\d+", "", text)  # Remove number
            text = text.encode('ascii', 'replace').decode(
                'ascii')  # Remove non ASCII
            # Remove hastag, mention
            text = ' '.join(
                re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
            return text

        def remove_nonaplhanumeric(text):
            text = re.sub('[^0-9a-zA-Z]+', ' ', text)
            return text

        def normalize_alay(text):
            return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

        def remove_stopword(text):
            text = ' '.join(
                ['' if word in stopwords.stopword.values else word for word in text.split(' ')])
            text = re.sub('  +', ' ', text)  # Remove extra spaces
            text = text.strip()
            return text

        def preprocess(text):
            text = lowercase(text)  # 1
            text = remove_nonaplhanumeric(text)  # 2
            text = remove_unnecessary_char(text)  # 3
            text = normalize_alay(text)  # 4
            text = remove_stopword(text)  # 5
            return text

        data_tweet = data_tweet[['Tweet', 'HS']]
        data_tweet['Tweet'] = data_tweet['Tweet'].apply(preprocess)

        X = data_tweet['Tweet']
        label = data_tweet['HS']
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vector = tfidf_vectorizer.fit_transform(X)
        tfidf_vector.shape

        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_vector, label, test_size=0.2, shuffle=True, random_state=42)

        import joblib

        clf = joblib.load('tree_model.pkl')
        inp = request.form['hate_speech']
        result = tfidf_vectorizer.transform([inp]).toarray()

        pred = clf.predict(result)
        accuracy = clf.score(X_test, y_test)
        if pred == 1:
            return jsonify({'hate_speech': 'Hate Speech', 'accuracy': accuracy})
        else:
            return jsonify({'hate_speech': 'Not Hate Speech', 'accuracy': accuracy})


@app.route("/bagging", methods=['POST'])
def baggingdecisiontree():
    if request.method == 'POST' or request.method == 'GET':
        data_tweet = pd.read_csv(
            'https://raw.githubusercontent.com/febrianandy/hate_speech/main/data.csv', encoding='latin-1')
        alay_dict = pd.read_csv(
            'https://raw.githubusercontent.com/febrianandy/hate_speech/main/new_kamusalay.csv', encoding='latin-1', header=None)
        stopwords = pd.read_csv(
            'https://raw.githubusercontent.com/febrianandy/hate_speech/main/stopwords.csv', encoding='latin-1')
        alay_dict = alay_dict.rename(columns={0: 'original',
                                              1: 'replacement'})
        stopwords = stopwords.rename(columns={0: 'stopword'})
        alay_dict_map = dict(
            zip(alay_dict['original'], alay_dict['replacement']))
        # Preprocessing Data

        def lowercase(text):
            return text.lower()

        def remove_unnecessary_char(text):
            text = re.sub('\n', ' ', text)  # digunakan untuk menghapus setiap karakter baris baru (newline character)
            text = re.sub('rt', ' ', text)   # Menghapus simbol yang berulang
            text = re.sub('user', ' ', text)  # Menghapus setiap username
           # Menghapus setiap url
            text = re.sub(
                '((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text) 
            text = re.sub('  +', ' ', text)  # Menghapus spasi
            text = re.sub(r"\d+", "", text)  # Menghapus karakter number
            text = text.encode('ascii', 'replace').decode(
                'ascii')  # Remove non ASCII
            #Menghapus karakter hastag ,mention
            text = ' '.join(
                re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
            return text

        def remove_nonaplhanumeric(text): #Fungsi ini menerima string text sebagai input dan menghapus semua karakter selain huruf dan angka
            text = re.sub('[^0-9a-zA-Z]+', ' ', text)
            return text

        def normalize_alay(text):  #Fungsi ini menerima string text sebagai input dan mengubah kata-kata slang menjadi kata yang lebih umum
            return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

        def remove_stopword(text): # Menghapus stop word adalah proses pembersihan data teks dengan menghapus kata-kata yang tidak bermakna atau tidak memiliki informasi yang signifikan
            text = ' '.join(
                ['' if word in stopwords.stopword.values else word for word in text.split(' ')])
            text = re.sub('  +', ' ', text)  # Remove extra spaces
            text = text.strip()
            return text

        def preprocess(text): #fungsi ini menerima string text sebagai input dan mengembalikan string yang telah dipreproses
            text = lowercase(text)  # 1
            text = remove_nonaplhanumeric(text)  # 2
            text = remove_unnecessary_char(text)  # 3
            text = normalize_alay(text)  # 4
            text = remove_stopword(text)  # 5
            return text

        data_tweet = data_tweet[['Tweet', 'HS']] #Mengambil kolom Tweet dan HS
        data_tweet['Tweet'] = data_tweet['Tweet'].apply(preprocess) #Menerapkan fungsi preprocess pada kolom Tweet

        X = data_tweet['Tweet'] #Mengambil kolom Tweet
        label = data_tweet['HS'] #Mengambil kolom HS
        tfidf_vectorizer = TfidfVectorizer() #Menggunakan TF-IDF untuk mengubah teks menjadi vektor numerik
        tfidf_vector = tfidf_vectorizer.fit_transform(X) #Menerapkan TF-IDF pada data teks
        tfidf_vector.shape #Mengecek bentuk vektor TF-IDF

        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_vector, label, test_size=0.2, shuffle=True, random_state=42) #Membagi data menjadi data latih dan data uji dengan perbandingan 80:20 dan random state 42 untuk memastikan bahwa data yang dibagi sama setiap kali dijalankan

        import joblib

        clf = joblib.load('bagging_model2.pkl') #Menggunakan model yang telah dibuat sebelumnya
        inp = request.form['hate_speech'] #Mengambil inputan dari form
        result = tfidf_vectorizer.transform([inp]).toarray() #Menerapkan TF-IDF pada inputan yang diberikan dan mengubahnya menjadi vektor numerik dengan bentuk array
        
        pred = clf.predict(result)
        accuracy = clf.score(X_test, y_test)
        
        if pred == 1:
            return jsonify({'hate_speech': 'Hate Speech', 'accuracy': accuracy})
        else:
            return jsonify({'hate_speech': 'Not Hate Speech', 'accuracy': accuracy})

if __name__ == "__main__":
    app.run(debug=True)
