from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import string

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('data.csv')
df['Label'] = df['Ratings'].apply(lambda x: 'Negative' if x <= 3 else 'Positive')
reviews = df[['Review text', 'Label']]
reviews['Review text'] = reviews['Review text'].astype(str)

# Instantiate WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Define clean function
def clean_text(doc):
    doc = doc.replace("READ MORE", " ")
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    doc = doc.lower()
    tokens = nltk.word_tokenize(doc)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

# Vectorize the data
vect = CountVectorizer(preprocessor=clean_text)
X = reviews['Review text']
y = reviews['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_num = vect.fit_transform(X_train)
X_test_num = vect.transform(X_test)

# Train the classifier
classifier = MultinomialNB()
classifier.fit(X_train_num, y_train)

# Function to scrape reviews from Flipkart
def get_reviews(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        review_containers = soup.find_all('div', class_=['col _2wzgFH K0kLPL', 'col _2wzgFH K0kLPL _1QgsS5'])
        dataset = []

        # Scrape ratings and reviews
        overall_rating = soup.find('div', class_=["_2d4LTz", "_3LWZlK _12yO4d"]).text
        dataset.append({'overall_rating' : overall_rating})
        
        for container in review_containers:
            if 'col' in container['class'] and '_2wzgFH' in container['class'] and 'K0kLPL' in container['class'] and '_1QgsS5' in container['class']:
                sub_row = container.find_all('div', class_='row')
                
                rating_regex = r'^\d+'
                
                review = sub_row[0].find('div').text.strip()
                review = review.replace('READ MORE' , '')
                
                rating = re.match(rating_regex, review).group()
                review = re.sub(rating_regex, '', review).strip()
                
                dataset.append({'rating' : rating, 'review': review})
                
            elif 'col' in container['class'] and '_2wzgFH' in container['class'] and 'K0kLPL' in container['class']:
                sub_row = container.find_all('div', class_='row')
                
                rating = sub_row[0].find('div').text.strip()
                
                review = sub_row[1].find('div').text.strip()
                review = review.replace('READ MORE' , '')
                
                dataset.append({'rating': rating, 'review': review}) 

        return dataset

    except Exception as e:
        print(f"Error: {e}")
        return []

# Function to extract product name from Flipkart URL
def extract_product_name(url):
    pattern = r'\/([a-zA-Z0-9-]+)\/product-reviews'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None

# Function to count positive and negative reviews
def count_positive_negative_reviews(reviews):
    positive_count = 0
    negative_count = 0
    
    for review in reviews:
        rating = int(review.get('rating', 0))
        if rating >= 4:
            positive_count += 1
        elif rating <= 2:
            negative_count += 1
    
    return positive_count, negative_count

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        url_input = request.form['url_input']
        
        # Handle input from URL
        if url_input:
            input_reviews = get_reviews(url_input)
            if input_reviews:
                input_text = [review['review'] for review in input_reviews if 'review' in review]
                product_name = extract_product_name(url_input)
                if product_name:
                    print("Product Name:", product_name)
                else:
                    print("Product name not found in the URL.")
            else:
                return render_template('result.html', message="No reviews found for the provided URL.")
        
        # Clean and vectorize input
        clean_input = [clean_text(review) for review in input_text]
        input_vector = vect.transform(clean_input)
        
        # Predict sentiment
        prediction = classifier.predict(input_vector)
        
        # Count positive and negative reviews
        positive_reviews = np.sum(prediction == 'Positive')
        negative_reviews = np.sum(prediction == 'Negative')
        
        return render_template('predict.html', positive_reviews=positive_reviews, negative_reviews=negative_reviews,product_name=product_name)

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        text_input = request.form['text_input']
        
        # Handle input text
        if text_input:
            input_text = text_input
        
        # Clean and vectorize input
        clean_input = clean_text(input_text)
        input_vector = vect.transform([clean_input])
        
        # Predict sentiment
        prediction = classifier.predict(input_vector)
        
        # Determine result (positive or negative)
        result = "Positive" if prediction[0] == "Positive" else "Negative"
        
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5007)


