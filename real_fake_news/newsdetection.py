import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)  # Removing numbers
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Removing extra spaces

    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]

    return ' '.join(words)

data = pd.read_csv('Fake_Real_News_Data.csv')

# Drop index column 
if '#' in data.columns:
    data.drop(columns=['#'], inplace=True)

# Converting labels to binary
data['label'] = data['label'].map({'FAKE': 0, 'REAL': 1})

# Keeping only text and label
data = data[['text', 'label']]

data['cleaned_text'] = data['text'].apply(clean_text)   # Uses NLTK for cleaning text
print(data.head())

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['text']).toarray()
y = data['label']  # label = 1 (real), 0 (fake)

#Training model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = MultinomialNB()  # or LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

def predict_and_explain(text):
    cleaned_text = clean_text(text)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()

    prediction = model.predict(vectorized_text)[0]
    
    if prediction == 0:
        result = "Fake News"
    else:
        result = "Real News"
    
    # Explanation: showing top features contributing to the prediction
    feature_names = np.array(tfidf.get_feature_names_out())
    top_features_idx = vectorized_text[0].argsort()[-5:][::-1]  # Top 5 
    top_features = feature_names[top_features_idx]

    explanation = "Top contributing words to the prediction: " + ", ".join(top_features)
    
    return result, explanation

