import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# NLTK resurslarini yuklab olish (agar internetga ulanish bo'lsa)
nltk.download('stopwords')
nltk.download('wordnet')

# Matnni oldindan qayta ishlash funksiyasi
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# 1. Ma'lumotlar to'plami yaratish
data = {
    'text': [
        "Win a free iPhone! Click here to claim your prize!",
        "Hurry up! Your account has been compromised, reset your password now!",
        "Hello, how are you doing today?",
        "Limited offer! Buy one get one free!",
        "Shall we meet for coffee tomorrow?"
    ],
    'label': ['spam', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})  # Spam: 1, Ham: 0

# Matnlarga oldindan ishlov berish
df['text'] = df['text'].apply(preprocess_text)

# 2. Ma'lumotlarni o'qitish va sinov to'plamlariga ajratish
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 3. Matnni raqamli ifodaga o'tkazish va model yaratish
text_clf = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),  # BoW usuli bilan matnni raqamlashtirish
    ('tfidf', TfidfTransformer()),  # TF-IDF hisoblash
    ('clf', MultinomialNB(alpha=0.1))  # Naïve Bayes klassifikatori
])

# 4. Modelni o'qitish
text_clf.fit(X_train, y_train)

# 5. Modelni baholash
y_pred = text_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred))

# 6. Yangi xabarlarni sinab ko'rish
new_messages = ["Congratulations! You have won a lottery!", "Hey, shall we meet for lunch?"]

new_messages = [preprocess_text(msg) for msg in new_messages]
predictions = text_clf.predict(new_messages)

for message, label in zip(new_messages, predictions):
    print(f"Message: {message} ---> {'Spam' if label == 1 else 'Ham'}")
