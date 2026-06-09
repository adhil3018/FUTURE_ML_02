import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==================================
# DOWNLOAD STOPWORDS
# ==================================

nltk.download('stopwords')

# ==================================
# LOAD DATASET
# ==================================

df = pd.read_csv("data/customer_support_tickets.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ==================================
# REMOVE NULL VALUES
# ==================================

df = df.dropna(
    subset=[
        'Ticket Subject',
        'Ticket Description',
        'Ticket Type'
    ]
)

# ==================================
# TEXT CLEANING
# ==================================

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r'[^a-zA-Z ]',
        ' ',
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ==================================
# COMBINE IMPORTANT TEXT COLUMNS
# ==================================

df['combined_text'] = (

    df['Ticket Subject'].astype(str)

    + " "

    + df['Ticket Description'].astype(str)

    + " "

    + df['Product Purchased'].astype(str)

)

# ==================================
# CLEAN TEXT
# ==================================

df['clean_text'] = df['combined_text'].apply(clean_text)

# ==================================
# FEATURES + LABELS
# ==================================

X = df['clean_text']

y = df['Ticket Type']

# ==================================
# SPLIT DATA
# ==================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

# ==================================
# PIPELINE
# ==================================

pipeline = Pipeline([

    (

        'tfidf',

        TfidfVectorizer(

            max_features=10000,

            ngram_range=(1, 2),

            stop_words='english'

        )

    ),

    (

        'model',

        LinearSVC()

    )

])

# ==================================
# TRAIN MODEL
# ==================================

print("\nTraining model...")

pipeline.fit(

    X_train,

    y_train

)

# ==================================
# PREDICT
# ==================================

y_pred = pipeline.predict(X_test)

# ==================================
# ACCURACY
# ==================================

accuracy = accuracy_score(

    y_test,

    y_pred

)

print("\nAccuracy Score:")
print(accuracy)

# ==================================
# CLASSIFICATION REPORT
# ==================================

print("\nClassification Report:\n")

print(

    classification_report(

        y_test,

        y_pred

    )

)

# ==================================
# CONFUSION MATRIX
# ==================================

cm = confusion_matrix(

    y_test,

    y_pred

)

plt.figure(figsize=(10, 7))

sns.heatmap(

    cm,

    annot=True,

    fmt='d'

)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ==================================
# SAVE MODEL
# ==================================

joblib.dump(

    pipeline,

    "models/ticket_classifier.pkl"

)

print("\nModel saved successfully!")

# ==================================
# TEST SAMPLE
# ==================================

sample = [

    "payment failed and refund not received"

]

prediction = pipeline.predict(sample)

print("\nTest Prediction:")
print(sample[0])
print("Predicted Category:", prediction[0])