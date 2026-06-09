
# AI Customer Support Ticket Classifier

This project is a Natural Language Processing (NLP) based machine learning application that automatically classifies customer support tickets into predefined categories such as Billing Inquiry, Technical Issue, Product Inquiry, Refund Request, and Cancellation Request.

The system uses text preprocessing techniques, TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, and a Linear Support Vector Classifier (LinearSVC) model to analyze ticket descriptions and predict the most relevant category. A user-friendly Streamlit web application was developed to allow real-time ticket classification through an interactive interface.

## Features

* Text preprocessing and cleaning
* NLP-based feature extraction using TF-IDF
* Machine Learning classification with LinearSVC
* Interactive Streamlit web application
* Real-time ticket category prediction
* Model persistence using Joblib
* Evaluation using Accuracy Score and Classification Report

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Streamlit
* Joblib

## Workflow

1. Data Collection and Preprocessing
2. Text Cleaning and Stopword Removal
3. TF-IDF Feature Extraction
4. Model Training using LinearSVC
5. Model Evaluation
6. Deployment using Streamlit
   
## Results

- The trained LinearSVC model achieved an accuracy of approximately 20–25% on the test dataset.

## Challenges & Limitations

Due to overlapping ticket descriptions and the synthetic nature of the dataset, the model struggled to distinguish between categories effectively, resulting in relatively low accuracy. This highlighted the importance of high-quality, well-separated, and realistic datasets in machine learning applications.

## Future Improvements

- Use real-world customer support datasets.
- Experiment with advanced NLP models such as BERT.
- Perform hyperparameter tuning and feature engineering.
- Address class overlap through improved data collection and preprocessing.

This project demonstrates the practical application of NLP and Machine Learning techniques for automating customer support workflows and improving ticket management efficiency.
=======
# FUTURE_ML_02
AI-powered Customer Support Ticket Classifier using NLP, TF-IDF Vectorization, LinearSVC, and Streamlit for automated ticket category prediction.
>>>>>>> e6c9baed33631ea683b53a577a701eb42d24064d
