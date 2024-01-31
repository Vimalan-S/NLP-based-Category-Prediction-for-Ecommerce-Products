# NLP-based-Category-Prediction-for-Ecommerce-Products

# Abstract:
This project focuses on the development of a robust text classification system for categorizing ecommerce product descriptions into four main categories: Household, Electronics, Clothing and Accessories, and Books. Various natural language processing (NLP) techniques and machine learning models are employed to evaluate and compare their effectiveness in achieving accurate category predictions.

# Introduction:
The aim of this project is to automate the categorization of ecommerce product descriptions, streamlining the process for effective inventory management and user experience enhancement.

# Data Preprocessing:
Cleaned and preprocessed the text data by removing punctuation, extra spaces, and converting text to lowercase. Combined category labels with product descriptions, creating a unified dataset for model training.

# Train-Test Split:
Split the dataset into training and testing sets using the train_test_split function from scikit-learn, allocating 80% for training and 20% for testing.

# Text Representation Techniques:
Implemented various text representation techniques to transform product descriptions into numerical features for model training.

# Bag of Words (BoW):
- Utilized CountVectorizer with unigram features.
- Trained a Multinomial Naive Bayes classifier on the BoW representation.

# Combination of Bag of Words and Bi-Gram:
- Extended BoW to include bi-gram features.
- Assessed the impact of bi-grams on classification accuracy.

# TF-IDF (Term Frequency-Inverse Document Frequency):
- Employed TF-IDF vectorization to capture term importance.
- Trained a K-Nearest Neighbors classifier on the TF-IDF representation.

# Spacy's Word Embeddings:
- Utilized Spacy's word embeddings to represent text data in a continuous vector space.
- Scaled embeddings using MinMaxScaler and trained a Multinomial Naive Bayes classifier.

# Gensim Library - Word2Vec Embeddings:
- Leveraged Google NEWS Word2Vec embeddings from the Gensim library.
- Trained a Random Forest classifier on the Word2Vec embeddings.

# Model Evaluation:
Assessed the performance of each model using appropriate evaluation metrics such as precision, recall, and F1-score. Analyzed the impact of different text representation techniques on classification accuracy.

# Conclusion:
The project successfully demonstrates the effectiveness of various NLP techniques and machine learning models for ecommerce text classification. The findings provide valuable insights into selecting suitable approaches for product categorization in real-world applications.
