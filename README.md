# Sentiment-Analysis---Amazon-Reviews


Overview
* Dataset: Datafiniti Product Database
* Problem: Sentiment analysis for Amazon product reviews
* Objective: Perform binary classification on Amazon reviews by pre-processing text, converting it into vectors, and applying various machine learning (ML) and deep learning (DL) models. The goal is to compare the performance of these models.
Data Preprocessing
* Data Loading: Loaded a dataset with columns including Review ID, categories, keys, manufacturer, date, and review text.
* Data Slicing: Filtered the data to include only Amazon reviews and ratings.
* Data Cleaning: Removed unwanted spaces, punctuations, converted text to lowercase, and removed English stopwords.
* Feature Extraction: Transformed text into vectors using TF-IDF and count vectorizers, creating tokens from all words in the corpus.
Methodology
* Code: Amazon_Review_Analysis.ipynb
    * Visualized Amazon product reviews.
    * Preprocessed raw reviews into cleaned reviews.
    * Used word embedding models (count vectorizer, TF-IDF, and Word2Vec) to convert text reviews into numerical representations.
    * Applied ML models (Random Forest Classifier, Multinomial NB, Bernoulli NB, SVM) with different parameter variations and plotted accuracies.
    * Applied DL models (Neural Networks, LSTM, BERT).
Results
* Models Used: Two feature representation methods and seven ML and DL models.
    * Neural networks were run for 50 epochs.
    * LSTM and BERT were run for 3 epochs.
* Performance:
    * Best DL Model: Neural Networks with 94.06% accuracy.
    * Best ML Model: Count Vectorizer + Multinomial Naive Bayes with 93.43% accuracy.
    * Least Performing Models: TF-IDF Vectorizer + Bernoulli Naive Bayes (89.58%) and Count Vectorizer + Multinomial Naive Bayes (86.87%).
Model Accuracies (%)
Model	                   Count Vectorizer	    TF-IDF Vectorizer
Random Forest Classifier ->   93.21,	            93.34
Multinomial NB	->              93.43,	            93.28
Bernoulli NB	->                90.19,	            89.58
SVM	         ->                 86.87,	            93.58
Neural Networks	  ->            94.06	
LSTM	       ->                 94.23	
BERT	         ->                94	
