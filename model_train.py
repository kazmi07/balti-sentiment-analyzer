"""
Train Balti Sentiment Analysis Model
Run this script once to train and save the model
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
import joblib
import os

def train_balti_model():
    print("="*50)
    print("BALTI SENTIMENT ANALYSIS MODEL TRAINING")
    print("="*50)
    
    # Load data
    df = pd.read_excel('sentiment_analysis.xlsx', sheet_name='sentiment_analysis.csv')
    
    # Clean data
    df = df[df['balti'].notna() & (df['balti'] != '')]
    df['sentiment'] = df['sentiment'].str.lower()
    valid_sentiments = ['positive', 'negative', 'neutral']
    df = df[df['sentiment'].isin(valid_sentiments)]
    
    print(f"\nTotal records: {len(df)}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    # Separate sentences from words
    df['is_sentence'] = df['balti'].str.contains(' ', na=False)
    sentences_df = df[df['is_sentence'] == True].copy()
    words_df = df[df['is_sentence'] == False].copy()
    
    print(f"\nSentences: {len(sentences_df)}")
    print(f"Words: {len(words_df)}")
    
    # Build lexicons
    positive_lexicon = words_df[words_df['sentiment'] == 'positive']['balti'].tolist()
    negative_lexicon = words_df[words_df['sentiment'] == 'negative']['balti'].tolist()
    neutral_lexicon = words_df[words_df['sentiment'] == 'neutral']['balti'].tolist()
    
    print(f"\nLexicon sizes:")
    print(f"Positive: {len(positive_lexicon)}")
    print(f"Negative: {len(negative_lexicon)}")
    print(f"Neutral: {len(neutral_lexicon)}")
    
    # Function to extract lexicon features
    def get_lexicon_features(text, pos_lex, neg_lex, neu_lex):
        text_lower = text.lower()
        words = text_lower.split()
        
        pos_count = sum(1 for w in words if w in pos_lex)
        neg_count = sum(1 for w in words if w in neg_lex)
        neu_count = sum(1 for w in words if w in neu_lex)
        
        # Multi-word phrases
        for phrase in pos_lex:
            if ' ' in phrase and phrase in text_lower:
                pos_count += 1
        for phrase in neg_lex:
            if ' ' in phrase and phrase in text_lower:
                neg_count += 1
        for phrase in neu_lex:
            if ' ' in phrase and phrase in text_lower:
                neu_count += 1
        
        return [pos_count, neg_count, neu_count]
    
    # Add features
    sentences_df.loc[:, 'pos_count'] = sentences_df['balti'].apply(
        lambda x: get_lexicon_features(x, positive_lexicon, negative_lexicon, neutral_lexicon)[0]
    )
    sentences_df.loc[:, 'neg_count'] = sentences_df['balti'].apply(
        lambda x: get_lexicon_features(x, positive_lexicon, negative_lexicon, neutral_lexicon)[1]
    )
    sentences_df.loc[:, 'neu_count'] = sentences_df['balti'].apply(
        lambda x: get_lexicon_features(x, positive_lexicon, negative_lexicon, neutral_lexicon)[2]
    )
    
    # Prepare data
    X_text = sentences_df['balti'].values
    X_lexicon = sentences_df[['pos_count', 'neg_count', 'neu_count']].values
    y = sentences_df['sentiment'].values
    
    # Split
    X_text_train, X_text_test, X_lex_train, X_lex_test, y_train, y_test = train_test_split(
        X_text, X_lexicon, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Vectorize text
    vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=5000)
    X_text_train_vec = vectorizer.fit_transform(X_text_train)
    X_text_test_vec = vectorizer.transform(X_text_test)
    
    # Combine features
    X_train_combined = hstack([X_text_train_vec, X_lex_train])
    X_test_combined = hstack([X_text_test_vec, X_lex_test])
    
    # Train and compare models
    models = {
        'MultinomialNB': MultinomialNB(alpha=1.0),
        'BernoulliNB': BernoulliNB(alpha=1.0),
        'ComplementNB': ComplementNB(alpha=1.0)
    }
    
    best_model = None
    best_name = None
    best_accuracy = 0
    
    for name, model in models.items():
        model.fit(X_train_combined, y_train)
        y_pred = model.predict(X_test_combined)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name}: {acc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name
    
    print(f"\n✅ Best model: {best_name} with accuracy {best_accuracy:.4f}")
    
    # Save model and files
    os.makedirs('model', exist_ok=True)
    
    joblib.dump(best_model, 'model/balti_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    joblib.dump(positive_lexicon, 'model/positive_lexicon.pkl')
    joblib.dump(negative_lexicon, 'model/negative_lexicon.pkl')
    joblib.dump(neutral_lexicon, 'model/neutral_lexicon.pkl')
    joblib.dump(best_name, 'model/model_type.pkl')
    
    print("\n✅ Model saved successfully in 'model/' directory!")
    
    return best_model, vectorizer, positive_lexicon, negative_lexicon, neutral_lexicon, best_name

if __name__ == "__main__":
    train_balti_model()