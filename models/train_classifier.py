import sys

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'])

import re
import pandas as pd
import numpy as np
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    """Load data from database.

    Args:
    database_filepath: filepath for the database file

    Returns:
    X: dataframe stores the message
    Y: dataframe stores the categories
    category_names: a list of all category column names
    """               
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * from DisasterResponse", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """Tokenize text.

    Args:
    text: message text

    Returns:
    clean_tokens: clean text tokens
    """                  
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Reduce words to their root form using default pos
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w) for w in words]
    
    return clean_tokens


def build_model():
    """Build a machine learning model.

    Args:
    None

    Returns:
    model: a machine learning model
    """                     
    # Build a machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # set parameters
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Report evaluation scores for the model.

    Args:
    model: a machine learning model
    X_test: messages in test dataset
    Y_test: categories in test dataset
    category_names: a list of all category column names

    Returns:
    classification report for each category
    """                         
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print('Performance report for category: {}'.format(category_names[i]))
        print(classification_report(np.array(Y_test)[:,i], y_pred[:,i]), '---------------------------------------------------')  


def save_model(model, model_filepath):
    """Export model as a pickle file.

    Args:
    model: a machine learning model
    model_filepath: filepath to save the model

    Returns:
    None
    """                             
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()