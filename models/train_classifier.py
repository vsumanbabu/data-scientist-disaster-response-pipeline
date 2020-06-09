# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt','wordnet'])
import re
import pickle
from bs4 import BeautifulSoup

def load_data(database_filepath):
    '''
    Input:
        database_filepath: String
            Filepath of the database wher messages and categories are stored


    Logic:
        Read messages and categories from sql database
        Seperate messages as Input variable
        Seperate categoreis into predict variable

    Output:
        X: Dataframe
            Messages
        Y: Dataframe
            Categories the messages belong to
        category_names: List
            Lables for different categories
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('response',engine)
    X = df['message']
    Y = df.iloc[:,5:]
    Y.fillna(0,inplace=True)
    category_names = category_names=list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    '''
    Input:
        text: String
            Input Message

    Logic:
        Tokenize Message into relevant tokens using NLP techniques

    Output:
        tokens: List
            List of tokens
    '''
    tokenizer = RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(text)
    return tokens


def build_model():
    '''
    Input: None

    Logic: Create a pipleline and build Model

    Output:
        Pipeline: Pipleline
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Input:
        model: ML/NLP Model
        X_test: Test set
        Y_test: Predicted data set
        category_names:

    Logic:
        Measure the Accuracy of the predicted model

    Output:
        Accuracy

    '''
    Y_pred = model.predict(X_test)
    category_names=list(Y.columns)
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    '''
    Input:
        model: Model
            Model Pipeline
        model_filepath: String
            File Path where Model needs to be saved

    Logic:
        Create a Pickle file for the Trained Model

    Output: Pickle File
        Save the Model as a Pickel File

    Example:
        model_filepath='../models/classifier.pkl'
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    '''
    python \
    models/train_classifier.py \
    data/DisasterResponse.db \
    models/classifier.pkl
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        np.any(np.isnan(Y))
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
