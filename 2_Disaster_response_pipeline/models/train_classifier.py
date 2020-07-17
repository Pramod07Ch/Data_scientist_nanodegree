import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from nltk.corpus import stopwords
import nltk
import pickle
nltk.download(['punkt','wordnet', 'stopwords'])

def load_data(database_filepath):
    """
    load data from the sqlite database, Create feature and lables
    
    """

    # Read the table as pandas dataframe
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', con=engine)

    # Split the dataframe into X and y
    X = df['message']
    y = df.drop(columns=['id','message','original','genre'], axis = 1)

    # Get the label names
    category_list = y.columns

    return X, y, category_list
    

def tokenize(text):
    """
    Text processing
    """
    # tokenize scentences
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # save cleaned tokens
    refined_tokens = [lemmatizer.lemmatize(t).lower().strip() for t in tokens]
    refined_tokens = [t for t in refined_tokens if t not in stopwords.words("english")]
    return refined_tokens

def build_model():
    """
    Build pipeline with Gridsearch CV
    """
    # pipeine
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier (RandomForestClassifier()))
    ])
    
    # GridSearchCV
    parameters = {'clf__estimator__max_features':['auto', 'sqrt'],
              'clf__estimator__n_estimators':[50, 100]}

    pipeline = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=3, n_jobs=-1)

    return pipeline


def evaluate_model(model, X_test, y_test, category_names):    
    """
    Evaluate the model - classification report column wise
    """
    y_pred = model.predict(X_test)
    # build classification report on every column
    report = []
    for i in range(len(y_test.columns)):
        
        f1 = f1_score(y_test.iloc[:, i].values, y_pred[:, i], average='micro')
        p_s = precision_score(y_test.iloc[:, i].values, y_pred[:, i], average='micro')
        recall = recall_score(y_test.iloc[:, i].values, y_pred[:, i], average='micro')
        report.append([p_s, recall, f1])
    # build dataframe
    report_df = pd.DataFrame(report, columns=['precision', 'recall', 'f1'],
                                    index = y_test.columns)   
    print(report_df)


def save_model(model, model_filepath):
    """
    save the model
    """
    filename = '{}'.format(model_filepath)
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(database_filepath)
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