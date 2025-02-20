import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load the data of  messages and categories then merge them
    """  
    # Read data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge the dfs
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    Cleaning the data - split, transfomr and drop
    """
    # split the categories into columns
    categories = df['categories'].str.split(';',expand=True)

    # get column names
    category_colnames = list(categories.loc[0, :].apply(lambda x : x.strip('-')[:-2]))
    categories.columns = category_colnames

    # value requirement in columns
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # select column values with binary values
#     categories = categories[categories['related'] != 2]
    
    # child alone column has single values, can be dropped
    categories.drop(['child_alone'], axis=1, inplace=True)
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # select column values with binary values
    df = df[df['related'] != 2]
    
    # drop duplicates
    df.drop_duplicates(keep=False,inplace=True)
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()