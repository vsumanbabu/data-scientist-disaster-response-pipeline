# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Input:
        messages_filepath: String
            File path for messages data
        categories_filepath: String
            File path for categories data

    Logic:
        Read csv files having messages and categories data
        Merge messges and categories

    Output:
        df_merge_message_categories: Dataframe
            Merge of messages and categories

    Examples -
        messages_filepath='../data/disaster_messages.csv'
        categories_filepath='../data/disaster_categories.csv'
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df_merge_message_categories = pd.merge(messages,categories,on='id')
    return df_merge_message_categories


def clean_data(df):
    '''
    Input:
        df: dataframe
            Merged dataframe of messages and categories
    Logic:
        Split the categories
        Rename Columns
        Convert to integers
        Drop duplicates

    Output:
        df: dataframe
            Cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)
    # drop the original categories column from `df`
    df.drop('categories',axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    Input:
        df: Dataframe
            Merged and split data of messages and responses

        database_filename: String
            Name of the database to be created

    Logic:
        Write the cleansed messages and categories to a sql database
    Output:
        Create database with table

    Example:
    database_filename=''sqlite:///../data/disaster.db''
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('response', engine,if_exists='replace', index=False)


def main():
    '''
    python  data/process_data.py         \
            data/disaster_messages.csv   \
            data/disaster_categories.csv \
            data/DisasterResponse.db
    '''
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
