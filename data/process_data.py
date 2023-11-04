# Import Python libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
    This function loads data from CSV files and combines them into a single dataframe.
    
    Args:
        messages_filepath (str): Filepath for the messages CSV file.
        categories_filepath (str): Filepath for the categories CSV file.
    
    Returns:
        pd.DataFrame: Combined dataframe containing messages and categories.
    """
    
    # Load messages.csv into a dataframe
    messages = pd.read_csv(messages_filepath)
    # Load categories.csv into a dataframe
    categories = pd.read_csv(categories_filepath)
    # Merge the messages and categories datasets using the common id
    # Assign this combined dataset to df
    df = pd.merge(messages, categories)
    return df

def clean_data(df):
    
    """
    This function cleans the input dataframe by splitting the categories into separate columns.
    
    Args:
        df (pd.DataFrame): Input dataframe to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned dataframe with separate category columns.
    """    
    
    #Split the values in the categories column on the ; character so that each value becomes a separate column
    categories = df['categories'].str.split(';', expand=True)
   
    # Select the first row of the categories dataframe to create column names for the categories data
    row = categories.iloc[0,:]
    
    # Using this row to extract a list of new column names for categories.
    category_colnames = [i.split('-')[0] for i in row]
    
    # Rename the columns of `categories` with new column names
    categories.columns = category_colnames
    

# 4: Convert category values to just numbers 0 or 1

    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0), For example, related-0 becomes 0, related-1 becomes 1
    for column in categories:
        # set each value to be the last character of the string with slicing
        categories[column] = categories[column].apply(lambda x: x.split('-')[1] if '-' in x else x)
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        # replace column values to 1 if they are other than 0 or 1
        categories.loc[~categories[column].isin({0,1}), column] = 1

# 5: Replace categories column in df with new category columns

    # drop the original categories column from `df`
    del df['categories']

    # concatenate the original dataframe (df) with the new `categories` dataframe
    df = pd.concat([df, categories], axis= 1)

# 6: Remove duplicates

    # drop the duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):

    """
    This function saves the cleaned dataframe to an SQLite database.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe to be saved.
        database_filename (str): Filename for the SQLite database
    """    
    
    
    dbname = database_filename.replace(".db", "")
    dbname = dbname.split("/")[-1]
    
    # with pandas to_sql method combined with the SQLAlchemy library
    engine = create_engine('sqlite:///'+database_filename)
    
    if engine.dialect.has_table(engine.connect(), table_name = dbname):
        df.to_sql(name=dbname, con=engine, if_exists='replace', index=False)
        
    else:
        df.to_sql(dbname, engine, index=False)


def main():

    """
    Main function for the ETL process. Loads, cleans, and saves data.

    This function is the entry point for the ETL (Extract, Transform, Load) process.
    It loads data from input CSV files, cleans the data by splitting categories into
    separate columns, and saves the cleaned data to an SQLite database.

    Args:
        None (sys.argv is used to pass file paths).

    Returns:
        None
    """
    
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