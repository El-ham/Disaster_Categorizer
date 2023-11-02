import sys
import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk import word_tokenize
import re
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import pickle
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
nltk.download('punkt')
nltk.download('wordnet')


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):

        """
        Fit method for the TextLengthExtractor transformer.
        
        Args:
            X: Input data
            y: Target data (unused)
        
        Returns:
            self: Returns the instance of the transformer.
        """
        
        return self

    def transform(self, X):

        """
        Transform method for the TextLengthExtractor transformer.
        
        Args:
            X: Input data
        
        Returns:
            list of lists: A 2D array containing the length of each text in the input data.
        """
        
        # Calculate the length of each text in the input data
        text_lengths = [len(text) for text in X]
        # Reshape the result to a 2D array
        return [[length] for length in text_lengths]


def load_data(database_filepath):

    """
    Load data from a SQLite database.
    
    Args:
        database_filepath (str): The filepath to the SQLite database.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    pd.DataFrame: The features (X) and labels (Y).
    list: The category names.
    """
    
    db_name = database_filepath.replace(".db", "")
    db_name = db_name.split("/")[-1]
    
    # Load dataset from database with read_sql_table
    engine = create_engine('sqlite:///'+ database_filepath )
    df = pd.read_sql_table(db_name, con= engine)

# 2: Define feature and target variables X and Y

    # Define X and y
    X = df['message']
    y= df[['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'water', 'food', 'shelter', 'child_alone',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']]

    # delete columns with just one value
    
    for c in y.columns:
        if y[c].value_counts(dropna=False).nunique()==1:
            del df[c]
            del y[c]

    return X, y, y.columns


def tokenize(text):
    
    """
    Tokenize text data.

    Args:
        text (str): Text data to tokenize.

    Returns:
        list: List of tokens.
    """
    
    #get list of all urls
    url_regex= 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls= re.findall(url_regex, text)
    
    # replace each url with placeholder
    for url in detected_urls:
        text= text.replace(url, 'urlplaceholder')
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #iterate through each token
    clean_tokens=[]
    
    for tok in tokens:
        
        # lemmatize, normalize case and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok.lower().strip(), pos= "v")
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline.

    Returns:
        GridSearchCV: A GridSearchCV instance with a machine learning pipeline.
    """
    
    # text processing and model pipeline
    pipeline_tfidf_txt = Pipeline([
        
        ('features', FeatureUnion([
            
            ('nlp_pipeline', Pipeline([
                ('token', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())            
            ])),
            
            ('txt_length', TextLengthExtractor())
            
        ])),
        
        ('multi_output', MultiOutputClassifier(LogisticRegression()))
    ])

    # define parameters for GridSearchCV
    parameters_logestic_tfidf_txt = {    
        'features__nlp_pipeline__token__ngram_range': [(1, 1), (1, 2), (2, 2)],
        #'multi_output__estimator__n_estimators': [10, 20, 30],
        'features__nlp_pipeline__token__max_features': [1000, 5000, 10000],
        'multi_output__estimator__C': [0.1, 1, 10]
        
    }
    
    # create gridsearch object and return as final model pipeline
    cv_logestic_tfidf_txt = GridSearchCV(pipeline_tfidf_txt, param_grid = parameters_logestic_tfidf_txt)

    return cv_logestic_tfidf_txt



def evaluate_model(model, X_test, y_test, category_names):

    """
    Evaluate a machine learning model.

    Args:
        model: The trained machine learning model.
        X_test: The test features.
        y_test: The true labels.
        category_names: The names of the categories.

    Returns:
        None
    """
    
    y_pred_tfidf_logestic_txtlength_cv = model.predict(X_test)
    
    y_pred_tfidf_logestic_txtlength_cv_df = pd.DataFrame(y_pred_tfidf_logestic_txtlength_cv, columns=category_names)
      
    for column in category_names:
        true_labels = y_test[column]
        predicted_tfidf_logestic_txtlength_cv_labels = y_pred_tfidf_logestic_txtlength_cv_df[column]
        report_tfidf_logestic_txtlength_cv = classification_report(true_labels, predicted_tfidf_logestic_txtlength_cv_labels)
        print(report_tfidf_logestic_txtlength_cv)


def save_model(model, model_filepath):
    
    """
    Save the trained machine learning model to a file using pickle.

    Args:
        model: The trained machine learning model.
        model_filepath (str): The filepath to save the model.

    Returns:
        None
    """
    
    # Export model as a pickle file
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)


def main():

    """
    Main function for the machine learning pipeline.

    This function loads data from a SQLite database, builds and trains a machine
    learning model, evaluates the model, and saves it to a pickle file.

    Args:
        None (sys.argv is used to pass file paths).

    Returns:
        None
    """
    
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