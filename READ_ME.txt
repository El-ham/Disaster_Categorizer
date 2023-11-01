# Disaster Response Pipeline

Hello, welcome to my Disaster Response Pipeline project! In this project, I've developed a system that can classify text messages related to disaster responses into various categories. This system is designed to be a valuable resource during emergencies to efficiently categorize and respond to incoming messages.

## Overview

Disaster response is a critical task, and quick and accurate classification of messages is crucial for effective disaster management. My project focuses on creating a system that automates this classification process using machine learning techniques.

## Data Pipelines

### ETL Process

- **Extract**: I begin by extracting data from two sources: `messages.csv` and `categories.csv`. These datasets contain messages and their corresponding categories, respectively.

- **Transform**: Next, I perform data cleaning using pandas to prepare the data for modeling. This involves handling duplicates, splitting categories into separate columns, and converting category values into binary (0 or 1).

- **Load**: After data transformation, I load the cleaned dataset into an SQLite database (`DisasterResponse.db`).

### Machine Learning Pipeline

- **Data Splitting**: I split the data into a training set and a test set to prepare for model development and evaluation.

- **Text Processing and Model Development**: I create a machine learning pipeline that includes text processing steps and a multi-output classification model. The pipeline includes tokenization, TF-IDF transformation, TextLengthExtractor, and a Logistic Regression classifier. I've also used NLTK for text processing and scikit-learn's GridSearchCV for hyperparameter tuning.

- **Model Export**: The trained model is then exported as a pickle file (`classifier.pkl`) for future use.

## Python Scripts

I've organized my work into two Python scripts:

- `process_data.py`: This script processes the data and stores it in the SQLite database.
- `train_classifier.py`: This script builds the machine learning pipeline, trains the model, and exports it as a pickle file.

The script is designed to accept file paths as arguments, allowing users to use their data for model training.

## Flask Web App

I've created a web application using Flask to showcase the results. Users can interact with the system through the web app, which displays message classification results and data visualizations.

## Usage

To run the data processing and model training pipelines, use the following commands:

- `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
- `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

After running the pipelines, you can Go to `app` directory: `cd app` and launch the web app using `python run.py`. The web app will be accessible through your web browser.


## Acknowledgments

I would like to express my gratitude to Udacity for providing the project framework and guidance.

Thank you for exploring my Disaster Response Pipeline project, and I hope you find it useful and informative.