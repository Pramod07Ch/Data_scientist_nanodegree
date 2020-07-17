# Disaster Response Pipeline Project

### Requirements
- Numpy
- Pandas
- scikit-learn
- sqlalchemy
- nltk
- joblib
- plotly
- flask

### About the project

In this project, the Disaster response pipeline is built with extension to web app where the entered message can classify it to the particular available category.

### Insructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Overview of project working

The steps followed before the web app are:
1. ETL (Extract, load and transform) pipeline

	In this step, the avaibale data (.csv files) are processed and merged together. The processed file is saved as databasefile. 

2. Machine learning pipeline
	In this step, the data is loaded from the database. Text processing, feature extraction is followed. Then a classifier is trained in a way that it can classify a givne message to particular category. Model is saved as .pkl file.

3. Web app (Flask)
	The model is then deployed using flask. It can be found [here](https://view6914b2f4-3001.udacity-student-workspaces.com/)

