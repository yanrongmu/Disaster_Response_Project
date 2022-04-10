# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Descriptions](#project)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code requires the following libraries:
1. pandas
2. numpy
3. re
4. sqlalchemy
5. nltk
6. sklearn

The code should run with no issues using Python versions 3.*.

## Project Descriptions<a name="project"></a>

For this project, I used a data set containing real messages that were sent during disaster events to create a machine learning pipeline for an API that classifies disaster messages, so that the messages can be sent to an appropriate disaster relief agency.

## File Descriptions<a name="files"></a>

Here's the file structure of the project:
~~~~~
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
~~~~~

### Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results<a name="results"></a>

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. Below are a few screenshots of the web app.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Appen for the data. Also thanks to Udacity for their support.
