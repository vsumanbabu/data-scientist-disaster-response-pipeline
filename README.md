# data-scientist-disaster-response-pipeline
Disaster Response Pipeline

# Project Structure/Skeleton

- app
| - template
| |- master.html            # main page of web app
| |- go.html                # classification result page of web app
|- run.py                   # Flask file that runs app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv    # data to process
|- process_data.py
|- InsertDatabaseName.db    # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl           # saved model

- README.md

### Setup
1. Install Miniconda - https://docs.conda.io/en/latest/miniconda.html
2. conda env remove -n disaster_reponse_pipeline_env
3. conda env create -f disaster_reponse_pipeline_env.yaml
4. conda info --envs
5. conda env list
6. conda activate disaster_reponse_pipeline_env
7. jupyter notebook
8. conda deactivate

### Github Repo
https://github.com/vsumanbabu/data-scientist-disaster-response-pipeline

### App Run Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
