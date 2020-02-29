# nycdsa_housing_ml_project

## Overview

The focus of this project was an analysis over home sales from 2006 to 2010 in Ames, Iowa. The goal of the project was to educate prospective buyers and sellers in the Ames, Iowa housing market in order to promote efficient, data-informed negotiating.

## Project Layout


The folder structure of the project is broken down into the following sections:

- **admin**: Coordination of project activites among team members.
- **code**: Combined, consesus code for purposes of project submission.
- **data**: Raw data files from Ames, Iowa Kaggle project. This includes the training data set of home sale samples with house price information (the target variable), a test data set of homes sales without pricing information, and the data dictionary associated with these files.
- **model_files**: Persisted, pre-trained model files utilized to make predictions from the unseen data in the test.csv file. For example, lr_log_model.joblib is the persisted linear regression model of the log-transformed home sale price derived from the "linear_model.ipynb/.py" files. 
- **pre_processeded_data**: Output files from pre-processing of raw data files from the pre_process.ipynb/.py files.
- **predictions**: Predicted house prices values from modles in 
- **work_area**: Represents experimentation area of individual team members prior to consilidation of thoughts and focus in the main repo section.

## Order of Evaluation

1. **code/EDA files**

- These files describe the initial preview and inspection of the Ames, Iowa housing sale data. The EDA (Explatory Data Analysis) highlights aspects of both the target variable (housing prices) and the potential feature set that were of greatest interest and then further explored and evaluated in the modeling phase of the project.

2. **code/pre_process files**

- Data preparation steps for purposes of model evaluation. This preparation was informed by the EDA process and set-up the modeling phase. As an example, the pre_process.ipynb/.py files process the initial raw train.csv file for purposes of modeling in the code/linear_model files. 

- The output of this process is reflected in the pre_processed_data folder. These output files are directly utilized in the modeling efforts.

- Pre-processing steps include feature engineering steps such as combining information from multiple columns (e.g., total building square footage as the sum total of basement sqft plus first and second floor square footage) and encoding of categorical variables (e.g, )

3. **code/modeling files**

- Modeling files utilized the pre-processed data files (output of 2nd step noted above) for purposes of developing and evaluating statistical models.

- Model evaluation steps include cross-validation and a review of residuals from the 80% "training" subset of the approximately 1450 housing sale transactions and the performance of the model on the un-seen 20% "test" subset.  

- The results of this step was a persisted, pre-trained model file in the *model_file* subbfolder. 

- As an example, the linear regression model trained and evaluated in the *linear_model* files outputs the resulting model to the *lr_log_model.joblib* file in the *model_files* folder.

4. **code/predition files**

- Pre-processing measures developed in step 2 of the evaluation process are applied to the house sale sample data in the *data/test.csv* file for purposes of making predictions on this data utilizing the pre-trained models from step 3 and uploading to Kaggle for model evaluation.

- For example, the code in the *lr_process_test_and_predict.ipynb* file outputs utilizes the persisted model file *model_files/lr_log_model.joblib* to make predictions on the data from the test.csv file. These prediciton results are outputed to the *predicitons/linear_predictions.csv* file.

## Miscellaenous Comments

This project utilized the python external package Jupytext (https://jupytext.readthedocs.io/en/latest/index.html). This package is utilized to create a paired .py file associated with a notebook/.ipynb file of the same name. This pattern was not utilized for all notebook files but in many instances it was utilized to allow for quicker review of the input cell code and reviewing git diffs/updates via the .py file.