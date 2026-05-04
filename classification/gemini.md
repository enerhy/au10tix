# Goal:
* Build a classification project (including the necessary scripts). I will ask you to update the project later on with new ideas, features and improvements. 

* At the end we will have a multihead classicator that will use a single backbone - e.g. ResNet18, with two branches for Race Classification and Gender Classification of facial images. The network will have one loss with a learnable parameter for the combination of the two tasks.

# Basis: 
The basis for the project are the scripts currently included with their funictionalities
- classification.py -> this is the main training script
- classification_config.yaml -> config file with a simple structure that will be parsed

# Data Loading
The data will be parsed via the csv file. 
The csv file will have the following columns:
- img_path
- gender
- race
- split (e.g. 0 = train, 1 = val, 2 = test)
- version
- other_attributes

* gender is either 0 (male) or 1 (female)
* race is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).

The classification_config.yaml will give the path to the dataset csv file


# Tracking:
Use mlflow for tracking. However, use a simpe local mlflow -> update the code where needed
* track the config file plus the most important paramaeters

# Models storage 
- Base models are defined in /classification folder

# Storage
Use ML flow in the project and store the models accordingly.

# Desing:
Try to reuse as much of the current code as possible. What is not necessary should be removed. Follow the principles of clean code.

# Code testing:
For the testing I will provide you with the data file. 