###Breast cancer risk education and prediction application

Overview

This repository contains the codebook for a comprehensive web-based platform designed to educate individuals about breast cancer risk factors, provide interactive health activity tracking, and facilitate access to support groups. The platform leverages a machine learning model built with XGBoost to predict individual risk of developing breast cancer using data from the Breast Cancer Surveillance Consortium (BCSC), which provides de-identified patient data including demographics, clinical risk factors and screening outcomes. The data can be downloaded from https://www.bcsc-research.org/index.php/datasets/rf/ with subjecting to data use agreement.

##Features##
- Risk Prediction Model: An XGBoost classification model trained on BCSC data to estimate probability of developing breast cancer.
- Educational content: Informative videos covering healthy activities, diet recommendation for cancer patients
- Health Activity Tracker: A user-friendly component to log daily health activities and visualize progress during the day and over time.
- Support Links: list of local and national support groups, legal and financial counseling services and peer-based recovery support groups.

##Model Training and Evaluation##


##Model Evaluation##

Generates performance metrics including AUC, accuracy, precision, recall, and feature importance plots.

##Web Application##

The web application serves three main components:

Risk Calculator: Users can input personal risk factors to receive an individualized risk probability.

Activity Tracker: Interactive dashboard for logging and visualizing health activities.

Educational Hub: Videos organized by topic.
