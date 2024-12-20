# Final Project Report: Predicting Autism

**Course**: CS383 - Introduction to Machine Learning  
**Semester**: Fall 2024  
**Team Members**: Hilary Lutz
**Instructor**: Professor Adam Poliak

---

## Table of Contents
1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Problem Statement](#problem-statement)
4. [Related Work](#related-work)
5. [Data Description](#data-description)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Discussion](#discussion)
9. [Conclusion and Future Work](#conclusion-and-future-work)
10. [References](#references)

---

## Abstract

In this project, I sought to answer which traits are most indicative of an autism diagnosis using machine learning techniques taught in this class such as Logistic Regression and Decision Trees. The dataset I used was from an autism diagnosis analysis challenge from Kaggle with eight hundred data points and twenty one features. First, I imported the datasets using Pandas, loading them into a Dataframe for ease of processing. After that, for preprocessing I dropped the columns that didn’t provide any information of use, such as the ID column which didn’t actually correlate to any diagnosis because it is completely arbitrary. I also one hot encoded the data as part of preprocessing so I could use it more easily with Logistic Regression. In order to isolate the exact features that correlated most with an autism diagnosis I used ablation, dropping individual columns at a time to see which column’s loss had the greatest impact on accuracy. My results showed that one of the greatest indicators of whether or not someone had autism is whether or not they had a family member with autism. 

---

## Introduction

For this project, I sought to find out which traits are most important for determining an autism diagnosis using the Autism Prediction dataset from Kaggle. According to the CDC, around one out of thirty-six children are diagnosed with Autism Spectrum Disorder. This is important because autism can severely impact the lives of the people who have it, and thus identifying it quickly is essential. Machine learning is relevant to this because it can identify patterns that are not immediately apparent to humans. 
---

## Problem Statement

What traits are most indicative of an autism diagnosis? My original hypothesis was that the biggest factor, or one of the biggest factors, would be if the patient had a relative with autism. I didn’t think that factors such as jaundice would have much of an impact.

---

## Related Work

Dataset: Autism Prediction | Kaggle. This dataset came from Kaggle, and I chose it because it was the top dataset for this topic. 
---

## Data Description

- **Source(s)**: Kaggle.
- **Size and Format**: Eight hundred rows, twenty two features, and with String, Float, Integer, and Boolean data types.
- **Preprocessing**: Steps taken to clean or transform the data, including handling missing values or feature engineering. I used an imputer, a min-max scalar and ‘get dummies’ in order to clean and process the data so it could be used. Some key features were whether or not someone has a relative who has autism, the score of a certain autism test, where the patient is from, what ethnicity they are and what gender they are. 

---

## Methodology

1. The algorithms or models used: Logistic Regression, Decision Trees. 
2. Details of the training process: I used the train-test split in order to split my data into training and testing for the logistic regression and the Decision Tree classifier and then I used GridSearch and Kfold in order to tune the parameters in order to get the best results. I used the default train test split, which was 25% test, and 75% training data. 
3. Any hyperparameter tuning performed. I used GridSearchCV in order to tune my hyperparameters for both the Decision Tree and Logistic Regression.
4. I used Pandas, NumPy, and Sklearn. 

---

## Results
Top five important factors: 
According to Logistic Regression: Whether or not the patient had a relative with autism, whether or not the person used the app before, ethnicity, the age of the patient, A6 score on the A1-A10 test

According to the Decision Tree: A9 score on the A1-A10 test, A1 score on the A1-A10 test, A6 score on the A1-A10 test, the country where the patient lives, ethnicity

| Model          | Accuracy | Precision | Recall | F1 Score | Best Params
|-----------------|----------|-----------|--------|----------|----------|
| Logistic Reg.   | 0.856     |  0.63    |  0.75 |   0.685  | {'C': 1, 'max_iter': 500, 'penalty': 'l2'} |
| Decision Tree   | 0.747     |  0.42    | 0.5315  |  0.467   | {'criterion': 'entropy', 'max_features': 0.5}|


---

## Discussion

Sklearn worked well, and using Pandas for data preprocessing similarly worked well. The dataset is very unbalanced, so that affected the ability of the models to accurately predict the outcome. This was important because it gave the model a higher score despite the fact that it was not very good at identifying if someone actually did have autism, as since the vast majority of the people did not have autism if the model predicted that everyone had autism (it didn’t) then it would still have a relatively high score despite not being reliable. One hot encoding was difficult at first, and interpreting the dataset was difficult, especially with the spelling errors. Even interpreting what the data itself meant was difficult as some of the names are misleading. The true accuracy and the balanced accuracy were different, and I had to take that into account. The given results for the Logistic Regression algorithm show which features are the most important for predicting whether or not a patient has autism. However, the results are different from the Decision Tree model, which gives various scores on a specific autism test before any other results. I agree with the logistic regression model more because every metric for recording reliability is higher than the decision tree model.  

---

## Conclusion and Future Work

If I had more time and resources, I would test not just individual features, but combinations of features to see which ones worked the best together to predict whether or not someone has autism. I would also use more datasets, as the one I used is rather unbalanced, and only has eight hundred data points. I would also want to find out if any particular features corresponded with different levels of severity for autism, as well as whether or not they have any additional diagnoses. The most relevant features for an autism diagnosis according to the Kaggle dataset are, in order: whether or not a family member has autism, whether or not the patient used an autism testing app before, the ethnicity of the patient, the age of the patient, and then the scores of the answers to the questions on the autism screening exam. These results are not too surprising, as autism is known to have at least a slight genetic component, however I was expecting gender to be on the list when it is not. Jaundice was also a feature, but it did not make it to the most important features list. I felt that this project went relatively well. I didn’t learn anything groundbreaking, and the fact that my results were so different based on the random element and the model is a bit disconcerting to me because of how easy it would be, hypothetically, to skew the results in a certain way if one so pleased. I think sklearn was very helpful for this project, as it could do all of the algorithms I was planning on using anyways.
---

## References

Tensor Girl. Autism Prediction. https://kaggle.com/competitions/autismdiagnosis, 2022. Kaggle. 


