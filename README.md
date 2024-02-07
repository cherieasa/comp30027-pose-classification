# comp30027-pose-classification
COMP30027 Machine Learning Project 1 (2021 Semester 1). Using a Supervised Naives Bayes Learner to classify poses from key points corresponding to 11 main parts of the body in (x,y) points. With a analysis report.

There are 4 main functions or steps.

## 1. Preprocess Data - preprocess()
- Read in csv and set dataframe for training and testing data
- Impute missing data as NaN

## 2. Train - train()
- Calculate prior probabilities to build Naive Bayes Model
- Group by class to get each class attributes to define the Gaussian Distribution
- Get mean of each class and standard deviation
- Calculate prior probabilities using value counts and summation

## 3. Predict - predict()
- Predict classes for new items in the dataset
- Calculate conditional probabilities for each class

## 4. Evaluation - evaluate()
- Compare model class to ground truth labels

## Report

### Performance Metrics
  - macro / micro averaging and weighted average
  - precision / recall / F-score

### Cosine Similarity
  - Engineer own pose features using angles between points instead to create a less correlated feature.
