# comp30027-pose-classification
Using a Supervised Naives Bayes Learner to classify poses from key points corresponding to 11 main parts of the body in (x,y) points. With a analysis report.

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

## Keypoint dataset
Simplified keypoint dataset for yoga pose classification
K. Ehinger
Created for COMP30027 Machine Learning, sem 1, 2021, The University of Melbourne

Images: Yoga Pose Classification dataset
https://www.amarchenkova.com/2018/12/04/data-set-convolutional-neural-network-yoga-pose/

Column ids:
0. class
1. x1 (head)
2. x2 (shoulders)
3. x3 (elbowR)
4. x4 (wristR)
5. x5 (elbowL)
6. x6 (wristL)
7. x7 (hips)
8. x8 (kneeR)
9. x9 (footR)
10. x10 (kneeL)
11. x11 (footL)
12. y1 (head)
13. y2 (shoulders)
14. y3 (elbowR)
15. y4 (wristR)
16. y5 (elbowL)
17. y6 (wristL)
18. y7 (hips)
19. y8 (kneeR)
20. y9 (footR)
21. y10 (kneeL)
22. y11 (footL)

Missing values:
9999 = missing keypoint
