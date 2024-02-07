# Import Libraries
import pandas as pd
import numpy as np
import math
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# This function should prepare the data by reading it from a file and converting it into a useful format for training and testing
def preprocess():

    # read in CSV file
    train_df = pd.read_csv("train.csv", header=None)
    test_df = pd.read_csv("test.csv", header=None)

    # set index names
    train_df.columns = ['class', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11']
    test_df.columns = ['class', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11']

    # get unique class names
    preprocess.classes = train_df.iloc[:,0].unique()

    # impute missing values as NaN
    preprocess.train_df = train_df.replace(9999, np.NaN)
    preprocess.test_df = test_df.replace(9999, np.NaN)

    return

# This function should calculate prior probabilities and likelihoods from the training data and using
# them to build a naive Bayes model
def train(training_df):

    # ------------------ Calculate means and standard deviation of each class x attribute ------------------
    
    # Groupby class to get each class' attributes (to define Gaussian distribution)
    grouped_df = training_df.groupby(['class'])

    # Define array to store mean and standard deviation values
    mean_std = []
    
    # Iterate to calculate mean and standard deviation for each class x attribute 
    for class_name, points in grouped_df:

        each_class_mean_std = ()            

        # Get mean of each class - ignore NaN values
        class_mean = grouped_df.get_group(class_name).mean(skipna = True, axis = 0)
        class_std = grouped_df.get_group(class_name).std(skipna = True, axis = 0)
        
        # Store as a array
        each_class_mean_std = [class_mean.to_numpy(), class_std.to_numpy()]

        # 2D Array where each element contains an array of a class' attributes - mean (as an array) and std (as an array)
        mean_std.extend([each_class_mean_std])

    # ------------------ Calculate prior probabilities of each class ------------------

    # Get counts of each class
    prior_prob_dict = training_df['class'].value_counts().to_dict()
    # Get total count
    total_count = training_df['class'].value_counts().sum()

    # Change value of count to prior probability of each class in dictionary
    for each_class, each_count in prior_prob_dict.items():
        prior_prob_dict[each_class] = each_count/total_count

    return mean_std, prior_prob_dict

# This function should predict classes for new items in a test dataset (for the purposes of this assignment, you
# can re-use the training data as a test set)

def predict(testing_df, mean_std, prior_prob_dict):
    
    # In this predict function, it calculates the evaluation for test set separately from training set (which used as a test set) 
    
    # Array to store the likelihood of each instance given it is from that class for test 
    test_likelihood = []
    # Array to store the likelihood of each instance given it is from that class for training set (used as test) 
    train_likelihood = []

    # ------------------ Calculate likelihood for test set ------------------
    
    # For every instance in test set
    for i in range(len(testing_df.index)):

        # Store the likelihoods for that instance as a 1x22 array.
        instance_test_likelihood = []
        # Store the sum of the logs of each likelihood for that instance
        sum_prob = 0
        
        # For every type of class we want to assume the instance belongs to
        for j in range(len(preprocess.classes)):

            # Get value of prior probability of the class (in the order of preprocess.classes array) from the dictionary
            prob_cj = prior_prob_dict[preprocess.classes[j]]

            # For every attribute - calculate the likelihood for each (class x attribute)
            for k in range(len(testing_df.columns)-1):
                # Get value of X
                xi = testing_df.iloc[i,k+1]

                # If value of Xi is NaN - we ignore it
                if (math.isnan(xi) == False):
                    
                    # Get mean and std from train function of that class x attribute
                    mean_axc = mean_std[j][0][k]
                    std_axc = mean_std[j][1][k]
                    # Calculate conditional probability from Gaussian distribution function
                    conditional_prob = np.double((1/(std_axc*math.sqrt(2*math.pi)))*math.exp(-0.5*((xi-mean_axc)/std_axc)**2))
                    
                    # If conditional probability is 0, estimate log of 0 with a large negative value
                    if (conditional_prob == 0):
                        sum_prob += -sys.maxsize
                    else:
                        sum_prob += math.log(conditional_prob)
            
            # Calculate likelihood of each class x attribute from the formula and append it to our 1x22 array
            likelihood = math.log(prob_cj) + sum_prob
            instance_test_likelihood.append(likelihood)
            # Reset this value
            sum_prob=0
        # Append each instance to our main array
        test_likelihood.append(instance_test_likelihood)

    # Add this to a new test result dataframe for test_df
    test_likelihood_df = pd.DataFrame(data=test_likelihood, columns = preprocess.classes)

    # ------------------ Predict class ------------------

    # Find maximum value of each column in a row (instance) and return the class name
    max_index_test = test_likelihood_df.idxmax(axis=1)

    # Append the column of predicted class name to the dataframe
    test_likelihood_df = test_likelihood_df.assign(predicted = max_index_test)

    return test_likelihood_df

# This function should evaluate the prediction performance by comparing your model’s class outputs to ground
# truth labels

def evaluate(predicted_df, original_df):

    # Append the ground truth values to the predicted dataframe
    predicted_df = predicted_df.assign(truth = original_df.iloc[:,0])

    # Extract predicted and truth columns of each test case
    y_test_pred = predicted_df.loc[:,"predicted"]
    y_test_truth = predicted_df.loc[:, "truth"]

    # ------------------ Calculate Accuracy Score ------------------
    y_test_accuracy = accuracy_score(y_test_truth, y_test_pred)

    # ------------------ Calculate Error rate ------------------
    y_test_error = 1-y_test_accuracy

    return [y_test_accuracy, y_test_error]

# Function to calculate the performance metrics for multiclass evaluation
def performance_metrics(predict_df, truth_df):

    # Print confusion matrix for visualisation
    confusion_mat = confusion_matrix(truth_df, predict_df)
    print(confusion_mat)
    # Create dictionary to store each class' FP/FN/TP/TN
    class_score_dict = {}
    
    # Calculate FP/FN/TP/TN for each class and append to dictionary (as an array)
    each_class_score = []
    for i in preprocess.classes:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        # For all instances in our testing set
        for j in range(len(predict_df)):

            y_pred_val = predict_df.iloc[j]
            y_truth_val = truth_df.iloc[j]

            # If the predicted and the truth value is the same as the class - considered True Positive 
            if (y_pred_val == y_truth_val and y_truth_val == i):
                TP+=1
            # If the predicted matches the class we are considering but it is not the truth value - considered False Positive
            if (y_pred_val == i and y_truth_val != y_pred_val):
                FP+=1
            # If the predicted classifies as negative which is the truth value - considered True Negative
            if (y_pred_val != i and y_truth_val != i):
                TN+=1
            # If the predicted classifies as negative but the truth value is positive - considered False Negative
            if (y_pred_val != i and y_truth_val == i):
                FN+=1
        class_score_dict[i] = np.array([TP, FP, TN, FN])
    
    return class_score_dict

# Function to calculate the precision of a given class (vs the rest)
def calc_precision(performance_array):

    # TP / (TP + FP)
    TP = performance_array[0]
    FP = performance_array[1]
    precision_score = TP/(TP+FP)

    return precision_score

# Function to calculate the recall of a given class (vs the rest)
def calc_recall(performance_array):

    # TP/ (TP + FN)
    TP = performance_array[0]
    FN = performance_array[3]
    recall_score = TP/(TP+FN)
    
    return recall_score

# Function to calculate the F-score 
def calc_f_score(precision_score, recall_score):

    f_score = 2*precision_score*recall_score/(precision_score+recall_score)

    return f_score

# Function to calculate macro-average for multiclass evaluation
def macro_averaging(class_score_dict):

    # Get sum of P and R per class 
    sum_P = 0
    sum_R = 0
    count_class = 0
    # For each class, find the precision and recall and sum it up
    for i in class_score_dict.values():
        sum_P += calc_precision(i)
        sum_R += calc_recall(i)
        count_class+=1

    # Sum up all of P and R and divide by count -> average
    macro_P = sum_P/count_class
    macro_R = sum_R/count_class
    macro_PR = [macro_P, macro_R]

    return macro_PR

# Function to calculate micro-average for multiclass evaluation
def micro_averaging(class_score_dict):

    # Get sum of TP, FP, FN
    sum_TP = 0
    sum_FP = 0
    sum_FN = 0
    for i in class_score_dict.values():
        # Get value from index of dict value
        sum_TP += i[0]
        sum_FP += i[1]
        sum_FN += i[3]
    
    # Calculate micro precision and micro recall based on formula
    micro_P = sum_TP/(sum_TP + sum_FP)
    micro_R = sum_TP/(sum_TP + sum_FN)
    micro_PR = [micro_P, micro_R]

    return micro_PR

# Function to calculate weighted average for multiclass evaluation
def weighted_averaging(class_score_dict, total_predicted):

    # Keep count of sum of Precision and Recall
    sum_proportion_P = 0
    sum_proportion_R = 0
    # Get count of predicted instances classes
    proportion_dict = {}
    proportion_dict = total_predicted.value_counts().to_dict()
    # Get total number of instances
    total_N = total_predicted.shape[0]

    # For every class, calculate and sum the total precision*proportion and recall*proportion
    for class_name, perf_metrics in class_score_dict.items():
        # Calculate class precision
        class_precision = calc_precision(perf_metrics)
        # Calculate class recall
        class_recall = calc_recall(perf_metrics)
        # Calculate proportion
        proportion = proportion_dict[class_name]/total_N
        # Add to sum
        sum_proportion_P += (class_precision*proportion)
        sum_proportion_R += (class_recall*proportion)

    weighted_PR = [sum_proportion_P, sum_proportion_R]

    return weighted_PR

# Function to calculate the cosine similarity between 2 vectors 
def cosine_sim(vector1, vector2):

    if (vector1[0] == np.NaN or vector2[0] == np.NaN):
        return np.NaN

    cos_sim = np.dot(vector1, vector2) / (np.sqrt(np.dot(vector1, vector1)) * np.sqrt(np.dot(vector2, vector2)))

    return cos_sim

# Function to return a vector given two coordinate points
def calc_vector(point_i, point_j):

    if (point_i[0] == np.NaN or point_j[0] == np.NaN):

        return np.array([np.NaN, np.NaN])

    else:
        x = point_j[0]-point_i[0]
        y = point_j[1]-point_i[1]

    return np.array([x, y])

# Engineer new features based on their cosine distance
def cos_engineer_features(engineer_df):

    instance_array = []
    total_cos_sim = []
    # For every instance calculate vectors for each limb
    for i in range(len(engineer_df)):

        # Extract coordinate points needed
        x1_y1 = (engineer_df.iloc[i,1],engineer_df.iloc[i,12])
        x2_y2 = (engineer_df.iloc[i,2],engineer_df.iloc[i,13])
        x3_y3 = (engineer_df.iloc[i,3],engineer_df.iloc[i,14])
        x4_y4 = (engineer_df.iloc[i,4],engineer_df.iloc[i,15])
        x5_y5 = (engineer_df.iloc[i,5],engineer_df.iloc[i,16])
        x6_y6 = (engineer_df.iloc[i,6],engineer_df.iloc[i,17])
        x7_y7 = (engineer_df.iloc[i,7],engineer_df.iloc[i,18])
        x8_y8 = (engineer_df.iloc[i,8],engineer_df.iloc[i,19])
        x9_y9 = (engineer_df.iloc[i,9],engineer_df.iloc[i,20])
        x10_y10 = (engineer_df.iloc[i,10],engineer_df.iloc[i,21])
        x11_y11 = (engineer_df.iloc[i,11],engineer_df.iloc[i,22])

        # Calculate vectors of limbs used
        one_two = calc_vector(x1_y1, x2_y2)
        two_three = calc_vector(x2_y2, x3_y3)
        two_five = calc_vector(x2_y2, x5_y5)
        two_seven = calc_vector(x2_y2, x7_y7)
        three_four = calc_vector(x3_y3,x4_y4)
        five_six = calc_vector(x5_y5, x6_y6)
        seven_eight = calc_vector(x7_y7, x8_y8)
        seven_ten = calc_vector(x7_y7, x10_y10)
        eight_nine = calc_vector(x8_y8, x9_y9)
        ten_eleven = calc_vector(x10_y10, x11_y11)

        # Calculate cosine similarity between limbs 
        one_two_three = cosine_sim(one_two, two_three)
        one_two_five = cosine_sim(one_two, two_five)
        one_two_seven = cosine_sim(one_two, two_seven)
        two_three_four = cosine_sim(two_three, three_four)
        two_five_six = cosine_sim(two_five, five_six)
        two_seven_eight = cosine_sim(two_seven, seven_eight)
        two_seven_ten = cosine_sim(two_seven, seven_ten)
        seven_eight_nine = cosine_sim(seven_eight, eight_nine)
        seven_ten_eleven = cosine_sim(seven_ten, ten_eleven)

        # Store instance in an array
        instance_cos_sim = [one_two_three, one_two_five, one_two_seven, two_three_four, two_five_six, two_seven_eight, two_seven_ten, seven_eight_nine, seven_ten_eleven]
        total_cos_sim.append(instance_cos_sim)
    
    # Add all of our cosine similarity values into a dataframe
    total_cos_sim_df = pd.DataFrame(data = total_cos_sim, columns = ["one_two_three", "one_two_five", "one_two_seven", "two_three_four", "two_five_six", "two_seven_eight", "two_seven_ten", "seven_eight_nine", "seven_ten_eleven"])
    # Add truth value to first column
    total_cos_sim_df.insert(0, 'class', engineer_df.iloc[:,0])

    return total_cos_sim_df

# Function to predict the test instances using cosine similarity
def cosine_pred(training_df, testing_df):

    cos_mean_std = train(training_df)[0]
    cos_prior_prob = train(training_df)[1]
    test_cos_sim_df = cos_engineer_features(testing_df)

    cos_likelihood_df = predict(test_cos_sim_df, cos_mean_std, cos_prior_prob)
    
    return cos_likelihood_df

# ------------------ MAIN FUNCTION CALL ------------------

# Function to Preprocess csv files into a dataframe
preprocess()
# print(preprocess.train_df)
# print(preprocess.test_df)
# print(preprocess.classes)

# Returns a 2D array of mean and std values, and prior probability of the training set
mean_std = train(preprocess.train_df)[0]
prior_prob_dict = train(preprocess.train_df)[1]

# print(mean_std)
# print(prior_prob_dict)

testing_predict = predict(preprocess.test_df, mean_std, prior_prob_dict)
training_predict = predict(preprocess.train_df, mean_std, prior_prob_dict)

# Evaluate for testing and training data set and combined
testing_eval = evaluate(testing_predict, preprocess.test_df)
training_eval = evaluate(training_predict, preprocess.train_df)

# Total evaluation (combined test and train)
total_predict = testing_predict.append(training_predict)
total_truth = preprocess.test_df.append(preprocess.train_df)

total_eval = evaluate(total_predict, total_truth)

# Print statistics from evaluation
print("-------- (x,y) Values Test Data --------")
print("Data Accuracy: %f" % testing_eval[0])
print("Data Error rate: %f" % testing_eval[1])
print("-------- (x,y) Values Training Data --------")
print("Data Accuracy: %f" % training_eval[0])
print("Data Error rate: %f" % training_eval[1])
print("-------- (x,y) Values Combined Data --------")
print("Data Accuracy: %f" % total_eval[0])
print("Data Error rate: %f" % total_eval[1])

# ------------------ QUESTION 1 MAIN FUNCTION CALL ------------------

# ------------------ QUESTION 1 MAIN FUNCTION CALL ------------------

print("\n-------- Total Data Confusion Matrix --------")
x_y_class_dict = performance_metrics(total_predict.loc[:, "predicted"], total_truth.iloc[:, 0])
print(" ")
print(x_y_class_dict)
print(" ")
# Shows the class count in our training data
print(total_truth['class'].value_counts())

# Calculate statistics
macro_PR = macro_averaging(x_y_class_dict)
macro_F = calc_f_score(macro_PR[0], macro_PR[1])
micro_PR = micro_averaging(x_y_class_dict)
micro_F = calc_f_score(micro_PR[0], micro_PR[1])
weighted_PR = weighted_averaging(x_y_class_dict, total_predict.loc[:, "predicted"])
weighted_F = calc_f_score(weighted_PR[0], weighted_PR[1])

print("\n-------- Total Data Macro-averaging Statistics --------")
print("Precision: %f , Recall: %f , F-Score: %f" % (macro_PR[0], macro_PR[1], macro_F))
print("-------- Total Data Micro-averaging Statistics --------")
print("Precision: %f , Recall: %f , F-Score: %f" % (micro_PR[0], micro_PR[1], micro_F))
print("-------- Total Data Weighted-averaging Statistics --------")
print("Precision: %f , Recall: %f , F-Score: %f" % (weighted_PR[0], weighted_PR[1], weighted_F))

# ------------------ QUESTION 6 MAIN FUNCTION CALL ------------------

cos_train = cos_engineer_features(preprocess.train_df)
combined_total_df = (preprocess.test_df.append(preprocess.train_df, ignore_index = True))

# Predict using testing set
cos_test_predicted = cosine_pred(cos_train, preprocess.test_df)
# Predict using training set
cos_train_predicted = cosine_pred(cos_train, preprocess.train_df)
# Predict using testing and training combined
cos_total_predicted = cosine_pred(cos_train, combined_total_df)

# Evaluate newly engineered feature
cos_test_eval = evaluate(cos_test_predicted, preprocess.test_df)
cos_train_eval = evaluate(cos_train_predicted, preprocess.train_df)
cos_total_eval =  evaluate(cos_total_predicted, combined_total_df)

print("\n-------- Cosine Similarity Engineered Test Data --------")
print("Data Accuracy: %f" % cos_test_eval[0])
print("Data Error rate: %f" % cos_test_eval[1])
print("-------- Cosine Similarity Engineered Training Data --------")
print("Data Accuracy: %f" % cos_train_eval[0])
print("Data Error rate: %f" % cos_train_eval[1])
print("-------- Cosine Similarity Engineered Total Data --------")
print("Data Accuracy: %f" % cos_total_eval[0])
print("Data Error rate: %f" % cos_total_eval[1])

print("\n-------- Cosine Similarity Confusion Matrix --------")
x_y_cosine_class_dict = performance_metrics(cos_total_predicted.loc[:, "predicted"], combined_total_df.iloc[:, 0])
print(" ")
print(x_y_cosine_class_dict)

# Calculate performance metrics
macro_PR = macro_averaging(x_y_cosine_class_dict)
macro_F = calc_f_score(macro_PR[0], macro_PR[1])
micro_PR = micro_averaging(x_y_cosine_class_dict)
micro_F = calc_f_score(micro_PR[0], micro_PR[1])
weighted_PR = weighted_averaging(x_y_cosine_class_dict, cos_total_predicted.loc[:, "predicted"])
weighted_F = calc_f_score(weighted_PR[0], weighted_PR[1])

print("\n-------- Total Cosine Similarity Data Macro-averaging Statistics --------")
print("Precision: %f , Recall: %f , F-Score: %f" % (macro_PR[0], macro_PR[1], macro_F))
print("-------- Total Cosine Similarity Data Micro-averaging Statistics --------")
print("Precision: %f , Recall: %f , F-Score: %f" % (micro_PR[0], micro_PR[1], micro_F))
print("-------- Total Cosine Similarity Data Weighted-averaging Statistics --------")
print("Precision: %f , Recall: %f , F-Score: %f" % (weighted_PR[0], weighted_PR[1], weighted_F))


