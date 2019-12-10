# SETUP ENVIRONMENT
# conda install -c anaconda pandas
# conda install -c anaconda numpy
# conda install -c conda-forge matplotlib
# conda install -c anaconda scikit-learn
# https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/

import pandas as pd
import numpy as np

# THIS CODE EXECUTES THE FOLLOWING
# 1. From the category .log file loads videos belonging to the category
# 2. From the projected_train.csv loads all videos
# 3. Generates a list of booleans (0,1) for videos belonging to .log category 
# 4. Get 20% of the set as test
# 5. Runs Random Forrest regressor
# 6. Prints output

#---------------------------------------------------------------------------------
def randomForest (nrEstimator, videosCateg, videosAll):

    #loop videos to generate boolean array of videos/categories
    isCateg = []
    for i in range(len(videosAll)): 
        categ = 0
        for j in range(len(videosCateg)):
            if (videosAll[i] == videosCateg[j]):
                categ = 1
        isCateg.append(categ)

    dataset = pd.read_csv('projected_train.csv')

    # get attributes and labels
    X = dataset.iloc[:, 1:361].values # values of 361 inception values reduced by PCA
    y = isCateg

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=nrEstimator)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # View the predicted probabilities of the first 20 observations

    return y_test, y_pred
    # from sklearn import metrics
    # print('estimators:', nrEstimator)
    # #print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # #print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
def calcPrecisionRecall (groundTruth, rfPrediction):
  
  precision = 0
  recall = 0

  truePositives = 0 # identified and really belonging to category
  positivesTotal = 0 # total number of items in the category (ground truth)
  totalIdentified = 0 # total identified as belonging to category
  totalSample = len(groundTruth) # 1842 for this sample

  # 1. count positives total in sample
  for i in range (len(groundTruth)):
      if (groundTruth[i] == 1): 
          positivesTotal += 1

  # 2. count total identified (true positives + falsePositives)
  for i in range (len(rfPrediction)):
    if (rfPrediction[i] == 1): 
        totalIdentified += 1

  # 3. count true positives (true in category and predicted correctly)
  for i in range(len(groundTruth)):
        if (groundTruth[i] == rfPrediction[i]) and (groundTruth[i] == 1):
            truePositives+=1

  # 4. calculate Precision
  precision = truePositives / totalIdentified

  # 5. calculate Recall
  recall = truePositives / positivesTotal

  return precision, recall
#---------------------------------------------------------------------------------

groundTruth = []
rfPrediction = []
precision = 0
recall = 0
precisionList = []
recallList = []


# load array of videos from log (LIST OF VIDEOS BELONGING TO CATEGORY)
fileSet =  set(open('science_train.log').read().split())
vidCateg = list(fileSet) 
# create array of booleans for all videos from the csv file

# load all videos from CSV in a array (LIST OF ALL VIDEOS)
my_csv = pd.read_csv('projected_train.csv')
vidAll = my_csv['video'].values

#forestSize = [400,500,600,700]
forestSize = [200,300,400,500,600,700,800]
totalRuns = 10

for size in forestSize:
  print ('forest size', size)
  for i in range (totalRuns):
    print ('run', i)
    groundTruth, rfPrediction =  randomForest (size, vidCateg, vidAll)
    pr, rl = calcPrecisionRecall (groundTruth, rfPrediction)
    precision += pr
    recall += rl

  precision = precision / totalRuns
  recall = recall / totalRuns
  precisionList.append(precision)
  recallList.append(recall)
  precision = 0
  recall = 0

print ('precision ' + str(precisionList)[1:-1]  )
print ('recall    ' + str(recallList)[1:-1]  )


# 1st
# test with different trees
# calculate PR of each
# generate graph Precisionx-x Recall-Y

# 2nd
# Save best Forest
# Calculate PR
