# SETUP ENVIRONMENT
# conda install -c anaconda pandas
# conda install -c anaconda numpy
# conda install -c conda-forge matplotlib
# conda install -c anaconda scikit-learn

import pandas as pd
import numpy as np

# THIS CODE EXECUTES THE FOLLOWING
# 1. From the category .log file loads videos belonging to the category
# 2. From the projected_test.csv loads all videos
# 3. Generates a list of booleans (0,1) for videos belonging to .log category 
# 4. Get 20% of the set as test
# 5. Runs Random Forrest regressor
# 6. Prints output


# load array of videos from log
fileSet =  set(open('science.log').read().split())
videosCateg = list(fileSet) 
# create array of booleans for all videos from the csv file

# load all videos from CSV in a array
my_csv = pd.read_csv('projected_test.csv')
videosAll = my_csv['video'].values

#loop videos to generate boolean array of videos/categories
isCateg = []
for i in range(len(videosAll)): 
    categ = 0
    for j in range(len(videosCateg)):
        if (videosAll[i] == videosCateg[j]):
            categ = 1
    isCateg.append(categ)

#print(videosCateg)
#print (videosAll)
#print (isCateg)

dataset = pd.read_csv('projected_test.csv')

# get attributes and labels
X = dataset.iloc[:, 1:361].values # values of 361 inception values reduced by PCA
y = isCateg

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))