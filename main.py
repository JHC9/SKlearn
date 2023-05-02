import os
from sklearn import *
os.listdir()
import pandas as pd
avo_TRAIN = pd.read_csv('train_set.csv')
#drop NaN
avo_TRAIN.fillna(0, inplace=True)
# Select data for learning
f = ['employee_id','department','region','education','gender','recruitment_channel','no_of_trainings','age','previous_year_rating','length_of_service','KPIs_met >80%','awards_won?','avg_training_score']
X = avo_TRAIN[f]
y = avo_TRAIN.is_promoted

model = ensemble.RandomForestRegressor(random_state=2020, n_estimators=50)
Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.8)
print("training split: ", len(Xtrain), "; test split: ", len(Xtest))
model.fit(X, y)
model.score(Xtest, ytest)
