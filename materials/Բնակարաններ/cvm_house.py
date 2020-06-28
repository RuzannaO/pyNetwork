import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
# from sklearn.externals import joblib

houses=pd.read_csv('houses_train (1).csv',encoding='ISO-8859-1')

del houses['url']
del houses['Unnamed: 0']
del houses['region']
del houses['street']

houses_af=pd.get_dummies(houses, columns=['district','floor','condition','building_type'])

print(houses_af.head().to_string())

del houses_af['price']


x=houses_af.values
y=houses['price'].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
model=ensemble.GradientBoostingRegressor()
#
# # those attrs are the responsible for the performance of  our  gradientBoosting function
# # now we only determine the right combination for the best model
#
# #
param_grid = {
    'n_estimators': [500, 1000, 3000],
    'max_depth': [4, 6],
    'min_samples_leaf': [3, 5, 9, 17],
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'max_features': [1.0, 0.3, 0.1],
    }
#
# # define the grid search we want to run. Run it with 4 cpus in parallel.
gs_cv = GridSearchCV(model, param_grid, n_jobs=4)
#
# # Run the grid search - on only the training data!
gs_cv.fit(x_train, y_train)
#
# # Print the params that gave us the best result!
print(gs_cv.best_params_)
