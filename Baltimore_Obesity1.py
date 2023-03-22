import numpy as np
import os
import seaborn as sns
import pandas as pd
# Load the data  import pandas as pd 

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
obesity= pd.read_csv("Baltimore_Obesity1.csv",encoding = "ISO-8859-1" )



#The goal here is to build an advance ML model that can predict propensity for a neighborhood to have obese residents due to the influence of environmental forces such as parks and fast food resetaurants 
#Here we use a similar technique that has been employed before for predicting house prices using ML
#Reference: https://medium.com/@manilwagle/predicting-house-prices-using-machine-learning-cab0b82cd3f

# Explore the data
obesity.head()

#Stats on the dataset and features
obesity.info()

#explore numerical variables
obesity.describe()

#Look at the distribution of each numeric variable
obesity.hist(bins=50, figsize=(20,15))
plt.show()


#We divide the datasets into train and test split with 80% of the data for model building and 20% of the data for testing the model. 

#Random sampling is used to create, train and test datasets. The closest fastfood location is assumed to be a good indicator of obesity for a census tract relative to others. To ensure that the test datasets is representative of various levels of proximity (a numeric variable) meaning that we will convert cloest fastfood into a categorical variable and create different levels of proximity of the cloest fastfood and use stratififed sampling instead of random sampling. 

from sklearn.model_selection import train_test_split
train_set, test_set= train_test_split(obesity, test_size=0.2, random_state=42)

obesity["ClosestFF"].hist()

## Here we check for the correct number of bins for the response variable.
##We then examine the distribution of closest proximity to a fast food restaurant and created 5 levels of proximity category
obesity["FF_cat"]= pd.cut(obesity["ClosestFF"],
                            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                            labels= [1,2,3,4,5])
obesity["FF_cat"].hist()

##The above displays the distribution of closest proximity and created a Closest FF proximity category. 

# Stratified sampling based on FF_cat (closest fast food proximity) to make the datasets more random and representative
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(obesity, obesity["FF_cat"]):
    strat_train_set = obesity.loc[train_index]
    strat_test_set = obesity.loc[test_index]

## Check if the strata worked for entire datasets
obesity["FF_cat"].value_counts() / len(obesity)
                                 
#We now look to see if the same proportion has been applied in the test sets.
strat_test_set["FF_cat"].value_counts() / len(strat_test_set)




###We return to the original state by dropping the FF _cat from the dataset 
for set_ in (strat_train_set, strat_test_set):
    set_.drop("FF_cat", axis=1, inplace=True)   

#Exploring high density areas

obesity.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

## We look at BMI with circle representing district population and color representing BMI

obesity.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=obesity["population"]/100, label="population", figsize=(10,7),
            c="OBESITYCrudePrev", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.legend()
##


## Exploring high density areas
obesity.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

## We look at BMI with circle representing district population and color representing BMI

obesity.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=obesity["population"]/100, label="population", figsize=(10,7),
            c="OBESITYCrudePrev", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.legend()



    
## Correlation plot
from pandas.plotting import scatter_matrix
attributes = ["OBESITYCrudePrev", "ClosestFF", "FF1mi",
              "FF2mi","ClosestParks","Parkshalfmi","Parks1mi","population"]
scatter_matrix(obesity[attributes], figsize=(12, 8))
#######save_fig("scatter_matrix_plot")


##Since Closest FF proximity is most important variable, we look at this more deeper
sns.jointplot(x="ClosestFF", y="OBESITYCrudePrev", data=obesity)



##Data Preparation to Feed into Machine Learning Models
### Here first we will create a copy and separate the target variable so that we are not doing the same transformation
obesity= strat_train_set.drop("OBESITYCrudePrev", axis=1)
obesity_labels = strat_train_set["OBESITYCrudePrev"].copy()

##As our dataset is complete - it is not necessary to impute missing values
#As all our attributes are numerical (no text attribute) we can feed it into the ML models. Otherwise we would need the hot encoding technique for the next step.
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse=False)

# We standardize all the numeric features except for target variable so they are not in differnet scales so the ML models will perform better.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([

        ('std_scaler', StandardScaler()),
    ])
obesity_num_tr = num_pipeline.fit_transform(obesity)
obesity_num_tr



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])

obesity_num_tr = num_pipeline.fit_transform(obesity)
obesity_num_tr


#If needed, the next step would require us to transform the categorical variable where applicable

num_attribs = list(obesity)
##cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs), 
    ])
obesity_prepared = full_pipeline.fit_transform(obesity)
obesity_prepared.shape




#Training a Machine Learning Model. Here we will train several ML models with the goal of finding the best model that fits our data, especially the test datasets. We will start with the Linear Regression.




#Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(obesity_prepared, obesity_labels)


#We have a working linear regression. We will try to do some prediction on few of the instances.
some_data = obesity.iloc[:5] 
some_labels = obesity_labels.iloc[:5] 
some_data_prepared = full_pipeline.transform(some_data)  
print("Predictions:", lin_reg.predict(some_data_prepared))

###And we compare the prediciton with the actual values.
### Compare against actual values
print("Labels:", list(some_labels))


#In first instance, our model is off by around 1.08 (percent of census tract with obese adults) Let’s measure RMSE of the regression model (not bad!); however this may not be represntative of how well the modle performs

from sklearn.metrics import mean_squared_error  
obesity_predictions = lin_reg.predict(obesity_prepared) 
#prepared
lin_mse = mean_squared_error(obesity_labels, obesity_predictions) 
lin_rmse = np.sqrt(lin_mse) 
lin_rmse
print(lin_rmse)


#The RMSE tells us that model has typical prediction error of 6.742 which is too big. We could try to add more feature or try more complex model to make model more accurate. Here we will try more complex models.




### We will now use the Decision Trees ML technique to see if we can produce a better model ###
### Using DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(obesity_prepared, obesity_labels)

#Now we have build a model. We evaluate the decision tree model again using RMSE.

obesity_predictions = tree_reg.predict(obesity_prepared)
tree_mse = mean_squared_error(obesity_labels, obesity_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

#This may result in an incurrent rmse (e.g. 0.0 = 100% accuracy which is not correct). Since, as we don’t want to touch the test dataset until we find our final model, we will usea  10 fold cross validation technique to split the training set into further training and validation set.

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, obesity_prepared, obesity_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


#We then examine the result decision tree model after cross validation. We examine the scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

#It is obvious here that the linear regression performed better than decision tree which has mean error of  7.3135 and a stnadard deviation of 1.053. We now look to see what will be the RMSE if we apply 10 fold cross validation in the regression.


## We use cross validation on linear regression
lin_scores = cross_val_score(lin_reg, obesity_prepared, obesity_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

#As evident here, the linear regression is still performing better than the decision tree as the linear regression still has a high mean error, this time of 7.118,  compare to 7.3135 for decision trees.




#Random Forest
#We now try a Random forest ML technique - which works by building multiple trees on random subset of features and averaging out their predictions.
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(obesity_prepared, obesity_labels)


#We then want to look at the RMSE of random forest on training sets.
obesity_predictions = forest_reg.predict(obesity_prepared) 
forest_mse = mean_squared_error(obesity_labels, obesity_predictions) 
forest_rmse = np.sqrt(forest_mse) 
print(forest_rmse)

# It is 2.0608 whichis good as it means tha thte model prediciton error is ~2 on training sets. Let's see if we will get a different result if we use cross validation on random forest



from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, obesity_prepared, obesity_labels,scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


#This is good. We see that thus far this is the best model with the error rate of 5.37 even though we see that error rate is pretty high in validation datasets compare to training sets. This suggests that there might be over fitting issue. 



#We will try our final model next

#Support Vector Machine
#We now try the Support Vector Machine ML technique

from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(obesity_prepared, obesity_labels)
obesity_predictions = svm_reg.predict(obesity_prepared)
svm_mse = mean_squared_error(obesity_labels, obesity_predictions)
svm_rmse = np.sqrt(svm_mse)
print("Our svm_rmse:")
print(svm_rmse)
display_scores(svm_rmse)


#RMSE of 7.08158 from the Support Vector Machine shows that Random Forest is still the best ML technique



#Fine-tuning
#We will fine tune our random forest model using grid search technique. This technique requires us to tell which hyper parameters we want to experiment and what values to try out. The grid search technique will the evaluate all the possible combination of hyper parameters values, using cross validation.


### Using grid search to fine tune the model. Random forest
from sklearn.model_selection import GridSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
forest_reg = RandomForestRegressor(random_state=42)


# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(obesity_prepared, obesity_labels)
print("grid_search.best_params_:")

print(grid_search.best_params_)


#We now look at the result of hyper parameter combination tested during the grid search.
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
    
#We see that combination of 4 feature and 10 estimators gives the lowest RMSE of 5.7165. When the problem and data in hand is large, it is t ypically recommended to use randomized search rather in lieu of the grid search as below.

print("Randomized hyper parameter search:")

# Randomized hyper parameter search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }
forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(obesity_prepared, obesity_labels)
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
    
#Using grid search, we now analyze the best model and its error. Lets start by looking the importance of features in the random forest model.

# Feature Importance
feature_importances = grid_search.best_estimator_.feature_importances_
print("feature_importances:")
feature_importances

#
attributes = num_attribs 
print("Features that are most important:")
print(sorted(zip(feature_importances, attributes), reverse=True))


#This step allows us the opportunity to understand which feature are most important - or has the highest impact, sometimes the lowest candidate variable can be dropped. Our model has different considerations however as we look at both of the opposing environmental forces (eg proximity to fast food and parks). As we seen earlier, proximity to fast food restaurants is top feature for the model.


#Evaluation of the Model on the Test Set
#We now evaluate the random forest model on the test set and deploy it into production.

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("OBESITYCrudePrev", axis=1)
y_test = strat_test_set["OBESITYCrudePrev"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)




#The RMSE of 3.93 is really good , so this is our final model and we can deploy this random forest model into the production. Computing the prediction interval of model is always a good ideas as it makes us aware how much the error can fluctuate.

# Computing 95% confidence interval
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
print( np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors))) )
#This tell us that the prediction error can fluctuate anywehre between 2.4878 and 5.0228089. We can accept this gap in confidence interval. 

##This model could be optimized and tuned more to add accuracy either by adding new features or engineering new features. This model can be used to predict the precentage of adults with obesity in a neighborhood which we can use to extrapolate to propensity to obesity due to enviornmental forces in any geographic location by just slightly fine tuning the features and parameters.

#future research should incorprate adding additional layers to the model, incorporate demographic attributes and consider its interplay with enviornmental variables. 
      

#Having the features availalbe, we can predict the perecentage of a neighborhood or census tract that may be obese
