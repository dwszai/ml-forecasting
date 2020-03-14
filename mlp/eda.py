# **Exploratory Data Analysis (EDA)**

# In[1]:
""" 1. Import Necessary Libraries and datasets"""

# a. Load libraries
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import fuzzywuzzy
from fuzzywuzzy import process
import plotly.express as px
import warnings

# Created python modules
import missing_module as mm
import visualization_module as vm
import feature_engineer_module as fm
import algorithm_module as am

warnings.filterwarnings('ignore')
print("Libraries imported.")

# b. Load dataset
url = "https://aisgaiap.blob.core.windows.net/aiap6-assessment-data/scooter_rental_data.csv"
data = pd.read_csv(url)
print("Dataset loaded.")
data.head()

# c. Overview of dataset report

profile = ProfileReport(data)
profile

"""
Small feature engineering before seperating the dataset
    * Date is not compared in the correlation since its in data-time format.
    * Extract month feature from datatime to observe its correlation
    * Year not extracted due to limited observations in dataset (2011 and 2012 only)
    * Day is an insignificant correlation (extremely low correlation against target variables, 0.003)
    * Drop all the duplicate rows since they are redundant information
"""
# Drop duplicates from dataset
fm.duplicate_drop(data)

# Add month feature before seperating dataset
data['month'] = pd.to_datetime(data['date']).dt.month
# Month feature added
data.head()

# d. Seperate dataset into train and test sets (80/20)
train = data.iloc[:-3480, :]     
test = data.iloc[-3480:, :]

# Describe full data, train and test sets
print("Original dataset has {} rows and {} columns.".format(data.shape[0], data.shape[1]))
print("\nSeperating dataset into train and test set...\n")
print("Train set has {} rows and {} columns.".format(train.shape[0], train.shape[1]))
print("Test set has {} rows and {} columns.".format(test.shape[0], test.shape[1]))

# e. Understand training set
print(train.head())
# Statistics information about train set
print(train.describe().T)
# Understand each features datatype
print(train.dtypes)
# Display no. of numerical and categorical data types
print("\nThe total number of each data type: \n{}".format(train.dtypes.value_counts()))

# In[2]:
""" 2. Missing values"""
# Check train set for missing values (if any)
mm.check_missing(train)
# Check test set for missing values (if any)
mm.check_missing(test)
# Nice! No missing value at all (for future purposes if missing values involved)
mm.heatmap_missing(train)
mm.heatmap_missing(test)

# In[3]:
"""3. Target variable"""

""" 
It is very important to first understand our target variable(s) before proceeding with further steps.
* Target variable: Total number of users
* Made up of 2 target variables:
    * 1) guest-users
    * 2) registered-users
"""

# a. Plot visualization charts (histogram, boxplot and qq plot)

# Findings: abnormal distribution, right-skewed, outliers present
vm.plot_chart(train, 'guest-users')
# Findings: abnormal distribution, right-skewed, lesser outliers present
vm.plot_chart(train, 'registered-users')

# b. Check for Skewness and Kurtosis
print("Guest users:")
fm.skew_kurtosis_value(train, 'guest-users')
print("\nRegistered users:")
fm.skew_kurtosis_value(train, 'registered-users')

# c. Correlation of all features vs target variable
print("\nGuest users:")
guest_feat = (train.corr())["guest-users"].sort_values(ascending=False)[1:]
print(guest_feat)
print("\nRegistered users:")
reg_feat = (train.corr())["registered-users"].sort_values(ascending=False)[1:]
print(reg_feat)

"""
From above information (visualizaton charts, skew/kurt values, correlation stats table):
    * We can infer that guest and registered users are highly correlated. 
    * We can safely proceed on to combining guest and registered users into total users feature.
"""

# Target variable: total-users (guest + registered)
train['total-users'] = train['guest-users'] + train['registered-users']
test['total-users'] = test['guest-users'] + test['registered-users']

print("Added new feature: total-users")
print(train.head())

# Correlation of total-user vs other features
total_feat = (train.corr())["total-users"].sort_values(ascending=False)[1:]
print(total_feat)
# Total users values
print("\nTotal users:")
fm.skew_kurtosis_value(train, 'total-users')
# Findings: abnormal distribution, right-skewed, lesser outliers present
vm.plot_chart(train, 'total-users')

# In[4]:
"""4. Correlation charts"""

# Heatmap
vm.heatmap_full(train)

# Focused heatmap
vm.heatmap_focused(train, 'total-users', 6)

# Scatter plot (features to compare)
features = ['total-users', 'temperature', 'feels-like-temperature', 'hr', 'month', 'windspeed']
vm.scatter_plots(train, features)
"""
Based on the scatter plots above:
    * we can see several features with linear relationship with the target variable
    * the most popular hr and month periods
    * Outliers not visibly present (features vs target plots)
"""
# In[5]:
"""5. Outliers"""

# a. Univariate analysis (increase array dimension by 1)
total_users_adj = StandardScaler().fit_transform(train['total-users'][:, np.newaxis])
high_range = total_users_adj[total_users_adj[:, 0].argsort()][-10:]     # last 10 in range
low_range = total_users_adj[total_users_adj[:, 0].argsort()][:10]       # first 10 in range
print('Outer low range of the distribution:')
print(low_range)
print('\nOuter high range of the distribution:')
print(high_range)
"""Findings:
    * Low range values close to 0
    * High range values far from 0
    * No outliers present
"""
# b. Bivariate analysis
# Using statistics to check for outliers in temperature first
feature = 'temperature'
train[feature].describe()                
print("Max value of temperature: {}\n75th percentile value of weather: {}\nNo obvious outlier present".format(train[feature].max(), train[feature].quantile(0.75)))

# Scatter plot in-depth look
# Temperature vs total users
vm.bivariate_scatter(train, 'temperature', 'total-users')
# Feels-like-temperature vs total users
vm.bivariate_scatter(train, 'feels-like-temperature', 'total-users')

# Box plot hr vs total-users
vm.bivariate_boxplot(train, 'hr', 'total-users')
# Month vs total users
vm.bivariate_boxplot(train, 'month', 'total-users')
# Box plot weather vs total-users
vm.bivariate_boxplot(train, 'weather', 'total-users')

# Using plotly interactive boxplot to navigate each points, a clearer understanding of each hr class
fig = px.box(train, x='hr', y='total-users')
fig.show()

fig = px.box(train, x='month', y='total-users')
fig.show()

# Make a train copy to compare before and after transformation chart
pre_train = train.copy()

# Check for regression in features
vm.regression_check(train, 'total-users', 'temperature', 'feels-like-temperature')

# Find error variance across true line
vm.error_variance(train, 'temperature', 'total-users')

print("Before transformation")
vm.plot_chart(train, 'total-users')

# Transform target variable using log transformation
train["total-users"] = np.log1p(train["total-users"])
print("After transformation")
# Plot newly transformed
vm.plot_chart(train, 'total-users')

# Comparing before and after adjusted target vs feature
vm.compare_error_variance(pre_train, train, 'temperature', 'total-users')

# In[6]:
"""6. Feature Engineering"""

# a. Repetitive values

# Saving the target variable for y train set 
y = train['total-users'].reset_index(drop=True)

# Saving the target variable for y test set
y_test = test['total-users'].reset_index(drop=True)

# Combine train and test datasets together
full_data = pd.concat((train, test)).reset_index(drop=True)
# Remove the target variable 
full_data.drop(['total-users'], axis=1, inplace=True)

# View and understand repetitive reason, drop if uninformative (Result: no repetitive values)
fm.repetitive(full_data)

# b. Duplicates
fm.duplicate_drop(full_data)

# c. Fix skewed features  
vm.skew_plot(full_data, 'temperature', 'windspeed')
# View data skewness list
fm.skewness_list(full_data)

# Fix skewness using boxcox transformation
fm.fix_skewness(full_data)
print("Skewness fixed")
# Fixed skewed features
vm.skew_plot(full_data, 'temperature', 'windspeed')

# d. Fix inconsistent data
"""
Categorical feature usually contains alot of inconsistent data:
    * Capitalization
    * Formats
    * Categorical Values
    * Addresses

In this dataset, there's only 1 categorical feature: weather
As shown above, it contains capitalization and categorical values error.
The issues will be addressed in the following code sections
"""
# Check for inconsistency in categorical feature
weather = fm.inconsistent_feature_check(full_data, 'weather')

# Fix capitalization inconsistency
fm.capitalization_fix(full_data, 'weather')
# get the top 10 closest matches to "clear"
matches = fuzzywuzzy.process.extract("clear", weather, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
list(matches)

# Replace incorrect to specified correct class name
fm.replace_matches(full_data, 'weather', 'clear', 80)

# Replace incorrect to specified correct class name
fm.replace_matches(full_data, 'weather', 'cloudy', 80)

# Check for anymore inconsistency in categorical feature
fm.inconsistent_feature_check(full_data, 'weather')

# e. Add dummies!

# Before that, we need to drop the date column since its not correlated to the target variable, month and hr are sufficient
full_data.drop(['date'], axis=1, inplace=True)
full_data.shape

# Create dummy variabes
full_data = pd.get_dummies(full_data).reset_index(drop=True)
full_data.shape

# Seperate into train and test set
X = full_data.iloc[:len(y), :]
X_test = full_data.iloc[len(y):, :]

print(full_data.shape, X.shape, y.shape, X_test.shape, y_test.shape)


# Decide whether to drop overfitted features
overfits = fm.overfit_features(X)
print("List of overfitted features: \n{}".format(overfits))
X = X.drop(overfits, axis=1)
print("List of overfitted features: \n{}".format(overfits))
X_test = X_test.drop(overfits, axis=1)
print("Overfit features dropped.")

# In[7]:
"""7. Algorithm Selection"""
# Algorithm class object
a = am.Algorithms()

# Regularization algorithms
a.regularization_models()
# Regression algorithms
a.regression_models()

# Score of each model
score = a.rmseCV(X, y, a.ridge)
print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = a.rmseCV(X, y, a.lasso)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = a.rmseCV(X, y, a.elastic_net)
print("Elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = a.rmseCV(X, y, a.svr)
print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = a.rmseCV(X, y, a.lightgbm)
print("LightGBM: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = a.rmseCV(X, y, a.xgboost)
print("Xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

#score = a.rmseCV(X, y, a.stack_reg)
#print("Stack: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

# In[8]:
"""8. Fitting Models"""

# Fit model function from algorithm module class
a.fit_models(X, y)

# Blending models
a.blend_models_predict(X)

# X train prediction
y_pred = a.blend_models_predict(X)
print("Predicted values (log transformed): ")
print(list(y_pred)[:10])

print("\nExponential transform predicted values..\n")
# expm1 to reverse the logp1 values of predicted y values, floor to round into actual values
y_pred = np.floor(np.expm1(y_pred))
print("Normal predicted values: ")
print(list(y_pred)[:10])

# Compare to actual train set y value (total users)
print("\nActual values for comparison: ")
print(list(np.floor(np.expm1(y)))[:10])

"""
Pretty close based on the first 10 observations! 
However we would not want it to be too accurate as it may be due to overfitting which can lead to inaccurate predicton in the test set.
Lets look at the accuracy score based on 2 performance metrics
"""

# In[9]:
"""9. Evaluation"""

# b. Accuracy score

# TRAIN SET

# Training set accuracy score, lower RMSLE better accuracy
print('RMSLE score on train data:')
print(a.rmsle(y, a.blend_models_predict(X)))

# Training set accuracy score, lower RMSE better accuracy
print('RMSE score on train data:')
print(a.rmse(y, a.blend_models_predict(X)))

# TEST SET

# Comparing accuracy score in different models to determine the best model to deploy

# Test set accuracy score, lower RMSLE better accuracy
print("RMSLE Accuracy Score for each model: \n")

print("Blended models:", a.rmsle(y_test, np.floor(np.expm1(a.blend_models_predict(X_test)))))
print("Stack Regressor:", a.rmsle(y_test, np.floor(np.expm1(a.stack_reg_models.predict(np.array(X_test))))))
print("XGB:", a.rmsle(y_test, np.floor(np.expm1(a.xgb_model.predict(X_test)))))
print("LGB:", a.rmsle(y_test, np.floor(np.expm1(a.lgb_model.predict(X_test)))))
print("SVR:", a.rmsle(y_test, np.floor(np.expm1(a.svr_model.predict(X_test)))))
print("Ridge:", a.rmsle(y_test, np.floor(np.expm1(a.ridge_model.predict(X_test)))))
print("ElasticNet", a.rmsle(y_test, np.floor(np.expm1(a.elastic_model.predict(X_test)))))
print("Lasso", a.rmsle(y_test, np.floor(np.expm1(a.lasso_model.predict(X_test)))))

# Test set accuracy score, lower RMSE better accuracy
print("RMSE Accuracy Score for each model: \n")

print("Blended models:", a.rmse(y_test, np.floor(np.expm1(a.blend_models_predict(X_test)))))
print("Stack Regressor:", a.rmse(y_test, np.floor(np.expm1(a.stack_reg_models.predict(np.array(X_test))))))
print("XGB:", a.rmse(y_test, np.floor(np.expm1(a.xgb_model.predict(X_test)))))
print("LGB:", a.rmse(y_test, np.floor(np.expm1(a.lgb_model.predict(X_test)))))
print("SVR:", a.rmse(y_test, np.floor(np.expm1(a.svr_model.predict(X_test)))))
print("Ridge:", a.rmse(y_test, np.floor(np.expm1(a.ridge_model.predict(X_test)))))
print("ElasticNet", a.rmse(y_test, np.floor(np.expm1(a.elastic_model.predict(X_test)))))
print("Lasso", a.rmse(y_test, np.floor(np.expm1(a.lasso_model.predict(X_test)))))

"""
Seems like the test results have a close or lower RMSLE/RMSE scores which shows that we did not overfit our model!
Nice, lets submit our prediction!

But...

Model selection will be based on:
    * performance score
    * various performance metrics
    * types of models fits certain problem
    * validation score
    * final results! 
Of course, considerations for different dataset or business problems will be factored in the selection.
"""
# Create new submission file for prediction values
print('Submit prediction')
# Model selection
selected_model = a.stack_reg_models.predict(np.array(X_test)) 
pred_test = np.floor(np.expm1(selected_model))
#pred_test = blend_models_predict(X_test)
submission = pd.DataFrame({'Id': X_test.index, 'total-users': pred_test})
submission.to_csv('submission.csv', index=False)

# Glimpse of submitted results vs actual results
print("Submitted results: ")
print(list(pred_test)[:10])
print("Actual values for comparison: ")
print(list(y_test)[:10])

