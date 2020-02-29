# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: 'Python 3.7.6 64-bit (''base'': conda)'
#     language: python
#     name: python37664bitbaseconda78814975a87e45dd93a41087a924c115
# ---

# + pycharm={"is_executing": false}
import pandas as pd
# %pylab inline

# + pycharm={"is_executing": false}
file_path = r"../pre_processed_data/pre_processed.csv"

# import pre_processed file 
pre_process_df = pd.read_csv(filepath_or_buffer=file_path, index_col=0, header=0)
pre_process_df.sample(5)

# + pycharm={"is_executing": false}
len(pre_process_df.columns)

# + pycharm={"is_executing": false}
# remove target variables from data frame
X = pre_process_df.loc[:, pre_process_df.columns.difference(["saleprice", "log_saleprice"])]

y = pre_process_df["saleprice"]
y_log = pre_process_df["log_saleprice"]

print(X.shape)
print(y.shape)
y_log.shape

# + pycharm={"is_executing": false}
# train, test split with train set to 80%
# A linear regression model will be evaluated first in the absence of regularization
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_log, test_size= 0.2, random_state=42)

# + pycharm={"is_executing": false}
from sklearn.linear_model import HuberRegressor, LinearRegression
# Evaluation of Huber regressor against SalePrice w/o log-transform 
# Huber regression is a linear model that is more robust to outliers than the standard model, which penalizes the model
# for higher deviations.

hr = HuberRegressor()
hr.fit(X=X_train, y=y_train)
print(f"Train R2 is {hr.score(X=X_train, y=y_train)}")
print(f"Test R2 is {hr.score(X=X_test, y=y_test)}")


# + pycharm={"is_executing": false}
# Standard regression w/o log transform 
lr = LinearRegression()
lr.fit(X=X_train, y=y_train)
print(f"Train R2 is {lr.score(X=X_train, y=y_train)}")
print(f"Test R2 is {lr.score(X=X_test, y=y_test)}")

# + pycharm={"is_executing": false}
# Standard regression w/log transform 
lr_log = LinearRegression()
lr_log.fit(X=X_train_log, y=y_train_log)
print(f"Train R2 is {lr_log.score(X=X_train_log, y=y_train_log)}")
print(f"Test R2 is {lr_log.score(X=X_test_log, y=y_test_log)}")

# There is a slight improvement (~2%) in the train R2 and test R2 utilizing log transform 

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Model Evaluation - Linear Regression
# ### The following section evaluates the random error, constant variance and normal distribution with mean 0 assumption of linear model in the context of the four initial models utilizing a residual plot from Yellowbrick.
#

# + pycharm={"is_executing": false}
# Residual Plot for Huber LR with no log-transform
from yellowbrick.regressor import ResidualsPlot
rpv_hr = ResidualsPlot(hr)
rpv_hr.fit(X=X_train, y=y_train)
rpv_hr.score(X=X_test, y=y_test)
rpv_hr.poof()


# + pycharm={"is_executing": false}
rpv_lr = ResidualsPlot(lr)
rpv_lr.fit(X=X_train, y=y_train)
rpv_lr.score(X=X_test, y=y_test)
rpv_lr.poof()



# + pycharm={"is_executing": false}
# Residual Plot for LR with log transform 
rpv_lr_log = ResidualsPlot(lr_log)
rpv_lr_log.fit(X=X_train_log, y=y_train_log)
rpv_lr_log.score(X=X_test_log, y=y_test_log)
rpv_lr_log.poof()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Model Evaluation of Ordinary Least Squares -Log Transform
# - Evaluation of log-transformed OLS model as the residuals plot appeared to satisfy most of the principal assumptions of linear regression. 

# + pycharm={"is_executing": false}
import statsmodels.api as sm
X_add_constant = sm.add_constant(X_train_log)
ols_log = sm.OLS(y_train_log, X_add_constant)
ans_log = ols_log.fit()
print(ans_log.summary())

# + [markdown] pycharm={"name": "#%% md\n"}
# - based on the OLS review, several factors are deemed non-significant by the model (e.g., there is not enough evidience to support that they are important to predicting sales price). These preidctors are bedrooms, lotfrontage, and whether a home is a new home. 
# - There are several coefficients that on the surface do not appear to make sense - namely the negative coefficient associated with two_plus_cr_garabe, this coefficient is negative whereas the domain association with this feature being that a two car or more garage capacity is good for a house. 
# - The model will be recaliberated dropping these three features.
# - homeage is being excluded in favor of remodelage due to its lower significance. 
# - garagecars is being dropped because there is an overalp between that variable and the "two_plus_cr_garg" feature

# + pycharm={"is_executing": false}
X_train_log = X_train_log.loc[:, X.columns.difference(["bedroomsabvgr", "lotfrontage", "newHome", "homeage", "garagecars"])]
X_test_log = X_test_log.loc[:, X.columns.difference(["bedroomsabvgr", "lotfrontage", "newHome", "homeage", "garagecars"])]


lr_log.fit(X=X_train_log, y=y_train_log)
lr_log.score(X=X_test_log, y=y_test_log)
print(f"Train R2 is {lr_log.score(X=X_train_log, y=y_train_log)}")
print(f"Test R2 is {lr_log.score(X=X_test_log, y=y_test_log)}")

# + [markdown] pycharm={"name": "#%% md\n"}
# - The train and test R2 are very similar to the model (e.g., 90/89%) prior to dropping the four variables. 
# - The residual plot and stats model output will be evaluated to confirm that the prior assumptions still hold as well as to identify any other items to potentiall exclude before proceeding to cross-validation.
#

# + pycharm={"is_executing": false}
rpv_lr_log = ResidualsPlot(lr_log)
rpv_lr_log.fit(X=X_train_log, y=y_train_log)
rpv_lr_log.score(X=X_test_log, y=y_test_log)
rpv_lr_log.poof()

# + pycharm={"is_executing": false}
X_add_constant = sm.add_constant(X_train_log)
ols_log = sm.OLS(y_train_log, X_add_constant)
ans_log = ols_log.fit()
print(ans_log.summary())

# -

# - Evaluation of OLS for non-log LR

# + pycharm={"is_executing": false}
X_add_constant_non_log = sm.add_constant(X_train)
ols = sm.OLS(y_train, X_add_constant_non_log)
ans = ols.fit()
print(ans.summary())

# + pycharm={"is_executing": false}
# Prediction error plot to further evaluate normality of residual distribution
from yellowbrick.regressor import prediction_error

visualizer = prediction_error(lr_log, X_train_log, y_train_log)

# + [markdown] pycharm={"name": "#%% md\n"}
# - qq plot of prediction error appears to follow in a straight line, which is indicative of a normally distributed error term.
#

# + pycharm={"is_executing": false}
from yellowbrick.regressor import cooks_distance

cd_visualizer = cooks_distance(X=X_train, y=y_train_log)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Cross Validation through YellowBrick
# - linear log model is evaluated via 4-k fold

# + pycharm={"is_executing": false}
from sklearn.model_selection import KFold

from yellowbrick.model_selection import CVScores

# Instantiate the KFold settings
cv = KFold(n_splits=4, random_state=42)

cv_visualizer = CVScores(model=lr_log, cv=cv, scoring="r2")

cv_visualizer.fit(X=X_train_log, y=y_train_log) # fit data into visualizer 
cv_visualizer.poof()

# + [markdown] pycharm={"name": "#%% md\n"}
# - Median cross-validation R2 score is 89% and fairly consistent. 
# - Evaluating next via sci-kit learn's model selection package
#

# + pycharm={"is_executing": false}
from sklearn.model_selection import cross_val_score
lr_r2_scores = cross_val_score(estimator = lr_log, X = X_train_log, y = y_train_log, scoring = 'r2', cv= 4)
lr_rmse = -1 * cross_val_score(estimator= lr_log, X = X_train_log, y = y_train_log, scoring = "neg_mean_squared_error", cv=4)
def display_cv_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
 
print("CV scores for R2 are:")   
display_cv_scores(lr_r2_scores)
print("")
print("CV scores for RMSE are:")
display_cv_scores(lr_rmse)

# + [markdown] pycharm={"name": "#%% md\n"}
# - Based upon cross-validation and test R2, we appear to have a strong and consistent predictor of housing prices.
# - The mean average is .89, which is also the value of the test R2 on the 20% hold out test-set. The standard deviation is also fairly low (e.g., 1.3%), which indicates that our model is not overly sensitive to the input data set.
#
# -

# ## VIF Evaluation
#
# - VIF (Variance Inflation Factor) is reviewed to determine if there is any multi-collinearity. Generally, any feature with a VIF value of 5 or higher is generally regarded as a feature that is likely to be co-linear with another feature or combination of other features in the model. 

# +
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

features_vif = "+".join(X_train_log.columns)

features_target = pd.concat([X_train_log, y_train_log], axis="columns")

y_VIF, X_VIF = dmatrices('log_saleprice ~' + features_vif, features_target, return_type = "dataframe")

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_VIF.values, i) for i in range(X_VIF.shape[1])]
vif["features"] = X_VIF.columns

vif.round(1)

# -

# - There are no features with a VIF factor greater than 5, the general cutoff for multicollinearity "concern." The highest VIF value are 3.1 and 3.2 for adj_ovr_qual and good_ament_ct, respctively.

# ## Model Output as Picket Object
# - Output of model object to be utilized in prediction

# + pycharm={"is_executing": false}
from joblib import dump

dump(lr_log, '../model_files/lr_log_model.joblib')

# + pycharm={"name": "#%%\n", "is_executing": false}
lr_log.predict(X_test_log)



# + pycharm={"name": "#%%\n", "is_executing": false}
X_test_log.columns

# + pycharm={"name": "#%%\n"}


