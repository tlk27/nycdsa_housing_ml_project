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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Introduction
#
# We are examining the Ames, Iowa data set found on [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). The data set contains data on approximately 1,400 home sales between 2006 and 2010. 
#
# Our goal is to predict the price of homes from an unknown test set. Moreover, we seek to examine some select features that impact the price of home sales. This information can be used by realtors, homeowners, or homebuyers to make more informed decisions when buying or selling a home.   

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# # adjust figure size & dpi as needed
# from pylab import rcParams
# rcParams['figure.figsize'] = 10,5
# rcParams['figure.dpi'] = 300

# # to display max columns or rows as needed 
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows', None)

# %matplotlib inline

# +
train_o = pd.read_csv('./house-prices-advanced-regression-techniques/train.csv')

test_o = pd.read_csv('./house-prices-advanced-regression-techniques/test.csv')

# +
train = train_o.copy()

test = test_o.copy()
# -

# Make the columns names lowercase:

train.columns = map(str.lower, train.columns)

print("The training dimensions: ", train.shape)
print("The test dimensions: ", test.shape) #price column missing and 1 record missing

# Examination of the target variable indicates a more normal distribution after taking the log transformation.
# - The transformation has a low skewness, however the kurtosis is also lower than the normal distribution (3).
#     - This means that the distribution has shorter tails and a lower peak. However, the transformation is more normal.  

sns.distplot(train.saleprice)

print("Skewness before log transformation: ", train['saleprice'].skew())
print("Kurtosis before log transformation: ", train['saleprice'].kurt())
print("-" * 55)
print("Skewness after log transformation: ", np.log(train['saleprice']).skew())
print("Kurtosis after log transformation: ", np.log(train['saleprice']).kurt())

sns.distplot(np.log(train.saleprice))

# Furthermore, we saw similar results from the Q-Q plot. Both graphs indicate that a log transformation of sale price helps to better satisfy the normality assumption required for linear regression. However, We will have to check the results on the error terms after our model is fit.

sm.qqplot(train.saleprice, line='s') 
plt.show()

sm.qqplot(np.log(train.saleprice), line='s') 
plt.show()

# +
y_target = train.saleprice

y_log = np.log(train.saleprice)

train['sale_log'] = y_log
# -

train.dtypes

np.sum(train.isnull())

# ## Missing Values
#
# **LotFrontage**
# - Values appear to be MCAR, meaning, no other features in the data set appear to describe why the value would be missing.
# - Median imputation was chosen  because it was more conservative than the mean value for most neighborhoods.
# - While other imputation methods may provide better results, this helps with interpretability of the model.
#
# **MasVnrArea**
# - The missing values are assumed to represent a lack of masonry veneer type. Although the data does not indicate why these values might be missing, They will be filled with 0 based on the the noted assumption. 

cols = ['alley','bsmtqual','bsmtcond','bsmtexposure','bsmtfintype1',
        'bsmtfintype2','fireplacequ','garagetype','garagefinish',
        'garagequal','garagecond','poolqc','fence','miscfeature']


def fill_na(data, columns):
    '''
    This function fills missing values for the columns where NaN means no feature. 
    "None" was chosen to fill columns that contained NaN values that actually lack a feature.
    If LotFrontage or MasVnrArea are in the data set it fills those with the median and 0, respectively
    
    -------------------------------------------------------------------
    Inputs: data, list of columns.
    
    Returns: Filled NaN values.
    '''
    
    if 'lotfrontage' in data.columns:
        data['lotfrontage'] = data['lotfrontage'].fillna(0)
        print("%s now has %d  NaN" % ('lotfrontage', np.sum(data['lotfrontage'].isnull() )))
        print("-" * 50 + "\n")
        
    if 'masvnrarea' in data.columns:
        data['masvnrarea'] = data['masvnrarea'].fillna(0)
        print("%s now has %d  NaN" % ('masvnrarea', np.sum(data['masvnrarea'].isnull() )))
        print("-" * 50 + "\n")
        
    for col in columns:
        
        data[col] = data[col].fillna("None")

        print("%s now has %d  NaN" % (col, np.sum(data[col].isnull())))
        print("-" * 50 + "\n")



fill_na(train, cols)


# ## EDA
#
# Below are the results from our exploratory data analysis phase. This phase, helped inform the the the remaining portions of the project including feature selection and engineering as well as the model itself. 

def boxplot_(data, column, target, fig_num):
    '''
    This function creates a box plot for the specified columns in a data frame.
    Additionally, it returns the value counts for that column.
    --------------------------------------------------------------------------
    Inputs: 
    
    data frame, 
    column (in string format),
    target / y variable,
    fig_num for displaying each figure
    
    Returns: Plot and value counts. 
    '''
    x = data[column]
    
    y = data[target]
    
    plt.figure(fig_num, figsize=(12, 7))
    
    sns.boxplot(x, y, data = data)
    
    print("Value Counts: \n")
    print(x.value_counts())
    print("-" *50)



def scatterplot_(data, column, target, fig_num):
    '''
    This function creates a scatter plot or linear model plot for the specified columns in a data frame.
    --------------------------------------------------------------------------
    Inputs:
    
    data frame, 
    column (in string format),
    target / y variable,
    fig_num for displaying each figure
    
    Returns: Plot
    '''
    x = data[column]
    
    y = data[target]
    
    plt.figure(fig_num, figsize=(12, 7))
    
    sns.scatterplot(x, y, data = data)



# **Square Feet Measurements**
#
# The data has several features for various home square footage measurements. Often, the total square feet of the home is listed when purchasing or selling a home. Thus, we combined those features to give us the total building square feet. Overall, there is a clear linear trend, however, there are some clear outliers that will have to be examined further in the preprocessing portion. 

train['bldg_sqft'] = train[["totalbsmtsf", "1stflrsf", "2ndflrsf"]].sum(axis = "columns")

scatterplot_(train, 'bldg_sqft', 'sale_log', 1)
plt.xlabel('Home Square Feet')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Home Square Feet')

# To get a more accurate display of usable lot space we also combined porches and decks into a single variable as outside square feet. Additionally, we subracted the first floor square feet and the outside square feet to give us a better estimate of usable lot space.

# +
train['outside_sf'] = train[["wooddecksf", "openporchsf", "3ssnporch", "screenporch", "enclosedporch"]].sum(axis="columns")

train['adj_lot_area'] = train["lotarea"] - train['outside_sf'] - train["1stflrsf"]
# -

# Outside square feet encompases decks, open porches, enclosed porches, three season porches, and screened porches. There appears there may be a slight trend in outside square feet as log sale price increases. We can also see there are several homes that do not have any outside square feet.

scatterplot_(train, 'outside_sf', 'sale_log', 2)
plt.xlabel('Outside Square Feet')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Outside Square Feet')

# Similarly, there appears to be a linear trend with regards to adjusted lot square feet but outliers may have a high leverage on any trend. 

scatterplot_(train, 'adj_lot_area', 'sale_log', 3)
plt.xlabel('Adjusted Lot Square Feet')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Adjusted Lot Square Feet')

boxplot_(train, 'salecondition', 'sale_log', 4)
plt.xlabel('Sale Condition')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Sale Condition')

# **Neighborhood**
#
# Below, we can see varation in log price amond different neighborhoods. The top 3 neighborhoods based upon median home price sales appear to be "NridgHt", "NoRidge", and "StoneBr."

# +
boxplot_(train, 'neighborhood', 'sale_log', 5)

plt.xlabel('Neighborhoods')
plt.ylabel('Log Sale Price')
plt.xticks(rotation = 90)
plt.title('Log Sale Price by Neighborhood')
# -

# **Home Style, Garage, and Rooms**
#
# We can see that most homes are a single family home. Additionally, there does appear to be a trend in higher median log sale prices in homes that have a garage. Below, the size of the garage was measured in car capacity or the number of cars that could fit in the garage. There appears to be a drop in log sale prices after a garage can hold more than 3 cars, however, the price is still above no garage. 

# +
boxplot_(train, 'bldgtype', 'sale_log', 6)

plt.xlabel('Type of Home')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Home Type')

# +
boxplot_(train, 'garagecars', 'sale_log', 7)

plt.xlabel('Size of Garage in Car Capacity')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Garage Size')
# -

# Note that the number of bedrooms above grade does not include any bedrooms in the basement. While there is some variation in log sale price among the number of bedrooms there appears to be less variation in sale price in homes with two or three bedrooms. This could be because they have the most frequent sales. 

# +
boxplot_(train, 'bedroomabvgr', 'sale_log', 8)

plt.xlabel('Number of Bedrooms Above Grade')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Number of Bedrooms')
# -

# The number of bathrooms was another category that was divided. Below we combined the number of full and half baths throughout the home. Overall, it appears that the number of bathrooms a home has may have an impact on log median sale price. However, after 4 bathrooms the median sale price drops. Furthermore, there are very few homes that have 4 or more bathrooms. 

train['total_baths'] = train["fullbath"]  + train["halfbath"]/ 2 + train["bsmtfullbath"] \
                   + train["bsmthalfbath"] / 2

# +
boxplot_(train, 'total_baths', 'sale_log', 9)

plt.xlabel('Number of Bathrooms')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Number of Bathrooms')
# -

# **Home Quality and Features** 
#
# There is a clear pattern between overall home quality and and the log sale price. As the home quality rating increase median log sale price appears to increase also. 

# +
boxplot_(train, 'overallqual', 'sale_log', 10)

plt.xlabel('Overall Home Quality')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Overall Home Quality')
# -

# Below are quaility ratings for fireplaces, kitchens, and the basement finish area. It appears homes with none or fair ratings typically have lower median log sale prices than homes with better rated features. 

# +
boxplot_(train, 'fireplacequ', 'sale_log', 11)

plt.xlabel('Fireplace Quality')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Fireplace Quality')

# +
boxplot_(train, 'kitchenqual', 'sale_log', 12)

plt.xlabel('Kitchen Quality')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Kitchen Quality')

# +
boxplot_(train, 'bsmtfintype1', 'sale_log', 13)

plt.xlabel('Finished Basement Ratings')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Rating of Finished Basement Area')

# +
boxplot_(train, 'bsmtqual', 'sale_log', 14)

plt.xlabel('Basement Height Ratings')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Rating of Basement Height')
# -

# **Electrical, Heating, and Cooling**
#
# Most homes utilize a standard circuit breaker system for their electricity. Homes with better heating and those with central air conditioning typically have higher median log sale prices.

# +
boxplot_(train, 'electrical', 'sale_log', 15)

plt.xlabel('Electrical Systems')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Electrical System')

# +
boxplot_(train, 'heatingqc', 'sale_log', 16)

plt.xlabel('Heating Quality')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Heating Quality')

# +
boxplot_(train, 'centralair', 'sale_log', 17)

plt.xlabel('Central Air (yes / no)')
plt.ylabel('Log Sale Price')
plt.title('Log Sale Price by Central Air')
