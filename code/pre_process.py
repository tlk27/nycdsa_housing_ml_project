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
import janitor as jn
import pandas as pd

# %pylab inline


# + pycharm={"name": "#%%\n", "is_executing": false}
file_path = r"..\data\train.csv"

# column list to import from train csv file based upon initial
# EDA work 
import_list = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Alley','LotShape', 'Neighborhood', 'Condition1', 
              'Condition2', 'BldgType', 'HouseStyle','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 
              'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
              'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'CentralAir', 
              'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
              'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces',
              'FireplaceQu', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch', 'YrSold', 
              'SaleType', 'SaleCondition','Electrical',"HeatingQC","Fireplaces", "FireplaceQu", "BsmtQual", "BsmtFinType1", 
              "BsmtFinType2", 'LotFrontage', 'LotArea', 'GarageCars', 'OverallCond', 'SalePrice'] 

# + pycharm={"name": "#%%\n", "is_executing": false}
# Index will be the ID of the house sale
initial_df = pd.read_csv(filepath_or_buffer=file_path, usecols=import_list, index_col=0)

# convert column names to lowercase and replace spaces with underscores
cleaned_df = jn.clean_names(initial_df)

# + pycharm={"name": "#%%\n", "is_executing": false}
cleaned_df.sample(5)

 # remove outliers based on Pre_process draft based upon Cook's distance > 3 times 
# mean absolute average 
# this step is performed prior to imputation steps  
cleaned_df = cleaned_df.drop([1299, 524], axis="rows") 

# + pycharm={"name": "#%%\n", "is_executing": false}
# Combined square footage metrics for building sf and outside sf (e.g., porch space)
bldg_sqft = cleaned_df[["totalbsmtsf", "1stflrsf", "2ndflrsf"]].sum(axis = "columns")
outside_sf = cleaned_df[["wooddecksf", "openporchsf", "3ssnporch", "screenporch", "enclosedporch"]].sum(axis="columns")
lot_sf = cleaned_df["lotarea"] - cleaned_df['1stflrsf'] - outside_sf

# + pycharm={"name": "#%%\n", "is_executing": false}
# Combining above-basement and basement baths
total_baths = cleaned_df["fullbath"]  + cleaned_df["halfbath"]/ 2 + cleaned_df["bsmtfullbath"] \
                   + cleaned_df["bsmthalfbath"] / 2


# + pycharm={"name": "#%%\n", "is_executing": false}
# Various Dummifications
# 0=1 flat if the building type is a single family home
sgl_famly_hm = cleaned_df["bldgtype"].apply(lambda x: 0 if x == '1Fam' else 1)

# top 3 neighborhoods based upon median home price sales and general spread of prices based upon boxplot
top_3_nbrhd = cleaned_df["neighborhood"].isin(["NridgHt", "NoRidge", "StoneBr"]).map({False: 0, True: 1})

# bottom 5 neighborhoods based upon median home and boxplot inspection
btm_5_nbrhd = cleaned_df["neighborhood"].isin(["MeadowV", "IDOTRR", "BrDale", "OldTown", "Edwards"]).map({False: 0, True: 1})

# Fireplaces that are Excellent, Good or Typical/TA 
good_frplc = cleaned_df["fireplacequ"].isin(["Ex", "Gd", "TA"]).map({False: 0, True: 1})

# remodel age was general found to be more individually correlated with SalePrice than homeage than total home age
# new_home = sold within past 5 years 
homeage = cleaned_df["yrsold"] - cleaned_df["yearbuilt"]
remodelage = cleaned_df["yrsold"] - cleaned_df["yearremodadd"]
newHome = homeage < 5
newHome = newHome.map({False: 0, True: 1})

# + pycharm={"name": "#%%\n", "is_executing": false}
# Various measures where higher amenity ratings that were associated with higher home prices
# these are being combined into a single "positive amentities count" feature 

# 1 Excellent Heating (important for a cold place :-))
excl_heating = cleaned_df["heatingqc"].isin(["Ex"]).map({False: 0, True:1})

# 2 basement has GLQ (Good Living Quarter) in either 
bsmt_gd_lvg = (cleaned_df["bsmtfintype1"].isin(["GLQ"]) | cleaned_df["bsmtfintype2"].isin(["GLQ"])).map({False: 0, True:1})
bsmt_gd_lvg.sum()

# 3 Good, Excellent and "Typical"/TA fireplaces
good_frplc = cleaned_df["fireplacequ"].isin(["Ex", "Gd", "TA"]).map({False: 0, True: 1})

# 4 Good and Excellent Kitchens being combined together 
ktch_groups = cleaned_df["kitchenqual"].map({"TA": "ktch_okay", "Fa": "ktch_okay", "Gd": "ktch_good", "Ex": "kitch_topnotch"})
ktch_dummies = pd.get_dummies(data=ktch_groups).drop("ktch_okay", axis="columns")

# 5 Excellent Basement Quality
excl_bsmt = cleaned_df["bsmtqual"].isin(["Ex"]).map({False: 0, True:1})
excl_bsmt.sum()

good_ament_ct = pd.concat([excl_heating, excl_bsmt, ktch_dummies, bsmt_gd_lvg, good_frplc], axis = "columns").sum(axis="columns")

# + pycharm={"name": "#%%\n", "is_executing": false}
# Various measures where lower amenity ratings were associated with lower higher prices (versus the average/highly rated) 
# these are being combined into a single "negative amentities count" feature

# 1 No fireplace
no_fireplace = cleaned_df["fireplaces"] == 0
no_fireplace = no_fireplace.map({False: 0, True: 1})

# 2 No Central AC
no_central_ac = cleaned_df['centralair'].isin(['N']).map({False: 0, True:1}) 

# Electirical aside from standard circuitbreaker
bad_electrical = cleaned_df['electrical'].isin(['Mix', 'FuseP', 'FuseF', 'FuseA']).map({False: 0, True:1})

bad_ament_ct = pd.concat([no_central_ac, no_fireplace, bad_electrical], axis="columns").sum(axis="columns")


# + pycharm={"name": "#%%\n", "is_executing": false}
# remaining features

#1 credit for having garage space for two or more cars
two_plus_cr_garg = cleaned_df["garagecars"].apply(lambda x: 1 if x >= 2 else 0)

# houses with a 4 or less overall condition showed on average lower sale price then
# houses with an overall condition rating of 5 or higher 
neg_ovrll_cond = cleaned_df["overallcond"].apply(lambda x: 1 if x <= 4 else 0)

# House LotFrontage with NAs filled in as zeros
cleaned_df["lotfrontage"] = cleaned_df["lotfrontage"].fillna(0)

adj_lot_area = cleaned_df["lotarea"] - outside_sf - cleaned_df["1stflrsf"]

abnormal_sale = (cleaned_df["salecondition"] == "Abnorml").map({False: 0, True: 1}).fillna(0)

adj_ovr_qual = cleaned_df["overallqual"].apply(lambda x: 0 if x <=3 else x - 3)

# + pycharm={"name": "#%% \n", "is_executing": false}
# specifying the target and explanatory variables 
list_of_features = [
    bldg_sqft, total_baths, good_ament_ct, btm_5_nbrhd, newHome, neg_ovrll_cond, adj_ovr_qual, adj_lot_area,
    bad_ament_ct, abnormal_sale, outside_sf, two_plus_cr_garg, sgl_famly_hm, top_3_nbrhd, homeage, remodelage,
    cleaned_df[["lotfrontage", "garagecars", "bedroomabvgr", "saleprice"]]
]

features_pls_trgt = pd.concat(list_of_features, axis="columns")

# specifying column name for items inputted as Series
features_pls_trgt.columns = [
    "bldg_sf", "total_baths", "good_ament_ct", "btm_5_nbrhd", "newHome","neg_ovrll_cond", "adj_ovr_qual", 
    "adj_lot_area", "bad_ament_ct", "abnormal_sale", "outside_sf", "two_plus_cr_garg","sgl_famly_hm", "top_3_nbrhd",
    "homeage", "remodelage", "lotfrontage", "garagecars", "bedroomsabvgr", "saleprice",
]

features_pls_trgt.sample(5)

# + pycharm={"name": "#%%\n", "is_executing": false}
# creating field that is the log of sale price
features_pls_trgt["log_saleprice"] = np.log(features_pls_trgt["saleprice"])

file_path_pre_process = r"..\pre_processed_data\pre_processed.csv"

# output X and y to pre_processed CSV file (non-standardized)
features_pls_trgt.to_csv(file_path_pre_process)

# + pycharm={"name": "#%%\n", "is_executing": false}
print(features_pls_trgt.shape)

features_pls_trgt.columns

# + pycharm={"name": "#%%\n"}


