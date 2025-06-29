import pandas as pd
import numpy as np
import copy
import xgboost as xgb
import time

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lm
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

#Part I
def data_process(X,category,stand_mean = None,stand_std = None):
    # X : pd.DataFrame, No PID, No price
    row,col = X.shape
    column_names = X.columns
    X_new = np.empty([row,1])
    if category == False:
        stand_mean = {}
        stand_std = {}
        column_names_new = []
        category_dict = {}
        for column_name in column_names:
            if X[column_name].dtype != 'object':
                column_temp = X[column_name].to_numpy().reshape(row,-1)
                scaler = StandardScaler()
                scaler.fit(column_temp)
                column_temp = scaler.transform(column_temp)
                stand_mean[column_name] = scaler.mean_
                stand_std[column_name] = scaler.scale_
                # print(column_temp.shape)
                column_temp = pre_process(column_temp)
                X_new = np.concatenate((X_new,column_temp),axis = 1)
                column_names_new = column_names_new + [column_name,]


            else:
                encoder = OneHotEncoder(sparse=False,categories='auto')
                # print(X[column_name].shape)
                encoded_data = encoder.fit_transform(X[[column_name]])
                X_new = np.concatenate((X_new,encoded_data),axis = 1)
                column_names_new = column_names_new + list(encoder.categories_[0])
                category_dict[column_name] = encoder.categories_
        # X_new_array = np.array(X_new)
        X_new = np.delete(X_new, 0, axis=1)
        X_new = np.nan_to_num(X_new, nan=0)
        return X_new,column_names_new,category_dict,stand_mean,stand_std

    else:
        for column_name in column_names:
            if X[column_name].dtype != 'object':
                column_temp = X[column_name].to_numpy().reshape(row,-1)
                scaler = StandardScaler(with_mean=stand_mean[column_name],with_std=stand_std[column_name])
                scaler.fit(column_temp)
                column_temp = scaler.transform(column_temp)
                column_temp = pre_process(column_temp)
                X_new = np.concatenate((X_new,column_temp),axis = 1)
                # column_names_new = column_names_new + column_name
            else:
                encoder = OneHotEncoder(sparse=False,categories=category[column_name],handle_unknown='ignore')
                encoded_data = encoder.fit_transform(X[[column_name]])
                X_new = np.concatenate((X_new,encoded_data),axis = 1)
                # column_names_new = column_names_new + list(encoder.categories_[0])
                # category_dict[column_name] = encoder.categories_
        X_new = np.delete(X_new, 0, axis=1)
        X_new = np.nan_to_num(X_new, nan=0)
        return X_new
    
def pre_process(data):
    sigma = 1.9
    data_new = copy.deepcopy(data)
    index_max = np.where(data > sigma)
    index_min = np.where(data < -sigma)
    data_new[index_max] = sigma
    data_new[index_min] = -sigma
    return data_new

folder_address = './'
name_list = ['/']
train_data = []
train_PID = []
train_price = []
train_data_name = []
train_stand_mean = []
train_stand_std = []
# train_data_object = []
test_data_object = []
train_data_category = []
test_data = []
test_PID = []
# test_price = []
train_name = '/train.csv'
test_name = '/test.csv'
# test_y_name = '/test_y.csv'

# train_drop_name = ['PID','Sale_Price', 'Longitude','Latitude','Year_Built','Year_Remod_Add', 'Full_Bath', 'Half_Bath', 'Total_Bsmt_SF']
# test_drop_name = ['PID', 'Longitude','Latitude','Year_Built','Year_Remod_Add', 'Full_Bath', 'Half_Bath', 'Total_Bsmt_SF']
train_drop_name = ['PID','Sale_Price','Garage_Yr_Blt', 'Garage_Area', 'Garage_Cond', 'Total_Bsmt_SF', 'TotRms_AbvGrd',
                    'BsmtFin_SF_1','First_Flr_SF','Second_Flr_SF', 'Bedroom_AbvGr', 'Full_Bath', 'Half_Bath', 'Bsmt_Full_Bath',
                    'Bsmt_Half_Bath', 'Open_Porch_SF', 'Enclosed_Porch', 'Three_season_porch', 'Screen_Porch',
                    'Street', 'Utilities', 'Land_Slope', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 
                    'Misc_Feature','Low_Qual_Fin_SF', 'Pool_Area', 'Misc_Val', 'Longitude', 'Latitude']
test_drop_name = ['PID','Garage_Yr_Blt', 'Garage_Area', 'Garage_Cond', 'Total_Bsmt_SF', 'TotRms_AbvGrd',
                    'BsmtFin_SF_1','First_Flr_SF','Second_Flr_SF', 'Bedroom_AbvGr', 'Full_Bath', 'Half_Bath', 'Bsmt_Full_Bath',
                    'Bsmt_Half_Bath', 'Open_Porch_SF', 'Enclosed_Porch', 'Three_season_porch', 'Screen_Porch',
                    'Street', 'Utilities', 'Land_Slope', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 
                    'Misc_Feature','Low_Qual_Fin_SF', 'Pool_Area', 'Misc_Val', 'Longitude', 'Latitude']

# train_drop_name = ['PID','Sale_Price']
# test_drop_name = ['PID',]

for index, fold_name in enumerate(name_list):
    # print(index)
    #read train data
    train_file_name = folder_address+fold_name+train_name
    data = pd.read_csv(train_file_name)
    PID = data['PID']
    train_PID.append(PID)
    Price = data['Sale_Price']
    train_price.append(np.log(Price))
    X = data.drop(train_drop_name,axis = 1)

    #total bath room
    total_bathroom = data['Full_Bath'] + data['Half_Bath']
    age = data['Year_Remod_Add']-data['Year_Built']
    total_area = data['Total_Bsmt_SF']+data['Gr_Liv_Area']
    
    X['total_bathroom'] = total_bathroom
    X['age'] = age
    X['total_area'] = total_area

    category = False
    X_new,column_names_new,category_dict,stand_mean,stand_std = data_process(X,category)
    train_data.append(X_new)
    train_stand_mean.append(stand_mean)
    train_stand_std.append(stand_std)
    train_data_category.append(category_dict)
    train_data_name.append(column_names_new)

    #read test file
    test_file_name = folder_address+fold_name+test_name
    data = pd.read_csv(test_file_name)
    PID = data['PID'].to_numpy()
    test_PID.append(PID)
    X = data.drop(test_drop_name,axis = 1)

    total_bathroom = data['Full_Bath'] + data['Half_Bath']
    age = data['Year_Remod_Add']-data['Year_Built']
    total_area = data['Total_Bsmt_SF']+data['Gr_Liv_Area']
    
    X['total_bathroom'] = total_bathroom
    X['age'] = age
    X['total_area'] = total_area


    category = category_dict
    X_new = data_process(X,category,train_stand_mean[index],train_stand_std[index])
    test_data.append(X_new)

    # test_price_name = folder_address+fold_name+test_y_name
    # data = pd.read_csv(test_price_name)
    # test_price.append(np.log(data.drop(['PID'],axis = 1).to_numpy()))

# RMSE = np.zeros(10)
Predict_Price = []
Ridge_start = time.time()
for folder_index in range(len(name_list)):

    alphas = np.logspace(-6, 6, 13)  # Range of alpha values to test
    ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)

    # num_folds = 200
    # lasso_cv = LassoCV(alphas=None, cv=KFold(n_splits=num_folds))

    # Fit the model to your data
    ridge_cv.fit(train_data[folder_index],train_price[folder_index])

    # Get the optimal alpha value selected by LassoCV
    best_alpha = ridge_cv.alpha_
    cv_score = np.mean(ridge_cv.cv_values_, axis=0)

    # Fit the Lasso model with the optimal alpha to the entire dataset
    final_ridge_model = Ridge(alpha=best_alpha)
    final_ridge_model.fit(train_data[folder_index],train_price[folder_index])
    y_pred = final_ridge_model.predict(test_data[folder_index])
    Predict_Price.append(np.exp(y_pred))
    # rmse = np.sqrt(mean_squared_error(y_pred,test_price[folder_index]))
    # RMSE[folder_index] = rmse

Ridge_end = time.time()
Ridge_running_time = Ridge_end-Ridge_start
print(f"Ridge took {Ridge_running_time:.3f} seconds to run")

#data print
file_name = "mysubmission1.txt"
PID_all = [item for sublist in test_PID for item in sublist]
Price = [item for sublist in Predict_Price for item in sublist]
# Calculate the maximum length of each column name
max_col1_len = max(len("PID_all"), max(len(str(pid)) for pid in PID_all))
max_col2_len = max(len("Price"), max(len(f"{price:.2f}") for price in Price))

# Open the file for writing
with open(file_name, "w") as file:
    # Write the column names as the first row with padding
    file.write(f"PID,".ljust(max_col1_len + 2))
    file.write(f"Sale_Price".ljust(max_col2_len + 2))
    file.write("\n")
    
    # Write data from both lists in two columns with padding
    for pid, price in zip(PID_all, Price):
        row_data = f"{pid},".ljust(max_col1_len + 2) + f"{price:.2f}".ljust(max_col2_len + 2) + "\n"
        file.write(row_data)

#Part II
# Data processing

def data_process(X,category):
    # X : pd.DataFrame, No PID, No price
    row,col = X.shape
    column_names = X.columns
    X_new = np.empty([row,1])
    if category == False:
        column_names_new = []
        category_dict = {}
        for column_name in column_names:
            if X[column_name].dtype != 'object':
                column_temp = X[column_name].to_numpy().reshape(row,-1)
                X_new = np.concatenate((X_new,column_temp),axis = 1)
                column_names_new = column_names_new + [column_name,]
            else:
                encoder = OneHotEncoder(sparse=False,categories='auto')
                # print(X[column_name].shape)
                encoded_data = encoder.fit_transform(X[[column_name]])
                X_new = np.concatenate((X_new,encoded_data),axis = 1)
                column_names_new = column_names_new + list(encoder.categories_[0])
                category_dict[column_name] = encoder.categories_
        # X_new_array = np.array(X_new)
        X_new = np.delete(X_new, 0, axis=1)
        X_new = np.nan_to_num(X_new, nan=0)
        return X_new,column_names_new,category_dict

    else:
        for column_name in column_names:
            if X[column_name].dtype != 'object':
                column_temp = X[column_name].to_numpy().reshape(row,-1)
                X_new = np.concatenate((X_new,column_temp),axis = 1)
                # column_names_new = column_names_new + column_name
            else:
                encoder = OneHotEncoder(sparse=False,categories=category[column_name],handle_unknown='ignore')
                encoded_data = encoder.fit_transform(X[[column_name]])
                X_new = np.concatenate((X_new,encoded_data),axis = 1)
                # column_names_new = column_names_new + list(encoder.categories_[0])
                # category_dict[column_name] = encoder.categories_
        X_new = np.delete(X_new, 0, axis=1)
        X_new = np.nan_to_num(X_new, nan=0)
        return X_new

# Data input


folder_address = './'
name_list = ['/']
train_data = []
train_PID = []
train_price = []
train_data_name = []
train_data_category = []
test_data = []
test_PID = []
# true_price = []
# true_price_PID = []
train_name = '/train.csv'
test_name = '/test.csv'
# testy_name = '/test_y.csv'

for index, fold_name in enumerate(name_list):
    # print(index)
    #read training data
    train_file_name = folder_address + fold_name + train_name
    data = pd.read_csv(train_file_name)
    PID = data['PID'] # Save the PID in the dataset into a seperate column
    train_PID.append(PID) 
    Price = np.array(data['Sale_Price'])
    Price_log = np.log(Price)
    train_price.append(Price_log) # Response = price
    X = data.drop(['PID','Sale_Price'],axis = 1) # All predictors 
    category = False
    X_new,column_names_new,category_dict = data_process(X,category)
    train_data.append(X_new)
    train_data_category.append(category_dict)
    train_data_name.append(column_names_new)

    #read test file
    test_file_name = folder_address+fold_name+test_name
    data = pd.read_csv(test_file_name)
    PID = data['PID']
    test_PID.append(PID)
    X = data.drop(['PID'],axis = 1)
    category = category_dict
    X_new = data_process(X,category)
    test_data.append(X_new)

    #read test_y file
    # true_price_file = folder_address + fold_name + testy_name
    # data = pd.read_csv(true_price_file)
    # PID = data['PID']
    # true_price_PID.append(PID)
    # Price = np.array(data['Sale_Price'])
    # Price_log = np.log(Price)
    # true_price.append(Price_log)

# Attempt two: XGboost model for the price prediction task

np.random.seed(2417)

rmse_xgb = []
Xgb_start = time.time()
xgb_model = XGBRegressor(n_estimators = 500,eta = 0.05, subsample= 0.5)
Predict_Xgb = []
for index in range(len(name_list)):
    xgb_model.fit(train_data[index], train_price[index])
    y_pred_xgb = xgb_model.predict(test_data[index])
    Predict_Xgb.append(np.exp(y_pred_xgb))
    # rmse_index = np.sqrt(mean_squared_error(true_price[index], y_pred_xgb))
    # rmse_xgb.append(rmse_index)
    # print(rmse_index)
Xgb_end = time.time()
Xgb_running_time = Xgb_end-Xgb_start
print(f"Xgb took {Xgb_running_time:.3f} seconds to run")
#data print
file_name = "mysubmission2.txt"
PID_all = [item for sublist in test_PID for item in sublist]
Price = [item for sublist in Predict_Xgb for item in sublist]
# Calculate the maximum length of each column name
max_col1_len = max(len("PID_all"), max(len(str(pid)) for pid in PID_all))
max_col2_len = max(len("Price"), max(len(f"{price:.2f}") for price in Price))

# Open the file for writing
with open(file_name, "w") as file:
    # Write the column names as the first row with padding
    file.write(f"PID,".ljust(max_col1_len + 2))
    file.write(f"Sale_Price".ljust(max_col2_len + 2))
    file.write("\n")
    
    # Write data from both lists in two columns with padding
    for pid, price in zip(PID_all, Price):
        row_data = f"{pid},".ljust(max_col1_len + 2) + f"{price:.2f}".ljust(max_col2_len + 2) + "\n"
        file.write(row_data)

                
