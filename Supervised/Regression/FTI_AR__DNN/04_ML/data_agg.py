#################### Data Pre-processing: Aggregating Stock Returns  ####################
#
#  (C) 2022, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2022/04/04
# Last Updated: 2022/04/04
#
# Github:
# https://github.com/yoshisatoh/CFA/blob/main/FTI_AR__US_Equity_Indices/04_ML/data_agg.py
#
#
########## Input Data Files
#
#FTIdma.csv
#AR.csv
#y2.csv
#
#
########## Usage Instructions
#
#Run this py script on Windows Command Prompt as follows:
#python data_agg.py "FTIdma.csv" "AR.csv" "y2.csv"
#
#
########## References
#
#
#
####################################################################################################




########## install Python libraries
#
# pip on your Terminal on MacOS (or Command Prompt on Windows) might not work.
#pip install pandas
#
# If that's the case, then try:
#pip install --upgrade pandas --trusted-host pypi.org --trusted-host files.pythonhosted.org
#
# If it's successful, then you can repeat the same command for other libraries (e.g., numpy).
#
#
########## import Python libraries
#
import sys
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt




########## arguments
for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

arg_ticker_csv_1     = str(sys.argv[1])       #FTIdma.csv
arg_ticker_csv_2     = str(sys.argv[2])       #AR.csv
arg_ticker_csv_3     = str(sys.argv[3])       #y2.csv




########## Calculate daily returns

df_1 = pd.read_csv(arg_ticker_csv_1, index_col='Date')
df_2 = pd.read_csv(arg_ticker_csv_2, index_col='Date')
df_3 = pd.read_csv(arg_ticker_csv_3, index_col='Date')
#print(df_EQ_Return_1)

df_3_cumprod = (1 + df_3).cumprod()
df_3_cumprod.to_csv('df_3_cumprod.csv', sep=',', header=True, index=True)


df  = pd.merge(df_1, df_2, how='outer', left_index=True, right_index=True)
df  = pd.merge(df,  df_3_cumprod, how='outer', left_index=True, right_index=True)
#print(df_EQ_Returns.dropna())
df  = df.dropna()
print(df)

df.to_csv('df.csv', sep=',', header=True, index=True)




########## 2 types of Target & Features

df_target_1 = df['FTI']
df_target_1.to_csv('df_target_1.csv', sep=',', header=True, index=True)
#
df_target_1.to_csv('df_target_1_train_targets_raw.csv', sep=',', header=True, index=False)
df_target_1.to_csv('df_target_1_test_targets_raw.csv',  sep=',', header=True, index=False)


df_target_2 = df['AR']
df_target_2.to_csv('df_target_2.csv', sep=',', header=True, index=True)
#
df_target_2.to_csv('df_target_2_train_targets_raw.csv', sep=',', header=True, index=False)
df_target_2.to_csv('df_target_2_test_targets_raw.csv',  sep=',', header=True, index=False)



df_features = df.drop('FTI', axis=1)
df_features = df_features.drop('AR', axis=1)
df_features.to_csv('df_features.csv', sep=',', header=True, index=True)
#
df_features.to_csv('df_features_train_data_raw.csv', sep=',', header=True, index=False)
df_features.to_csv('df_features_test_data_raw.csv',  sep=',', header=True, index=False)
