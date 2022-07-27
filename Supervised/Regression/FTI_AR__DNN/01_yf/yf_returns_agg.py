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
# https://github.com/yoshisatoh/CFA/blob/main/FTI_US_Equity_Indices/01_yf/yf_returns_agg.py
#
#
########## Input Data Files
#
#y_Returns_^GSPC.csv
#y_Returns_^DJI.csv
#y_Returns_^IXIC.csv
#
#
########## Usage Instructions
#
#Run this py script on Windows Command Prompt as follows:
#python yf_returns_agg.py "y_Returns_^GSPC.csv" "y_Returns_^DJI.csv" "y_Returns_^IXIC.csv"
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

arg_ticker_csv_1     = str(sys.argv[1])       #"y_Returns_^GSPC.csv"
arg_ticker_csv_2     = str(sys.argv[2])       #"y_Returns_^DJI.csv"
arg_ticker_csv_3     = str(sys.argv[3])       #"y_Returns_^IXIC.csv"





########## Calculate daily returns

df_EQ_Return_1 = pd.read_csv(arg_ticker_csv_1, index_col='Date')
df_EQ_Return_2 = pd.read_csv(arg_ticker_csv_2, index_col='Date')
df_EQ_Return_3 = pd.read_csv(arg_ticker_csv_3, index_col='Date')
#print(df_EQ_Return_1)

df_EQ_Returns  = pd.merge(df_EQ_Return_1, df_EQ_Return_2, how='outer', left_index=True, right_index=True)
df_EQ_Returns  = pd.merge(df_EQ_Returns,  df_EQ_Return_3, how='outer', left_index=True, right_index=True)
#print(df_EQ_Returns.dropna())
df_EQ_Returns  = df_EQ_Returns.dropna()
print(df_EQ_Returns)

df_EQ_Returns.to_csv('y.csv', sep=',', header=True, index=True)



