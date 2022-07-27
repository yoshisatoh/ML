#################### Data Pre-processing: Aggregating Stock Returns  ####################
#
#  (C) 2022, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2022/04/04
# Last Updated: 2022/04/05
#
# Github:
# https://github.com/yoshisatoh/CFA/blob/main/FTI_AR__US_Equity_Indices/03_yf_features/yf_returns_agg.py
#
#
########## Input Data Files
#
#y_Returns_ZT=F.csv
#y_Returns_ZF=F.csv
#y_Returns_ZN=F.csv
#y_Returns_ZB=F.csv
#y_Returns_CL=F.csv
#y_Returns_NG=F.csv
#y_Returns_GC=F.csv
#y_Returns_HG=F.csv
#y_Returns_ZC=F.csv
#y_Returns_ZS=F.csv
#
#
########## Usage Instructions
#
#Run this py script on Windows Command Prompt as follows:
#python yf_returns_agg.py "y_Returns_ZT=F.csv" "y_Returns_ZF=F.csv" "y_Returns_ZN=F.csv" "y_Returns_ZB=F.csv" "y_Returns_CL=F.csv" "y_Returns_NG=F.csv" "y_Returns_GC=F.csv" "y_Returns_HG=F.csv" "y_Returns_ZC=F.csv" "y_Returns_ZS=F.csv"
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

arg_ticker_csv_1     = str(sys.argv[1])       #"y_Returns_ZT=F.csv"
arg_ticker_csv_2     = str(sys.argv[2])       #"y_Returns_^ZF=F.csv"
arg_ticker_csv_3     = str(sys.argv[3])       #"y_Returns_ZZ=F.csv"
arg_ticker_csv_4     = str(sys.argv[4])       #"y_Returns_ZB=F.csv"
arg_ticker_csv_5     = str(sys.argv[5])       #y_Returns_CL=F.csv
arg_ticker_csv_6     = str(sys.argv[6])       #y_Returns_NG=F.csv
arg_ticker_csv_7     = str(sys.argv[7])       #y_Returns_GC=F.csv
arg_ticker_csv_8     = str(sys.argv[8])       #y_Returns_HG=F.csv
arg_ticker_csv_9     = str(sys.argv[9])       #y_Returns_ZC=F.csv
arg_ticker_csv_10     = str(sys.argv[10])       #y_Returns_ZS=F.csv



########## Calculate daily returns

df_EQ_Return_1 = pd.read_csv(arg_ticker_csv_1, index_col='Date')
df_EQ_Return_2 = pd.read_csv(arg_ticker_csv_2, index_col='Date')
df_EQ_Return_3 = pd.read_csv(arg_ticker_csv_3, index_col='Date')
df_EQ_Return_4 = pd.read_csv(arg_ticker_csv_4, index_col='Date')
df_EQ_Return_5 = pd.read_csv(arg_ticker_csv_5, index_col='Date')
df_EQ_Return_6 = pd.read_csv(arg_ticker_csv_6, index_col='Date')
df_EQ_Return_7 = pd.read_csv(arg_ticker_csv_7, index_col='Date')
df_EQ_Return_8 = pd.read_csv(arg_ticker_csv_8, index_col='Date')
df_EQ_Return_9 = pd.read_csv(arg_ticker_csv_9, index_col='Date')
df_EQ_Return_10 = pd.read_csv(arg_ticker_csv_10, index_col='Date')
#print(df_EQ_Return_1)

df_EQ_Returns  = pd.merge(df_EQ_Return_1, df_EQ_Return_2, how='outer', left_index=True, right_index=True)
df_EQ_Returns  = pd.merge(df_EQ_Returns,  df_EQ_Return_3, how='outer', left_index=True, right_index=True)
df_EQ_Returns  = pd.merge(df_EQ_Returns,  df_EQ_Return_4, how='outer', left_index=True, right_index=True)
df_EQ_Returns  = pd.merge(df_EQ_Returns,  df_EQ_Return_5, how='outer', left_index=True, right_index=True)
df_EQ_Returns  = pd.merge(df_EQ_Returns,  df_EQ_Return_6, how='outer', left_index=True, right_index=True)
df_EQ_Returns  = pd.merge(df_EQ_Returns,  df_EQ_Return_7, how='outer', left_index=True, right_index=True)
df_EQ_Returns  = pd.merge(df_EQ_Returns,  df_EQ_Return_8, how='outer', left_index=True, right_index=True)
df_EQ_Returns  = pd.merge(df_EQ_Returns,  df_EQ_Return_9, how='outer', left_index=True, right_index=True)
df_EQ_Returns  = pd.merge(df_EQ_Returns,  df_EQ_Return_10, how='outer', left_index=True, right_index=True)
#print(df_EQ_Returns.dropna())
df_EQ_Returns  = df_EQ_Returns.dropna()
print(df_EQ_Returns)

#df_EQ_Returns.to_csv('y.csv', sep=',', header=True, index=True)
df_EQ_Returns.to_csv('y2.csv', sep=',', header=True, index=True)



