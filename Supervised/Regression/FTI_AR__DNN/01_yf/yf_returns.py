#################### Data Pre-processing: Stock Return Calculation by using Close prices on Yahoo Finance  ####################
#
#  (C) 2022, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2022/04/04
# Last Updated: 2022/04/04
#
# Github:
# https://github.com/yoshisatoh/CFA/blob/main/FTI_US_Equity_Indices/01_yf/yf_returns.py
#
#
########## Input Data Files
#
#^GSPC.csv
#^DJI.csv
#^IXIC.csv
#
#
########## Usage Instructions
#
#Run this py script on Windows Command Prompt as follows:
#python yf_returns.py "^GSPC"
#python yf_returns.py "^DJI"
#python yf_returns.py "^IXIC"
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

arg_ticker     = str(sys.argv[1])       #'^GSPC'
arg_ticker_csv = arg_ticker + '.csv'    #'^GSPC.csv'







########## Calculate daily returns

df_EQ = pd.read_csv(arg_ticker_csv, usecols =['Date', 'Close'])
#print(df_EQ)

df_EQ_Return = df_EQ['Close'].pct_change()
#print(df_EQ_Return)

df_EQ_Return = pd.merge(df_EQ['Date'], df_EQ_Return, how='outer', left_index=True, right_index=True)
#print(df_EQ_Return)

df_EQ_Return.rename(columns={'Close': arg_ticker}, inplace=True)
print(df_EQ_Return)

df_EQ_Return.to_csv('y_Returns_' + arg_ticker_csv, sep=',', header=True, index=False)



