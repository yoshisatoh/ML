#################### Data Pre-processing: Stock Prices on Yahoo Finance ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/28
# Last Updated: 2022/04/04
#
# Github:
# https://github.com/yoshisatoh/CFA/blob/main/FTI_AR__US_Equity_Indices/03_yf_features/yf.py
#
########## Input Data File(s)
#
#
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python yf.py "^GSPC" 2012-04-01 2022-03-31 1d    #SP 500
#python yf.py "^DJI"  2012-04-01 2022-03-31 1d    #Dow Jones Industrial Average
#python yf.py "^IXIC" 2012-04-01 2022-03-31 1d    #NASDAQ Composite
#python yf.py TSLA 2020-10-28 2021-10-27 1d       #Tesla
#
#Generally,
#python yf.py (arg_ticker: a ticker on Yahoo Finance) (arg_start) (arg_end) (arg_interval)
#
#arg_ticker: a stock ticker e.g., TSLA
#
#arg_start, arg_end: yyyy-mm-dd
#
#arg_interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (optional, default is '1d')
#
#
########## Output Data File(s)
#
#ticker_history.csv
#
#
########## References
#
#yfinance 0.1.64
#https://pypi.org/project/yfinance/
#
#Yahoo! Finance - Futures
#https://finance.yahoo.com/commodities
#
####################




########## install Python libraries (before running this script)
#
#pip install yfinance  --upgrade
#pip install yfinance  --U
#python -m pip install yfinance 
#python -m pip install yfinance ==0.17.1
#
#If any of the above does not work in your environment, then try:
#pip install --upgrade yfinance  --trusted-host pypi.org --trusted-host files.pythonhosted.org



########## import Python libraries

import sys

import pandas as pd
#import matplotlib.pyplot as plt

import yfinance as yf




########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #yf.py

arg_ticker   = str(sys.argv[1])    #'TSLA'

arg_start    = str(sys.argv[2])    #"2020-10-28",
arg_end      = str(sys.argv[3])    #"2021-10-27",
arg_interval = str(sys.argv[4])    #'1d'    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (optional, default is '1d')




########## Getting stock prices

ticker = yf.Ticker(arg_ticker)

ticker_history = ticker.history(
    start    = arg_start,
    end      = arg_end,
    interval = arg_interval
).reset_index()

#ticker_history.to_csv('ticker_history.csv', header=True, index=False)
ticker_history.to_csv(arg_ticker + '.csv', header=True, index=False)
