cd 01_yf/


python yf.py "^GSPC" 2012-03-31 2022-04-01 1d    #SP 500
python yf.py "^IXIC" 2012-03-31 2022-04-01 1d    #NASDAQ Composite
python yf.py "^RUT"  2012-03-31 2022-04-01 1d    #Russell 2000


python yf_returns.py "^GSPC"
python yf_returns.py "^IXIC"
python yf_returns.py "^RUT"


python yf_returns_agg.py "y_Returns_^GSPC.csv" "y_Returns_^IXIC.csv" "y_Returns_^RUT.csv"


##### See the final stock return file:
##### y.csv