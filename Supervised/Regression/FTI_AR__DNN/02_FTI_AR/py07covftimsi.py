##### import libraries
import pandas as pd
import numpy as np
import csv

from py02param import *


y = pd.read_csv('y.csv', index_col=0)
ymmu = pd.read_csv('ymmu.csv', index_col=0)


cov = y.rolling(window=dma).cov()
#print(cov)

cov = cov.dropna(how="any")
#print(cov)

cov.to_csv("cov.csv")

#len(cov.columns)
#3

#cov[0:3]
#                     NTTR      NXTR       IXF
#Trade Date                                 
#2009-08-20 NTTR  0.000452  0.000392  0.000470
#           NXTR  0.000392  0.000402  0.000473
#           IXF   0.000470  0.000473  0.000786

#np.linalg.inv(cov[0:3])
#array([[ 14403.24297589, -13429.93270314,   -521.2216743 ],
#       [-13429.93270314,  21077.19070866,  -4666.48714028],
#       [  -521.2216743 ,  -4666.48714028,   4394.71065572]])

#print(np.linalg.inv(cov[0:3]))
#[[ 14403.24297589 -13429.93270314   -521.2216743 ]
# [-13429.93270314  21077.19070866  -4666.48714028]
# [  -521.2216743   -4666.48714028   4394.71065572]]

#np.linalg.inv(cov[0:3]).shape
#(3, 3)

# np.array(ymmu[0:1])
#array([[0.00788763, 0.00722348, 0.00716573]])

# np.array(ymmu[0:1]).shape
#(1, 3)

# np.array(ymmu[1:2])
#array([[0.01311559, 0.01776113, 0.01831083]])

# np.array(ymmu[0:1]).T
#array([[0.00788763],
#       [0.00722348],
#       [0.00716573]])

# np.array(ymmu[0:1]).T.shape
#(3, 1)

#np.array(ymmu[0:1]) * np.linalg.inv(cov[0:3]) * np.array(ymmu[0:1]).T
#(1, 3)(3, 3)(3, 1)
#(1, 3)(3, 1)
#(1)

#np.array(ymmu[0:1]).dot(np.linalg.inv(cov[0:3]))
# np.array(ymmu[0:1]).dot(np.linalg.inv(cov[0:3])).dot(np.array(ymmu[0:1]).T)
#array([[0.14915166]])

# ymmu
#                NTTR      NXTR       IXF
#Trade Date                             
#2009-08-20  0.007888  0.007223  0.007166
#2009-08-21  0.013116  0.017761  0.018311
#2009-08-24 -0.008065 -0.002483 -0.016052
#2009-08-25  0.001166  0.008367  0.004676
#2009-08-26  0.002793 -0.002977  0.002221
#...              ...       ...       ...
#2020-04-28 -0.014821 -0.010789  0.007708
#2020-04-29  0.040584  0.018247  0.026535
#2020-04-30 -0.020283 -0.015237 -0.026131
#2020-05-01 -0.045328 -0.020033 -0.033132
#2020-05-05 -0.000719 -0.000370 -0.000112
#
#[2701 rows x 3 columns]

# cov['2009-08-20':'2009-08-20']
#                     NTTR      NXTR       IXF
#Trade Date                                 
#2009-08-20 NTTR  0.000452  0.000392  0.000470
#           NXTR  0.000392  0.000402  0.000473
#           IXF   0.000470  0.000473  0.000786

# cov[0:3]
#                     NTTR      NXTR       IXF
#Trade Date                                 
#2009-08-20 NTTR  0.000452  0.000392  0.000470
#           NXTR  0.000392  0.000402  0.000473
#           IXF   0.000470  0.000473  0.000786
#
# cov[3:6]
#                     NTTR      NXTR       IXF
#Trade Date                                 
#2009-08-21 NTTR  0.000452  0.000392  0.000470
#           NXTR  0.000392  0.000402  0.000474
#           IXF   0.000470  0.000474  0.000786

# cov.index[0][0]
#'2009-08-20'
# cov.index[3][0]
#'2009-08-21'


# cov.index[0][0]
#'2009-08-20'
# np.array(ymmu[0:1]).dot(np.linalg.inv(cov[0:3])).dot(np.array(ymmu[0:1]).T)
#array([[0.14915166]])

# cov.index[3][0]
#'2009-08-21'
# np.array(ymmu[1:2]).dot(np.linalg.inv(cov[3:6])).dot(np.array(ymmu[1:2]).T)
#array([[1.05672594]])

# int(len(cov)/len(cov.columns))
#8103 / 3
#2701


#with open('FTI.csv', 'w') as f:
with open('FTI.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    #writer.writerow(["Trade Date", "FTI"])
    #writer.writerow(["Day", "FTI"])
    writer.writerow(["Date", "FTI"])


#tmp = [cov.index[0][0], float(np.array(ymmu[0:1]).dot(np.linalg.inv(cov[0:3])).dot(np.array(ymmu[0:1]).T))]
#tmp = [cov.index[3][0], float(np.array(ymmu[1:2]).dot(np.linalg.inv(cov[3:6])).dot(np.array(ymmu[1:2]).T))]
#tmp = [cov.index[6][0], float(np.array(ymmu[2:3]).dot(np.linalg.inv(cov[6:9])).dot(np.array(ymmu[2:3]).T))]


#for i in range(3):
#    print(i)
#
# 0
# 1
# 2


for i in range(int(len(cov)/len(cov.columns))):
    tmp = [cov.index[i*len(cov.columns)][0], float(np.array(ymmu[i:i+1]).dot(np.linalg.inv(cov[i*len(cov.columns):(i+1)*len(cov.columns)])).dot(np.array(ymmu[i:i+1]).T))]
    #
    #with open('FTI.csv', 'a') as f:
    with open('FTI.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(tmp)



##### Magnitude Surprise Index (MSI)

#with open('MSI.csv', 'w') as f2:
with open('MSI.csv', 'w', newline="") as f2:
    writer = csv.writer(f2)
    #writer.writerow(["Trade Date", "MSI"])
    #writer.writerow(["Day", "MSI"])
    writer.writerow(["Date", "MSI"])

for i in range(int(len(cov)/len(cov.columns))):
    tmp2 = [cov.index[i*len(cov.columns)][0], float(np.array(ymmu[i:i+1]).dot(np.linalg.inv(np.triu(np.tril(cov[i*len(cov.columns):(i+1)*len(cov.columns)])))).dot(np.array(ymmu[i:i+1]).T))]
    #
    #with open('MSI.csv', 'a') as f2:
    with open('MSI.csv', 'a', newline="") as f2:
        writer = csv.writer(f2)
        writer.writerow(tmp2)
