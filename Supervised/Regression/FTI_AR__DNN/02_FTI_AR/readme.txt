#Run this script on your MacOS Terminal (or your Windows Command Prompt)


cd ../02_FTI_AR

cp ../01_yf/y.csv y.csv


#You need to save the following py files (e.g., py01datay.py) on "YOUR_WORKING_DIRECTORY" before running this script.


#If you have your own y.csv file, then this part can be commented out.
#python py00generatedatay.py


##########


python


import py01datay

#You can set parameters in py02param.py
#
#import py02param
from py02param import *
#print(dma)

from py03mu import *

from py04yminusmu import *

from py05sigma import *

from py06corr import *

from py07covftimsi import *

from py08ftima import *

from py09ftipctrank import *

from py10msima import *

from py11csi import *

from py12csima import *

from py13csipctrank import *

from py20ar import *


quit()


python py99results.py

# See results.csv



##########

### References
#
#[1] Financial Turbulence
#Kritzman, M., and Y. Li. “Skulls, Financial Turbulence, and Risk Management.” Financial Analysts Journal, Vol. 66, No. 5 (2010), pp. 30-41.
#http://www.cfapubs.org/doi/abs/10.2469/faj.v66.n5.3
#
#[2] Correlation Surprise
#Kinlaw, Will, and David Turkington. 2014. “Correlation Surprise.”, Journal of Asset Management, Vol. 14, 6(2014), pp. 385-399.
#https://link.springer.com/article/10.1057/jam.2013.27
#
#[3] Absorption Ratio
#Kritzman, Mark, Yuanzhen Li, Sébastien Page, and Roberto Rigobon. 2011. “Principal Components as a Measure of Systemic Risk.” The Journal of Portfolio Management, Vol. 37, No. 4 (2011), pp. 112-126. 
#https://doi.org/10.3905/jpm.2011.37.4.112

