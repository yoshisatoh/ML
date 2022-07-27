#Run this script on your MacOS Terminal (or your Windows Command Prompt)


cd ../04_ML

cp ../01_yf/y.csv .

cp ../02_FTI_AR/FTIdma.csv .

cp ../02_FTI_AR/AR.csv .

cp ../03_yf_features/y2.csv .


python data_agg.py "FTIdma.csv" "AR.csv" "y2.csv"


python dlregrwgts1.py 50 l1l2 0.0001    #FTIDma
# See: test_targets_pred1.csv


python dlregrwgts2.py 50 l1l2 0.0001    #AR
# See: test_targets_pred2.csv