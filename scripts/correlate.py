import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
sbn.set()

pddata = pd.read_csv("../data/all_data.csv", sep=',')

## Get correlation of all number columns
idf = pddata.select_dtypes(include=np.number)
idfcol = np.array(idf.columns.tolist())
icor = np.array(idf.corr())

ycor = np.absolute(icor[0,:])

sort_ind = np.argsort(ycor)
sorty = ycor[sort_ind]
sortcol = idfcol[sort_ind]

topNum = 10     ## Most correlated values with Salaray

print('Top Correlated Number Cols to Salary')
for n in range(2,topNum+2):
    print('COL: %s' % (sortcol[-1*n]))
    print('     Val: %s' % (sorty[-1*n]))
