import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

csvr_path='I:\\csvr'
feats=[]
for cpath in sorted(os.listdir(csvr_path)):
    path=csvr_path+'\\'+cpath
    print(path)
    data=pd.read_csv(path)
    value=np.asarray(data['RATIO'])
    value=savgol_filter(value, 5, 1)
    
    value=np.reshape(value, (1,len(value)))
    feats.append(value)

ffeatures=np.concatenate(feats)

ffeatures=ffeatures[:49*4,:]

y=np.asarray(pd.read_csv('I:\\carbon_values.csv')['carbon_percent'])[:49]
Y=[]
for i in range(len(y)):
    a=np.full((4,1),y[i])
    Y.append(a)
    
y=np.concatenate(Y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(ffeatures, y,  test_size=0.20)

from sklearn.cross_decomposition import PLSRegression
pls2 = PLSRegression(n_components=7)
pls2.fit(X_train, y_train)
y_pred_pls = pls2.predict(X_test)


Y_test=pd.DataFrame(data=y_test, columns=['Y_TEST'])
Y_pred_test=pd.DataFrame(data=y_pred_pls, columns=['Y_PRED_TEST'])
Y=pd.concat([Y_test, Y_pred_test], axis =1)
Y.to_csv('C:\\Users\\Chitransh\\Desktop\\results.csv')

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_pred)

rmse=np.sqrt(mean_squared_error(y_test,y_pred_pls))
