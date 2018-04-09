"""
baseline -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
Basic test using the provided data matrix and the logestic Regression
"""

import sys

sys.path.append('../')

from Tools.Utils import X_train_matrix_0,X_train_matrix_1,X_train_matrix_2,\
X_train_0,X_train_1,X_train_2,Y_train_0,Y_train_1,Y_train_2,X_test_0\
,X_test_1,X_test_2,X_test_matrix_0,X_test_matrix_1,X_test_matrix_2

import numpy as np
from Classifiers.LogesticRegression import LogisticRegression
from Tools.Utils import accuracy_score
from Tools.Utils import train_test_split

X_train_full = np.concatenate((X_train_matrix_0,X_train_matrix_1,X_train_matrix_2))
Y_train_full = np.concatenate((Y_train_0,Y_train_1,Y_train_2)).reshape(-1)
X_test_full= np.concatenate((X_test_matrix_0,X_test_matrix_1,X_test_matrix_2))

X_train, X_val,y_train,y_val = train_test_split(X_train_full,Y_train_full,test_size=0.1)
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train,y_train,lambda_regularisation=0.01)

y_pred = logistic_reg.predict_class(X_test_full).reshape(-1)

y_train_pred = logistic_reg.predict_class(X_val).reshape(-1)

print('Precision =',accuracy_score(y_val.reshape(-1),y_train_pred))



with open("submission.csv", 'w') as f:
    f.write(',Bound\n')
    for i in range(len(y_pred)):
        f.write(str(i)+','+str(y_pred[i])+'\n')