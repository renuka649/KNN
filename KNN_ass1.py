# -*- coding: utf-8 -*-
"""
Created on Tue feb 3 22:53:46 2024

@author: renuk
"""

'''
problem ststement no : 1
1.	A glass manufacturing plant uses different earth elements to 
design new glass materials based on customer requirements. 
For that, they would like to automate the process of classification
as it’s a tedious job to manually classify them. Help the company 
achieve its objective by correctly classifying the glass type based 
on the other features using KNN algorithm.
Business Constraints:
Accuracy: The classification model should achieve high accuracy in predicting glass types based on features. This ensures that the automated classification process is reliable and aligns with customer requirements.

Interpretability: While accuracy is important, the model's decisions should be interpretable to stakeholders. They need to understand how the features contribute to the classification of glass types.

Scalability: The classification model should be scalable to handle a large volume of glass samples and features. It should efficiently process data as the manufacturing plant scales its operations.

Resource Efficiency: The model should be computationally efficient to minimize processing time and resource utilization. This ensures that the classification process remains cost-effective for the manufacturing plant.

Maximize:
Maximize the accuracy of the KNN classification model.
 Higher accuracy ensures that the model can correctly 
 classify glass types based on features, thereby reducing
 errors and improving customer satisfaction.
 
 Minimize:
Minimize the cost associated with misclassifications,
 particularly false positives and false negatives.
 Misclassifying glass types can lead to manufacturing 
 errors, product defects, and customer dissatisfaction.
 Minimizing misclassification costs helps mitigate 
 potential losses and maintains product quality.
'''


import pandas as pd
import numpy as np 
glass=pd.read_csv("glass.csv")
glass

glass.describe()
glass.info()

glass['Type'].value_counts()

#normalization
def norm(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

#let us apply normlization function to the dataset
glass_norm = norm(glass.iloc[:,0:9])

#now let us take X as input & Y as output
X = np.array(glass_norm.iloc[:,:])
Y = np.array(glass['Type'])

#split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
#to avoid the unbalancing of data during splitting the concept of 
#statified sampling is used

#now build the KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
pred

#now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,Y_test))
pd.crosstab(pred,Y_test)


#let us try to select the correct value of k
acc = []
#running the KNN algorithm for k=3 to 50 in step of 2
#k's value is selected as odd
for i in range(3,50,2):
    #declare model
    n = KNeighborsClassifier(n_neighbors=i)
    n.fit(X_train,Y_train)
    train_acc = np.mean(n.predict(X_train) == Y_train)
    test_acc = np.mean(n.predict(X_test) == Y_test)
    acc.append([train_acc,test_acc])

#lets plot the graph of train_acc and test_acc
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],'ro-')
plt.plot(np.arange(3,50,2),[i[1] for i in acc],'bo-')
#there are valiues like 3,5,7,9 where the accuracy is good

#lets try for K=3
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
accuracy_score(pred,Y_test)
#0.5581395348837209
pd.crosstab(pred,Y_test)





























