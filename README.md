# SFE
#### stepwise feature elimination in nonlinear SVR model

The python script for the stepwise feature elimination in nonlinear SVR model,more details can be seen in our paper of "Predicting the  Enzymatic Hydrolysis Half-lives of New Chemicals Using Support Vector Regression Models Based on Stepwise Feature Elimination' in the journal of  Molecular Informatics" (https://www.ncbi.nlm.nih.gov/pubmed/28627805)

#########################################################################################
Author:shenwanxiang

Emails:shenwanxiang@tsinghua.org.cn

Any bugs is welcomed

---
### The Stepwise Feature Elimination(SFE) version 1:

the feature was eliminated one by one based on the scoring of the target function.The target function f (x) was defined by the average R^2 or average mean squared error (MSE) of internal k-fold cross validation or test set validatin. 

In each cycle, the eliminated score of each feature was calculated after transversal deletion, 
finally the feature with minimal elimination MSE was really deleted.

Total N features were gradually eliminated until the last one feature(no stopping rules), 
and total N*(N-1)/2 SVR models were built during the elimination, 
the parameters of each model were optimized by grid search technique using 'gridregression.py'.
----

### Additional file information:
###
* SFE.py: 
this is the python script for backward stepwise feature selection.
Note: you can set your number of features to eliminate in this algorithm(des_num),
you can also define your own target function

The parameter of des_num is very important: if the number of total features is 100, however the des_num is set as 80 by yourself,then only the top 80 columns(features) will be eliminated, namely the last 20 features is fixed by yourself(if you think these 20 features are very important features). This can be very useful for reduction of the calculating cost, and the global optimization of the feature combination is easly achieved 


###
* gridregression.py: 
SFE.py needs the 'gridregression.py' to find best parameters in non-linear SVR model.
so make sure the gridregression.py is configured correctly


###
* svm-train and svm_predict:
Compiled from the libsvm 3.21 package,you should download libsvm and make sure svm-train and svm_predict
can be used correctly


###
* master file including train.data and test.data, which are the testing file for this SFE.py script
the data's dilimiter is '\t'(TAB key), the last column is data labels,others are features
