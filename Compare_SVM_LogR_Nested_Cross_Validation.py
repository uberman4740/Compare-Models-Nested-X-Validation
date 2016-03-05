'''
Use nested cross-validation to optimize the hyperparameters an SVM and logistic regression, and compare the performance of the 2 algorithms on he iris dataset.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

#Load the Iris dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.head())

#Assign the features to the numpy arrays.  Use LabelEncoder to transform the class labels from string to integer (Iris-setosa=0, versicolor=1, virginica=2)
from sklearn.preprocessing import LabelEncoder
x = df.loc[:, :3].values  #Assign the 4 features to the numpy array x
y = df.loc[:, 4:].values  #Assign the classification feature to the numpy array y
y = LabelEncoder().fit_transform(y)

#Split into training and test sets using 80:20 split
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)


#Begin nested cross-validation
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

#Create a pipeline with a standard scaler to standardize the data for the SVM
pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1, probability=True))])

#Specify values for the parameter c, which determines how large the margin is for the SVM
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

#List the hyperparameters you want to tune for the SVM
param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
                 {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

#Perform grid search cross validation on the parameters listed in param_grid, using accuracy as the measure of fit and number of folds (CV) = 5
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=5)
scores = cross_val_score(gs, x_train, y_train, scoring='accuracy', cv=5)
print('Optimized SVM Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#Create a pipeline with a standard scaler to standardize the data for the logistic regression
pipe_lr = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(random_state=1))])

#Specify values for the parameter c, the regularization parameter for logistic regression
#param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
#Line above commented out so that the same regularization parameters are tested for LR as with the SVM

#List the hyperparameters you want to tune for the logistic regression
param_grid = [{'clf__C': param_range}]

#Perform grid search cross validation on the parameters listed in param_grid, using accuracy as the measure of fit and number of folds (CV) = 5
gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='accuracy', cv=5)
scores = cross_val_score(gs, x_train, y_train, scoring='accuracy', cv=5)
print('Optimized Logistic Regression Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


#The SVM seems to be the more accurate model in this case, but the accuracies of both models are within one another's margin of error so it's impossible to say for certain


#Begin plot of ROC curves - Note that the ROC curve will show 1 class vs all
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp

#Specify a stratified k fold cross validation with 3 folds
cv = StratifiedKFold(y_train, n_folds=3, random_state=1)

#Note: trp = true positive rate, fpr = false positive rate
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

#Plot ROC curve for logistic regression - Note that the ROC curve will show classification of versicolor vs all others
for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(x_train[train], y_train[train]).predict_proba(x_train[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr) #Calculates the area under the curve (AUC)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)  #Calculates the average area under the curve (AUC)
plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':', color='black', label='perfect performance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic - Logistic Regression')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

#Reset variables for the next ROC curve, Note:  tpr = true positive rate, fpr = false positive rate
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

#Plot ROC curve for SVM - Note that the ROC curve will show classification of versicolor vs all others
for i, (train, test) in enumerate(cv):
    probas = pipe_svc.fit(x_train[train], y_train[train]).predict_proba(x_train[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr) #Calculates the area under the curve (AUC)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)  #Calculates the average area under the curve (AUC)
plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':', color='black', label='perfect performance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic - SVM')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
