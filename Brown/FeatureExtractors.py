import sklearn.ensemble as ensemble
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn import neighbors
from sklearn.feature_selection import RFE   #recursive feature elimination
from sklearn import feature_selection as fs
import numpy as np

#################   Classifier Results
def ClassifierResults(clf):
    
    print 'results would go here'
    return

#################   Class feature rankers
def LinearTopWords(classifier,n=20):
    tops=np.argsort(classifier.coef_[0])[-n:]
    #words=[vectorizer.get_feature_names()[i] for i in tops] 
    return tops

def RFETopWords(X,Y,n=20,clf=''):
    if clf=='knn':
        cl = neighbors.KNeighborsClassifier()
    if clf=='svm':
        cl=svm.LinearSVC()
    if clf=='mnb':
        cl=mnb()
    selector=RFE(cl,n,step=.05)
    selector=selector.fit(X,Y)
    tops=np.argsort(selector.support_)[-n:]
    #words=[vectorizer.get_feature_names()[i] for i in tops] 
    return selector,tops

def UnivariateSelection(X,Y,num):
    selector=fs.SelectKBest(fs.f_classif, k=num)
    selector=selector.fit(X, Y)
    tops=np.argsort(selector.pvalues_)[-num:]
    #words=[vectorizer.get_feature_names()[i] for i in tops] 
    return selector,tops

def EnsembleClassScores(clf,num):
    tops=np.argsort(clf.feature_importances_)[-num:]
    return clf,tops

################  Continuous regression feature selectors

def UnivariateRegression(X,Y,num):
    selector=fs.SelectKBest(fs.f_regression,k=num)
    selector=selector.fit(X, Y)
    tops=np.argsort(selector.pvalues_)[-num:]
    #words=[vectorizer.get_feature_names()[i] for i in tops]
    return selector,tops

def LassoRegressScores(X,Y,a,num):
    clf=lm.Lasso(alpha=a)
    clf=clf.fit(X,Y)
    tops=np.argsort(clf.coef_)[-num:]
    return clf,tops

def EnsembleRegressionScores(X,Y,num, ntrees, strategy=''):
    if strategy=='Random Forests':
        clf=ensemble.RandomForestRegressor(n_estimators=ntrees)
    if strategy=='AdaBoosting':
        clf=ensemble.AdaBoostRegressor(n_estimators=ntrees)
    if strategy=='Gradient Boosting':
        clf=ensemble.GradientBoostingRegressor(n_estimators=ntrees, learning_rate=1.0,max_depth=1,loss='ls')
    clf=clf.fit(X,Y)
    tops=np.argsort(clf.feature_importances_)[-num:]
    return clf,tops
