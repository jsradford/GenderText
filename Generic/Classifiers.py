import numpy as np


def GridSearch(X,Y,clfs,grids):
    '''
    takes some sample data np.array(X) and labels np.array(Y) and runs grid search over
    the specified classifiers
    grids is a list of gridsearch dictionaries
    '''
    from sklearn import svm
    from sklearn.naive_bayes import MultinomialNB as mnb
    from sklearn import neighbors as knn
    from sklearn import grid_search
    if clfs==[]:
        clfs=[knn.KNeighborsClassifier(),svm.SVC()]
        clfs=[svm.SVC()]
    if grids==[]:
        grids=[{'n_neighbors':[5,10,round(len(Y)/10.0)]},
            {'kernel':['linear', 'rbf'], 'C':[.001,.01,1, 10,100]}]
    
    for i,clf in enumerate(clfs):
        grid=grid_search.GridSearchCV(clf,grids[i],cv=5)
        grid.fit(X,Y)
        print("Best parameters set found on development set:")
        print()
        print(grid.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in grid.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        
    
    return


def Classify(trainT,trainL,clf='knn'):
    '''Code to train and test classifiers.  type can be 'knn' 'nb' or 'svm'
    returns the fit matrix #a dictionary of {twitterID: likelihood ratio}'''
    from sklearn import svm
    from sklearn.naive_bayes import MultinomialNB as mnb
    from sklearn import neighbors
    print 'Running Classifier '+ clf
    if clf=='knn':
        cl = neighbors.KNeighborsClassifier()
        cl.fit(trainT,trainL)
    if clf=='svm':
        cl=svm.SVC(C=100,gamma=.1,probability=True)
        cl.fit(trainT,trainL)
    if clf=='mnb':
        cl=mnb()
        cl.fit(trainT,trainL)
    return cl

def EnsembleClassScores(X,Y,ntrees=10,strategy=''):
    '''
    X, Y are vector, labels
    strategy can be 'Random Forests' , 'AdaBoosting' , 'Gradient Boosting'
    ntress is the number of trees to estimate
    returns the trained classifier
    '''
    print 'Running Classifier '+ strategy
    import sklearn.ensemble as ensemble
    if strategy=='Random Forests':
        clf=ensemble.RandomForestClassifier(n_estimators=ntrees)
    if strategy=='AdaBoosting':
        clf=ensemble.AdaBoostClassifier(n_estimators=ntrees)
    if strategy=='Gradient Boosting':
        clf=ensemble.GradientBoostingClassifier(n_estimators=ntrees, learning_rate=1.0,max_depth=1)
    clf=clf.fit(X.toarray(),Y)
    return clf

def LassoRegressScores(X,Y,a=1):
    ''' a is the lasso alpha value, see: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    '''
    clf=lm.Lasso(alpha=a)
    clf=clf.fit(X,Y)
    return clf

def CrossValidate(X,Y,IDX,cl,**kwargs):
    '''
    Input
        X - is array of (N,K) features
        Y - is array of (N,1) labels
        IDX - is the (N,1) array of keys
        clf is a string for the classifier method: 'svm','mnb','knn',etc.
        kwargs is for ShiffleSplit and should be {'n_iter':5, 'test_size':0.80,'random_state':0} for example
    Returns:
        Predictions= dict(Key: log likelihood of label[0])
    '''
    print 'running cross-val'
    from sklearn import cross_validation as cv
    from sklearn import svm
    from sklearn.naive_bayes import MultinomialNB as mnb
    from sklearn import neighbors as knn
    #predcition=cv.cross_val_predict(clf,X,Y,**kwargs)   #this _predict function only exists in an updated version of sklearn.
    Res={}
    print kwargs
    splits=cv.ShuffleSplit(X.shape[0],**kwargs)# n_iter=5, test_size=0.80,random_state=0)
    print 'running ', len(splits), ' splits in cross-validation'
    for trainidx,testidx in splits:
        if len(set(Y[trainidx]))==1:
            continue
        trainL=Y[trainidx]
        trainT=X[trainidx]
        testL=Y[testidx]
        testT=X[testidx]
        testIDX=IDX[testidx]
        if cl=='knn':
            cl = neighbors.KNeighborsClassifier()
        if cl=='svm':
            cl=svm.SVC(C=1,kernel='linear',probability=True)
        if cl=='mnb':
            cl=mnb()
        cl.fit(trainT,trainL)
        print 'accuracy of nth fold is ', cl.score(testT,testL)
        preds=cl.predict_proba(testT)
        if 0 in preds:
            for i,p in enumerate(preds):
                if p[0]==0:
                    preds[i][0]=.01
                    preds[i][1]=.99
                if p[1]==0:
                    preds[i][0]=.99
                    preds[i][1]=.01
        female=[np.log(p[0]/p[1]) for p in preds]
        res=dict(zip(testIDX,female))
        for k,v in res.iteritems():        
            if k in Res:
                Res[k].append(v)
            else:
                Res[k]=[v]

    return Res








