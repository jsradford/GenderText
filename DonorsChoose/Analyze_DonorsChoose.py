

'''
This code takes the data created from Make_Blogger_Data and assess the accuracy of classifiers used on
the data.  
'''
from sklearn import svm
from sklearn import cross_validation as cv
from sklearn import grid_search
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn import ensemble
import csv

import numpy as np

import FeatureSelection
import Vectorize
import Classifiers
import DrawSamples
import FeatureExtractors
import TopicModels
reload(Vectorize)
reload(FeatureSelection)
reload(Classifiers)
reload(DrawSamples)
reload(FeatureExtractors)
reload(TopicModels)

def ImportMeta(num=-1):
    '''
    ExactTeacherStrata.csv = 'projectid, male, subclass' #id, gender, matched category
    '''
    match_file='/home/jsradford/GenderText/DonorsChoose/ExactTeacherStrata.csv'
    genDict={'0':'female','1':'male'}
    with open(match_file,'rb') as f:
        matches=csv.reader(f,delimiter=',')
        for i,row in enumerate(matches):
            if i==0:
                continue
            if num==-1:
                metadata[row[0]]=[genDict[row[1]],row[2]]
            else:
                if i<num:
                    metadata[row[0]]=[genDict[row[1]],row[2]]
                else:
                    break
    return metadata

def ImportFeatureData(filename,num=-1):
    '''
    filenames are all '.txt'
    all files are [[header,word1,word2],[id,weight1,weight2]
    '''
    Data=[]
    with open(filename,'rb') as f:
        for i,line in enumerate(f.readlines()):
            if num==-1:
                if i>0:
                    Data.append([float(item) for item in line.strip().split(',')])
            else:
                if i<num and i>0:
                    Data.append([float(item) for item in line.strip().split(',')])
                elif i>num:
                    break
    return Data

def ImportCSVFeatureData(filename,num=-1):
    '''
    filenames are all '.txt'
    all files are [[header,word1,word2],[id,weight1,weight2]
    '''
    Data=[]
    with open(filename,'rb') as f:
        data=csv.reader(f)
        for i,line in enumerate(data):
            if num==-1:
                if i>0:
                    Data.append(line)
            else:
                if i<num and i>0:
                    Data.append(line)
                elif i>num:
                    break
    return Data

def Write_Scores(preds,head,fname):
    '''
    Takes an dictionary predictions from CrossValidate and saves to file 'fname'
    Input
        head=list('id','NamePreds')
    returns
    guess=np.array(N,1) of log likelihood ratios
    '''
    print 'writing scores to file ', fname
    with open(fname,'wb') as f:
        writer=csv.writer(f,delimiter=',')
        writer.writerow(head)
        for idx, score in preds.iteritems():
            writer.writerow([idx,score])
    return 

def Analyze_Raw(metadata,cl,**CVargs):
    '''
    mnb max's out at 69/70% accurate at 3,000 (or 600 training) texts.  Does not increase in accuracy after that.
    svm: grid search showed ideal is linear kernal with C=1,10, or 100; also max's out at 74% accurate for 3,000 (goes to 76 at 10,000)
    
    '''
    print 'running Raw analysis'
    
    #metadata=ImportMeta(-1)
    filename='/home/jsradford/GenderText/DonorsChoose/Data/Blogger_Raw_Scores.txt'
    data=ImportFeatureData(filename,-1)
    print 'drawing samples'
    vec=np.array([line[1:] for line in data])
    labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
    IDX=np.array([str(int(line[0])) for line in data])
    
    Preds=CrossValidate(vec,labels,IDX,cl,**CVargs)
    print 'standardizing scores'
    preds={}
    for k,score in Preds.iteritems():
        if np.inf in score:
            original=len(score)
            x = list(np.array(score)[np.logical_not(np.isinf(score))])
            try:max(x)
            except:continue
            x.append(max(x))
            preds[k]=np.mean(x)
        elif -1*np.inf in score:
            original=len(score)
            x = list(np.array(score)[np.logical_not(np.isinf(score))])
            try:min(x)
            except:continue
            x.append(min(x))
            preds[k]=np.mean(x)
        else:
            preds[k]=np.mean(score)
    m=np.mean(preds.values())
    sd=np.std(preds.values())

    for k,score in preds.iteritems():
        preds[k]=(score-m)/sd
    fname='Raw_Preds.csv'
    Write_Scores(preds,['id','raw_score'],fname)
    return

def Analyze_Individual(metadata,cl,**CVargs):
    '''
    grid search shows C>=1 is optimal
    accuracy is unrelated to sample size (remains 84-89% throughout)
    '''
    print 'INDIVIDUAL ANALYSIS'
    #metadata=ImportMeta(-1)
    filename='/home/jsradford/GenderText/DonorsChoose/Data/Blogger_Individual_Scores.txt'
    data=ImportFeatureData(filename,-1)
    vec=np.array([line[2:] for line in data if line[1]!=1.0])   #exclude cases where sex is never mentioned
    labels=np.array([metadata[str(int(line[0]))][0] for line in data if line[1]!=1.0])# if 'age' not in line])
    IDX=np.array([str(int(line[0])) for line in data if line[1]!=1.0])

    Preds=CrossValidate(vec,labels,IDX,cl,**CVargs)
    print 'standardizing scores'
    preds={}
    for k,score in Preds.iteritems():
        if np.inf in score:
            original=len(score)
            x = list(np.array(score)[np.logical_not(np.isinf(score))])
            try:max(x)
            except:continue
            x.append(max(x))
            preds[k]=np.mean(x)
        elif -1*np.inf in score:
            original=len(score)
            x = list(np.array(score)[np.logical_not(np.isinf(score))])
            try:min(x)
            except:continue
            x.append(min(x))
            preds[k]=np.mean(x)
        else:
            preds[k]=np.mean(score)
    m=np.mean(preds.values())
    sd=np.std(preds.values())
    for k,score in preds.iteritems():
        preds[k]=(score-m)/sd
    fname='Individual_Preds.csv'
    Write_Scores(preds,['id','indiv_score'],fname)
    
    return

def Analyze_Raw_Topic_Scores(metadata,cl,**CVargs):
    '''
    grid_search shows C>=1 is ideal. remains 71% from 500 through 7000
    remains at 71% at sample sizes from 500 through 10000.
    '''
    print 'RAW TOPIC ANALYSIS'

    #metadata=ImportMeta(-1)
    filename='/home/jsradford/GenderText/DonorsChoose/Data/Raw_Topic_Scores.csv'
    data=ImportCSVFeatureData(filename,-1)
    print 'drawing samples'
    vec=np.array([[float(l) for l in line[1:]] for line in data])   #exclude cases where sex is unknown
    labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
    IDX=np.array([str(int(line[0])) for line in data])

    Preds=CrossValidate(vec,labels,IDX,cl,**CVargs)
    print 'standardizing scores'
    preds={}
    for k,score in Preds.iteritems():
        if np.inf in score:
            original=len(score)
            x = list(np.array(score)[np.logical_not(np.isinf(score))])
            try:max(x)
            except:continue
            x.append(max(x))
            preds[k]=np.mean(x)
        elif -1*np.inf in score:
            original=len(score)
            x = list(np.array(score)[np.logical_not(np.isinf(score))])
            try:min(x)
            except:continue
            x.append(min(x))
            preds[k]=np.mean(x)
        else:
            preds[k]=np.mean(score)
    m=np.mean(preds.values())
    sd=np.std(preds.values())

    for k,score in preds.iteritems():
        preds[k]=(score-m)/sd
    fname='Raw_Topic_Preds.csv'
    Write_Scores(preds,['id','rawTopic_score'],fname)
    
    return


def Analyze_Nonword_Scores(metadata,cl,**CVargs):
    '''
    
    check rows 1535 and 15349 for inf data. Should no longer have to recode 8 and 12 (herndanV and LnM)
    
    '''
    print 'NONWORD ANALYSIS'

    #metadata=ImportMeta(-1)
    filename='/home/jsradford/GenderText/DonorsChoose/Data/Blogger_Nonword_Scores.csv'
    data=ImportCSVFeatureData(filename,-1)
    print 'drawing samples'
    vec=np.array([[float(l) for l in line[1:]] for line in data])   #exclude cases where sex is unknown
    #vec[:,8]=vec[:,8]*-1    #herndanV is always neg (changed in Make_Blogger_Data now)
    #vec[:,12]=vec[:,12]*-1   #LnM is always neg (changed in Make_Blogger_Data now)
    labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
    IDX=np.array([str(int(line[0])) for line in data])
    vec=np.delete(vec,[1535,15349],axis=0)
    labels=np.delete(labels,[1535,15349],axis=0)
    IDX=np.delete(IDX,[1535,15349],axis=0)

    Preds=CrossValidate(vec,labels,IDX,cl,**CVargs)
    print 'standardizing scores'
    preds={}
    for k,score in Preds.iteritems():
        if np.inf in score:
            original=len(score)
            x = list(np.array(score)[np.logical_not(np.isinf(score))])
            try:max(x)
            except:continue
            x.append(max(x))
            preds[k]=np.mean(x)
        elif -1*np.inf in score:
            original=len(score)
            x = list(np.array(score)[np.logical_not(np.isinf(score))])
            try:min(x)
            except:continue
            x.append(min(x))
            preds[k]=np.mean(x)
        else:
            preds[k]=np.mean(score)
    m=np.mean(preds.values())
    sd=np.std(preds.values())
    for k,score in preds.iteritems():
        preds[k]=(score-m)/sd
    fname='Nonwords_Preds.csv'
    Write_Scores(preds,['id','nonword_score'],fname)
    
    return


def Analyze_Structure_Scores(metadata,cl,**CVargs):
    '''
    SVM - should be linear and C =10. Accuracy maxes out around 3,000 at ~62% but doesn't grow much from 500
    MNB - sample size appears to be unrelated to accuracy. Hovers at 58-62% throughout.
    '''
    print 'STRUCTURE ANALYSIS'

    #metadata=ImportMeta(-1)
    #Data/DonorsChoose_Structure_'+cat+'_Scores.csv','wb')
    path='/home/jsradford/GenderText/DonorsChoose/Data/'
    PREDS=dict(zip(metadata.keys(),[[] for i in metadata.keys()]))

    for cat in set([line[1] for line in metadata.values()]):
        if cat=='category':
            continue
        #if cat=='Student' or cat=='indUnk':    #For big categories, use a different test conditions
        #    args={'n_iter':20, 'test_size':.9,'random_state':0}
        else:
            args=CVargs.copy()
        print 'RUNNINING ', cat, ' STRUCTURE SCORES'
        f='DonorsChoose_Structure_'+cat+'_Scores.csv'
        data=ImportCSVFeatureData(path+f,-1)
        vec=np.array([[float(l) for l in line[1:]] for line in data])   #exclude cases where sex is unknown
        labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
        IDX=np.array([str(int(line[0])) for line in data])
        Preds=CrossValidate(vec,labels,IDX,cl,**args)
        preds={}
        for k,score in Preds.iteritems():
            if np.inf in score:
                original=len(score)
                x = list(np.array(score)[np.logical_not(np.isinf(score))])
                try:max(x)
                except:continue
                x.append(max(x))
                preds[k]=np.mean(x)
            elif -1*np.inf in score:
                original=len(score)
                x = list(np.array(score)[np.logical_not(np.isinf(score))])
                try:min(x)
                except:continue
                x.append(min(x))
                preds[k]=np.mean(x)
            else:
                preds[k]=np.mean(score)
        m=np.mean(preds.values())
        sd=np.std(preds.values())

        for k,score in preds.iteritems():
            PREDS[k].append((score-m)/sd)
    #This uses a bagging model to score masculine or feminine
    for k, scores in PREDS.iteritems():
        PREDS[k]=np.mean(scores)
    
    fname='Structure_Preds.csv'
    Write_Scores(PREDS,['id','struct_score'],fname)
    
    return

def Analyze_Behavior_Scores(metadata,cl,**CVargs):
    '''
    grid_search across all seems to agree that C==10,000 for linear or rbf is optimal
    Sample size doesn't matter because number of texts in an area max out at 6,000
    '''
    print 'BEHAVIOR ANALYSIS'
    
    #num=1000
    #metadata=ImportMeta(-1)
    
    path='/home/jsradford/GenderText/DonorsChoose/Data/'
    PREDS={}
    for cat in set([line[1] for line in metadata.values()]):
        if cat=='category':
            continue
        #if cat=='Student' or cat=='indUnk':    #For big categories, use a different test conditions
        #    args={'n_iter':20, 'test_size':.9,'random_state':0}
        else:
            args=CVargs.copy()
        print 'RUNNINING ', cat, ' BEHAVIOR SCORES'
        f='Blogger_Behavior_'+cat+'_Scores.csv'
        data=ImportCSVFeatureData(path+f,-1)
        vec=np.array([[float(l) for l in line[1:]] for line in data])   #exclude cases where sex is unknown
        labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
        IDX=np.array([str(int(line[0])) for line in data])
        Preds=CrossValidate(vec,labels,IDX,cl,**args)
        preds={}
        for k,score in Preds.iteritems():
            if np.inf in score:
                original=len(score)
                x = list(np.array(score)[np.logical_not(np.isinf(score))])
                try:max(x)
                except:continue
                x.append(max(x))
                preds[k]=np.mean(x)
            elif -1*np.inf in score:
                original=len(score)
                x = list(np.array(score)[np.logical_not(np.isinf(score))])
                try:min(x)
                except:continue
                x.append(min(x))
                preds[k]=np.mean(x)
            else:
                preds[k]=np.mean(score)
        m=np.mean(preds.values())
        sd=np.std(preds.values())

        for k,score in preds.iteritems():
            preds[k]=(score-m)/sd
        PREDS.update(preds)
    fname='Behavior_Preds.csv'
    Write_Scores(PREDS,['id','behavior_score'],fname)
        
    return


def Analyze_SubTopic_Scores(metadata,cl,**CVargs):
    '''
    grid_search reveals little variation. Linear 100 or linear 10 seem best, but not a huge effect
    Number of cases doesn't matter for MNB because most are small sample sizes (largest is 6,000).
    '''
    print 'SUBTOPIC ANALYSIS'
    
    #num=1000
    #metadata=ImportMeta(-1)
    path='/home/jsradford/GenderText/DonorsChoose/Data/'
    PREDS={}
    for cat in set([line[1] for line in metadata.values()]):
        if cat=='category':
            continue
        if cat=='Student' or cat=='indUnk':
            args={'n_iter':20, 'test_size':.9,'random_state':0}
        else:
            args=CVargs.copy()
        print 'RUNNINING ', cat, ' SUBTOPIC SCORES'
        f='Blogger_'+cat+'_Topic_Scores.csv'
        data=ImportCSVFeatureData(path+f,-1)
        vec=np.array([[float(l) for l in line[1:]] for line in data])   #exclude cases where sex is unknown
        labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
        IDX=np.array([str(int(line[0])) for line in data])
        Preds=CrossValidate(vec,labels,IDX,cl,**args)
        print 'standardizing scores'
        preds={}
        for k,score in Preds.iteritems():
            if np.inf in score:
                original=len(score)
                x = list(np.array(score)[np.logical_not(np.isinf(score))])
                try:max(x)
                except:continue
                x.append(max(x))
                preds[k]=np.mean(x)
            elif -1*np.inf in score:
                original=len(score)
                x = list(np.array(score)[np.logical_not(np.isinf(score))])
                try:min(x)
                except:continue
                x.append(min(x))
                preds[k]=np.mean(x)
            else:
                preds[k]=np.mean(score)
        m=np.mean(preds.values())
        sd=np.std(preds.values())
        for k,score in preds.iteritems():
            preds[k]=(score-m)/sd
        PREDS.update(preds)
    fname='SubTopic_Preds.csv'
    Write_Scores(PREDS,['id','subtopic_score'],fname)
        
    
    return


def hybridTrial(metadata):
    '''
    This code takes two above feature sets and tests whether they change their collective and individual predictability
    Raw * Raw Topics = ? * .72 = .71 (No change in nb score)
    Subtopics * raw topics = .65 * .72 = .69
    '''
    print 'import raw topic scores'
    filename='/home/jsradford/GenderText/DonorsChoose/Data/Raw_Topic_Scores.csv'
    data=ImportCSVFeatureData(filename,-1)
    print 'drawing samples'
    vec=np.array([[float(l) for l in line[1:]] for line in data])   #exclude cases where sex is unknown
    labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
    IDX=np.array([str(int(line[0])) for line in data])
    print 'CV for RAW TOPICS'
    CVargs={'n_iter':3, 'test_size':.9,'random_state':0}
    cl=mnb()
        #Preds=CrossValidate(vec,labels,IDX,cl,**CVargs)
        
    print 'importing subtopic scores'
    path='/home/jsradford/GenderText/DonorsChoose/Data/'
    preds={}
    Data=[]
    for cat in set([line[1] for line in metadata.values()]):
        if cat=='category':
            continue
        if cat=='Student' or cat=='indUnk':
            args={'n_iter':20, 'test_size':.9,'random_state':0}
        else:
            args=CVargs.copy()
        print 'RUNNINING ', cat, ' SUBTOPIC SCORES'
        f='Blogger_'+cat+'_Topic_Scores.csv'
        data=ImportCSVFeatureData(path+f,-1)
        Data.append(data)
        #for line in data:
        #    for idx in IDX:
        #        if str(int(line[0]))==idx:
        #            rvec.append(line)
        #            break
        #vec=np.array([[float(l) for l in line[1:]] for line in data])   #exclude cases where sex is unknown
        #labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
        #IDX=np.array([str(int(line[0])) for line in data])
    print 'resorting cases to align with labels'
    rvec=[[] for i in IDX]
        #rlabels=[]
    
    for data in Data:
        for i,idx in enumerate(IDX):
            if idx in [str(int(line[0])) for line in data]:
                for line in data:
                    if idx==str(int(line[0])):
                        rvec[i]+=line[1:]
                        continue
                    #rvec.append(line[1:])
            else:
                rvec[i]+=[0 for i in data[0][1:]]
                
    #used to align RAW
    #for idx in IDX:
    #    if str(int(line[0]))==idx:
    #        rvec.append(line[1:])
    #        break
                    #rlabels.append(meta-data[str(int(idx))][0])
        
    rvec=np.append(vec,np.array(rvec),axis=1)
    
        
    print 'crossvalidate testing COMBINATION'
    CVargs={'n_iter':3, 'test_size':.9,'random_state':0}
    cl=mnb()
    Preds=CrossValidate(vec,labels,IDX,cl,**CVargs)
    
    CVargs={'n_iter':3, 'test_size':.9,'random_state':0}
    cl=mnb()
    cl=ensemble.AdaBoostClassifier(n_estimators=10)
    Preds=CrossValidate(vec,labels,IDX,cl,**CVargs)
    
    return


metadata=ImportMeta(2000)
##hybridTrial(metadata)
#rawkwargs={'n_iter':20, 'test_size':.9,'random_state':0}
##cl=mnb()
##cl=svm.SVC(C=10,kernel='linear',probability=True)
#cl=ensemble.RandomForestClassifier(n_estimators=10)
#Analyze_Raw(metadata,cl,**rawkwargs)
###
#indargs={'n_iter':20, 'test_size':.9,'random_state':0}
##cl=mnb()
##cl=svm.SVC(C=10,kernel='linear',probability=True)
#cl=ensemble.RandomForestClassifier(n_estimators=10)
#Analyze_Individual(metadata,cl,**indargs)
##
#strargs={'n_iter':20, 'test_size':.9,'random_state':0}
##cl=mnb()
##cl=svm.SVC(C=10,kernel='linear',probability=True)
#cl=ensemble.RandomForestClassifier(n_estimators=10)
#Analyze_Structure_Scores(metadata,cl,**strargs)     #Structure scores are accurate at around 60% using 50 words per cat.Will rerun to see if this imporves with more features.
##
#rtargs={'n_iter':20, 'test_size':.9,'random_state':0}
##cl=mnb()
##cl=svm.SVC(C=10,kernel='linear',probability=True)
#cl=ensemble.RandomForestClassifier(n_estimators=10)
#Analyze_Raw_Topic_Scores(metadata,cl,**rtargs)
###
nwargs={'n_iter':20, 'test_size':.9,'random_state':0}
#cl=mnb()
#cl=svm.SVC(C=10,kernel='linear',probability=True)
cl=ensemble.RandomForestClassifier(n_estimators=10)
Analyze_Nonword_Scores(metadata,cl,**nwargs)
###
#bargs={'n_iter':100, 'test_size':.1,'random_state':0}
##cl=mnb()
##cl=svm.SVC(C=10000,kernel='linear',probability=True)
#cl=ensemble.RandomForestClassifier(n_estimators=10)
#Analyze_Behavior_Scores(metadata,cl,**bargs)
#
#stargs={'n_iter':100, 'test_size':.1,'random_state':0}
##cl=mnb()
##cl=svm.SVC(C=100,kernel='linear',probability=True)
#cl=ensemble.RandomForestClassifier(n_estimators=10)
#Analyze_SubTopic_Scores(metadata,cl,**stargs)

#


#################################################
#
#   OLD CODE FOR TESTING SAMPLE SIZE AND GRID_SEARCH SVM
#
#################################################
#print 'importing data'
#num=1000
#metadata=ImportMeta(-1)
#for num in [500]:#,1000,3000,5000,7000]:
#    filename='/home/jsradford/GenderText/DonorsChoose/Data/Blogger_Nonword_Scores.csv'
#    data=ImportCSVFeatureData(filename,num)
#    print 'drawing samples'
#vec=np.array([[float(l) for l in line[2:]] for line in data if line[1]!=1.0])   #exclude cases where sex is unknown
#labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
#if num==500:
#    print vec.shape
#grids=[{'kernel':['linear','rbf'], 'C':[.01,.1,1,10,100,1000]}]
#grids=[{'kernel':['linear'], 'C':[1]}]
#grid=grid_search.GridSearchCV(svm.SVC(),grids,cv=2)
#grid.fit(vec,labels)
#print("Best parameters set for RAW TOPIC found on development set:")
#print()
#print(grid.best_params_)
#print()
#print("Grid scores on development set:")
#print()
#for params, mean_score, scores in grid.grid_scores_:
#    print("\t %0.3f (+/-%0.03f) for %r"
#          % (mean_score, scores.std() * 2, params))
#    trainT,trainL,testT,testL=trainTestSample(vec, labels, samp=.5)
    #print 'running classifier for RAW TOPIC'
    #classifier=Classify(trainT,trainL,clf='mnb')
    #print '\t FOR NUM TEXTS = ', num*.5
    #print '\t accuracy is ', classifier.score(testT,testL)

