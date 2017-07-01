

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
import Classifiers
reload(Classifiers)


def ImportMeta(num=-1):
    '''
    metadata=['id','sex','age','category','sign']
    '''
    import csv
    metadata={}
    print 'opening meta-data file'
    with open('Blogger/Data/bloggerMeta.csv','rb') as f:
        data=csv.reader(f)
        for i,row in enumerate(data):
            if num==-1:
                metadata[row[0]]=row[1:]
            else:
                if i<num:
                    metadata[row[0]]=row[1:]
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


def importArray(name):
    import cPickle
    scorefile='Blogger/Data/Blogger_'+name+'_Scores.pkl'
    idfile='Blogger/Data/Blogger_'+name+'_Ids.pkl'
    wordfile='Blogger/Data/Blogger_'+name+'_Names.pkl'
    with open(scorefile,'rb') as f:
        scores=cPickle.load(f)
    with open(idfile,'rb') as f:
        ids=cPickle.load(f)
    with open(wordfile,'rb') as f:
        words=cPickle.load(f)
    return scores,ids,words

def Write_Scores(preds,head,fname):
    '''
    Takes an dictionary predictions from Classifiers.CrossValidate and saves to file 'fname'
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

def Analyze_Raw(metadata,cl,outfile,**CVargs):
    '''
    mnb max's out at 69/70% accurate at 3,000 (or 600 training) texts.  Does not increase in accuracy after that.
    svm: grid search showed ideal is linear kernal with C=1,10, or 100; also max's out at 74% accurate for 3,000 (goes to 76 at 10,000)
    
    '''
    print 'running Raw analysis'
    
    #metadata=ImportMeta(-1)
    filename='Blogger/Data/Blogger_Raw_Scores.txt'
    data=ImportFeatureData(filename,-1)
    print 'drawing samples'
    vec=np.array([line[1:] for line in data])
    labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
    IDX=np.array([str(int(line[0])) for line in data])
    
    Preds=Classifiers.CrossValidate(vec,labels,IDX,cl,**CVargs)
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
    #fname='Raw_Preds.csv'
    Write_Scores(preds,['id','raw_score'],outfile)
    return

def Analyze_LIWC(metadata,cl,outfile,**CVargs):
    filename='Blogger/Data/Blogger_LIWC_Scores.csv'
    data=ImportCSVFeatureData(filename,-1)
    print 'drawing samples'
    vec=np.array([[float(l) for l in line[1:]] for line in data])   #exclude cases where sex is unknown
    labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
    IDX=np.array([str(int(line[0])) for line in data])
    
    Preds=Classifiers.CrossValidate(vec,labels,IDX,cl,**CVargs)
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
    #fname='LIWC_Preds.csv'
    Write_Scores(preds,['id','liwc_score'],outfile)
    return

def Analyze_Individual(metadata,cl,outfile,**CVargs):
    '''
    grid search shows C>=1 is optimal
    accuracy is unrelated to sample size (remains 84-89% throughout)
    '''
    print 'INDIVIDUAL ANALYSIS'
    #metadata=ImportMeta(-1)
    filename='Blogger/Data/Blogger_Individual_Scores.txt'
    data=ImportFeatureData(filename,-1)
    vec=np.array([line[2:] for line in data if line[1]!=1.0])   #exclude cases where sex is never mentioned
    labels=np.array([metadata[str(int(line[0]))][0] for line in data if line[1]!=1.0])# if 'age' not in line])
    IDX=np.array([str(int(line[0])) for line in data if line[1]!=1.0])

    Preds=Classifiers.CrossValidate(vec,labels,IDX,cl,**CVargs)
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
    #fname='Individual_Preds.csv'
    Write_Scores(preds,['id','indiv_score'],outfile)
    
    return

def Analyze_Structure_Scores(metadata,cl,outfile,**CVargs):
    '''
    SVM - should be linear and C =10. Accuracy maxes out around 3,000 at ~62% but doesn't grow much from 500
    MNB - sample size appears to be unrelated to accuracy. Hovers at 58-62% throughout.
    '''
    print 'STRUCTURE ANALYSIS'

    #metadata=ImportMeta(-1)
    filename='Blogger/Data/Blogger_Structure_Scores.csv'
    data=ImportCSVFeatureData(filename,-1)
    print 'drawing samples'
    vec=np.array([[float(l) for l in line[1:]] for line in data])   #exclude cases where sex is unknown
    labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
    IDX=np.array([str(int(line[0])) for line in data])
    cuts=[]
    print vec.shape
    for i,row in enumerate(vec):
        if np.mean(row)==0:
            cuts.append(i)
    print vec.shape
    vec=np.delete(vec,cuts,axis=0)
    labels=np.delete(labels,cuts,axis=0)
    IDX=np.delete(IDX,cuts,axis=0)
    
    Preds=Classifiers.CrossValidate(vec,labels,IDX,cl,**CVargs)
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
    #fname='Structure_Preds.csv'
    Write_Scores(preds,['id','struct_score'],outfile)
    
    return

def Analyze_Raw_Topic_Scores(metadata,cl,outfile,**CVargs):
    '''
    grid_search shows C>=1 is ideal. remains 71% from 500 through 7000
    remains at 71% at sample sizes from 500 through 10000.
    '''
    print 'RAW TOPIC ANALYSIS'

    #metadata=ImportMeta(-1)
    filename='Blogger/Data/Raw_Topic_Scores.csv'
    data=ImportCSVFeatureData(filename,-1)
    print 'drawing samples'
    vec=np.array([[float(l) for l in line[1:]] for line in data])   #exclude cases where sex is unknown
    labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
    IDX=np.array([str(int(line[0])) for line in data])

    Preds=Classifiers.CrossValidate(vec,labels,IDX,cl,**CVargs)
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
    #fname='Raw_Topic_Preds.csv'
    Write_Scores(preds,['id','rawTopic_score'],outfile)
    
    return


def Analyze_Nonword_Scores(metadata,cl,outfile,**CVargs):
    '''
    
    check rows 1535 and 15349 for inf data. Should no longer have to recode 8 and 12 (herndanV and LnM)
    
    '''
    print 'NONWORD ANALYSIS'

    #metadata=ImportMeta(-1)
    filename='Blogger/Data/Blogger_Nonword_Scores.csv'
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

    Preds=Classifiers.CrossValidate(vec,labels,IDX,cl,**CVargs)
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
    #fname='Nonwords_Preds.csv'
    Write_Scores(preds,['id','nonword_score'],outfile)
    
    return

def Analyze_Behavior_Scores(metadata,cl,outfile,**CVargs):
    '''
    grid_search across all seems to agree that C==10,000 for linear or rbf is optimal
    Sample size doesn't matter because number of texts in an area max out at 6,000
    '''
    print 'BEHAVIOR ANALYSIS'
    
    #num=1000
    #metadata=ImportMeta(-1)
    
    path='Blogger/Data/'
    PREDS={}
    for cat in set([line[2] for line in metadata.values()]):
        if cat=='category':
            continue
        if cat=='Student' or cat=='indUnk':
            args={'n_iter':20, 'test_size':.9,'random_state':0}
        else:
            args=CVargs.copy()
        print 'RUNNINING ', cat, ' BEHAVIOR SCORES'
        f='Blogger_Behavior_'+cat+'_Scores.csv'
        data=ImportCSVFeatureData(path+f,-1)
        vec=np.array([[float(l) for l in line[1:]] for line in data])   #exclude cases where sex is unknown
        labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
        IDX=np.array([str(int(line[0])) for line in data])
        Preds=Classifiers.CrossValidate(vec,labels,IDX,cl,**args)
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
    #fname='Behavior_Preds.csv'
    Write_Scores(PREDS,['id','behavior_score'],outfile)
        
    return


def Analyze_SubTopic_Scores(metadata,cl,outfile,**CVargs):
    '''
    grid_search reveals little variation. Linear 100 or linear 10 seem best, but not a huge effect
    Number of cases doesn't matter for MNB because most are small sample sizes (largest is 6,000).
    '''
    print 'SUBTOPIC ANALYSIS'
    
    #num=1000
    #metadata=ImportMeta(-1)
    path='Blogger/Data/'
    PREDS={}
    for cat in set([line[2] for line in metadata.values()]):
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
        Preds=Classifiers.CrossValidate(vec,labels,IDX,cl,**args)
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
    #fname='SubTopic_Preds.csv'
    Write_Scores(PREDS,['id','subtopic_score'],outfile)
        
    
    return


def Analyze_KBest_Scores(metadata,cl,outfile,**CVargs):
    filename='KBest'
    vec, ids,words=importArray(filename)
    
    print 'drawing samples'
    labels=np.array([metadata[idx][0] for idx in ids])# if 'age' not in line])
    IDX=np.array(ids)
    
    Preds=Classifiers.CrossValidate(vec,labels,IDX,cl,**CVargs)
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
    #fname='Brown/Results/Raw_Preds.csv'
    Write_Scores(preds,['id','kbest_score'],outfile)
    return


def Analyze_Word2Vec_Scores(metadata,cl,outfile,**CVargs):
    filename='Blogger/Data/Blogger_Word2Vec_Scores.csv'
    data=ImportCSVFeatureData(filename,-1)
    print 'drawing samples'
    vec=np.array([[float(l) for l in line[1:]] for line in data])   #exclude cases where sex is 
    labels=np.array([metadata[line[0]][0] for line in data])# if 'age' not in line])
    IDX=np.array([line[0] for line in data])
    Preds=Classifiers.CrossValidate(vec,labels,IDX,cl,**CVargs)
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
    #fname='Brown/Results/Raw_Preds.csv'
    Write_Scores(preds,['id','w2v_score'],outfile)
    return

def hybridTrial(metadata):
    '''
    This code takes two above feature sets and tests whether they change their collective and individual predictability
    Raw * Raw Topics = ? * .72 = .71 (No change in nb score)
    Subtopics * raw topics = .65 * .72 = .69
    '''
    print 'import raw topic scores'
    filename='Blogger/Data/Raw_Topic_Scores.csv'
    data=ImportCSVFeatureData(filename,-1)
    print 'drawing samples'
    vec=np.array([[float(l) for l in line[1:]] for line in data])   #exclude cases where sex is unknown
    labels=np.array([metadata[str(int(line[0]))][0] for line in data])# if 'age' not in line])
    IDX=np.array([str(int(line[0])) for line in data])
    print 'CV for RAW TOPICS'
    CVargs={'n_iter':3, 'test_size':.9,'random_state':0}
    cl=mnb()
        #Preds=Classifiers.CrossValidate(vec,labels,IDX,cl,**CVargs)
        
    print 'importing subtopic scores'
    path='Blogger/Data/'
    preds={}
    Data=[]
    for cat in set([line[2] for line in metadata.values()]):
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
    Preds=Classifiers.CrossValidate(vec,labels,IDX,cl,**CVargs)
    
    CVargs={'n_iter':3, 'test_size':.9,'random_state':0}
    cl=mnb()
    cl=ensemble.AdaBoostClassifier(n_estimators=10)
    Preds=Classifiers.CrossValidate(vec,labels,IDX,cl,**CVargs)
    
    return


metadata=ImportMeta(-1)
##hybridTrial(metadata)
#
#print 'doing Raw'
#
#rawkwargs={'n_iter':20, 'test_size':.9,'random_state':0}
##cl=mnb()
##cl=svm.SVC(C=10,kernel='linear',probability=True)
#cl=ensemble.RandomForestClassifier(n_estimators=1000)
#outfile='Blogger/rf/Raw_Preds.csv'
#Analyze_Raw(metadata,cl,outfile,**rawkwargs)
##
#print 'doing Individual'
#
#indargs={'n_iter':20, 'test_size':.9,'random_state':0}
##cl=mnb()
##cl=svm.SVC(C=10,kernel='linear',probability=True)
#cl=ensemble.RandomForestClassifier(n_estimators=1000)
#outfile='Blogger/rf/Individual_Preds.csv'
#Analyze_Individual(metadata,cl,outfile,**indargs)
##
#print 'doing LIWC'
#
#liwcargs={'n_iter':20, 'test_size':.9,'random_state':0}
##cl=mnb()
##cl=svm.SVC(C=10,kernel='linear',probability=True)
#cl=ensemble.RandomForestClassifier(n_estimators=1000)
#outfile='Blogger/rf/LIWC_Preds.csv'
#Analyze_LIWC(metadata,cl,outfile,**liwcargs)
##
#print 'doing Structure'
#
#strargs={'n_iter':20, 'test_size':.9,'random_state':0}
##cl=mnb()
##cl=svm.SVC(C=10,kernel='linear',probability=True)
#cl=ensemble.RandomForestClassifier(n_estimators=1000)
#outfile='Blogger/rf/Structure_Preds.csv'
#Analyze_Structure_Scores(metadata,cl,outfile,**strargs)     #Structure scores are accurate at around 60% using 50 words per cat.Will rerun to see if this imporves with more features.
##
#print 'doing Raw Topics'
#
#rtargs={'n_iter':20, 'test_size':.9,'random_state':0}
###cl=mnb()
###cl=svm.SVC(C=10,kernel='linear',probability=True)
#cl=ensemble.RandomForestClassifier(n_estimators=1000)
#outfile='Blogger/rf/RawTopic_Preds.csv'
#Analyze_Raw_Topic_Scores(metadata,cl,outfile,**rtargs)
###
#print 'doing Nonwords'
#
#nwargs={'n_iter':20, 'test_size':.9,'random_state':0}
###cl=mnb()
###cl=svm.SVC(C=10,kernel='linear',probability=True)
#cl=ensemble.RandomForestClassifier(n_estimators=1000)
#outfile='Blogger/rf/Nonword_Preds.csv'
#Analyze_Nonword_Scores(metadata,cl,outfile,**nwargs)
#
#print 'doing Behavior'
#
#bargs={'n_iter':100, 'test_size':.1,'random_state':0}
##cl=mnb()
##cl=svm.SVC(C=10000,kernel='linear',probability=True)
#cl=ensemble.RandomForestClassifier(n_estimators=1000)
#outfile='Blogger/rf/Behavior_Preds.csv'
#Analyze_Behavior_Scores(metadata,cl,outfile,**bargs)

print 'doing Subtopic'

stargs={'n_iter':100, 'test_size':.1,'random_state':0}
#cl=mnb()
#cl=svm.SVC(C=100,kernel='linear',probability=True)
cl=ensemble.RandomForestClassifier(n_estimators=1000)
outfile='Blogger/rf/SubTopic_Preds.csv'
Analyze_SubTopic_Scores(metadata,cl,outfile,**stargs)

#

print 'doing KBest'

#stargs={'n_iter':20, 'test_size':.5,'random_state':0}
#cl=mnb()
#cl=svm.SVC(C=10000,kernel='linear',probability=True)
cl=ensemble.RandomForestClassifier(n_estimators=10)
outfile='Blogger/rf/KBest_Preds.csv'
Analyze_KBest_Scores(metadata,cl,outfile,**stargs)


print 'doing Word2Vec'

from sklearn.naive_bayes import GaussianNB
#cl = GaussianNB()
#stargs={'n_iter':20, 'test_size':.5,'random_state':0}
#cl=mnb()
#cl=svm.SVC(C=10000,kernel='linear',probability=True)
cl=ensemble.RandomForestClassifier(n_estimators=10)
outfile='Blogger/rf/Word2Vec_Preds.csv'
Analyze_Word2Vec_Scores(metadata,cl,outfile,**stargs)


#################################################
#
#   OLD CODE FOR TESTING SAMPLE SIZE AND GRID_SEARCH SVM
#
#################################################
#print 'importing data'
#num=1000
#metadata=ImportMeta(-1)
#for num in [500]:#,1000,3000,5000,7000]:
#    filename='Blogger/Data/Blogger_Nonword_Scores.csv'
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

