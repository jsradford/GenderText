'''
Gendered Language Sandbox
   Create 8/21/2014
   This code is created as a sanbox to pull out small samples of texts along with any categorical
   information such as gender, grade, subject, price, etc.
'''
import numpy as np
import os
import random as rn
import csv
import operator
    
def getCSVData(fname,keyname,categories=''):
    '''this code generates the list of match classes to be included in the training set and those left out of the training set
    matchName is the name of the varaible in the file for the match group variable.  pct is the percent of matched groups to
    select from all match groups.  Returns a list of the selected matched groups and a list of those held out '''
    matches=dict(zip(categories,[{} for c in xrange(len(categories))]))
    classKey={}
    cidx={}
    f=open(fname,'rb')
    data=csv.reader(f)
    for i,line in enumerate(data):
        if i==0:
            for j,l in enumerate(line):
                if l.lower()==keyname.lower():
                    pid=j
                if l.lower()in [c.lower() for c in categories]:
                    for c in categories:
                        if l.lower()==c.lower():
                            cidx[j]=c
        if i>0:
            if '' not in [line[j] for j in cidx]:
                p=line[pid]
                #catvals=[]
                for j in cidx.keys():
                    catval=line[j]
                    #catvals.append(catval)
                    if catval in matches[cidx[j]].keys():
                        matches[cidx[j]][catval].append(p)
                    else:
                        matches[cidx[j]][catval]=[]
                        matches[cidx[j]][catval].append(p)
            #classKey[p]=catvals
    f.close()
    
    return matches#,classKey



def balanceSample(matches,targetCat='',num=0):
    print 'Drawing a balanced sample of ', str(num), ' cases'
    bmatch={}
    bmatch[targetCat]={}
    if num==0:
        mn=min([len(k) for k in matches[targetCat].values()])
        for val, ids in matches[targetCat].iteritems():
            bids=rn.sample(ids,mn)
            bmatch[targetCat][val]=bids
    else:
        mn=min([len(k) for k in matches[targetCat].values()])
        if num>mn:
            num=mn
        for val, ids in matches[targetCat].iteritems():
            bids=rn.sample(ids,num)
            bmatch[targetCat][val]=bids
            
    return bmatch

def randomSample(matches,targetCat='',num=0):
    print 'Drawing a random sample of ', str(num), ' cases'
    rmatch={}
    rmatch[targetCat]={}
    ids=[item for sublist in matches[targetCat].values() for item in sublist]
    rids=rn.sample(ids,num)
    for val,ids in matches[targetCat].iteritems():
        rmatch[targetCat][val]=[]
        for idx in ids:
            if idx in rids:
                rmatch[targetCat][val].append(idx)
    return rmatch
        
def trainTestSample(vector, vlabels, samp=''):
    print 'Diving sample into testing and training corpora'
    if type(samp)==np.ndarray:
        trainIds=list(samp)
    elif type(samp)==int or type(samp)==float:
        if samp<1:
            trainIds=rn.sample(xrange(np.shape(vlabels)[0]),int(round(np.shape(vlabels)[0]*samp)))
        else:
            trainIds=rn.sample(xrange(np.shape(vlabels)[0]),samp)

    testIds=[]
    ts=0
    tr=0
    for t in xrange(np.shape(vlabels)[0]):    
        if t not in trainIds:
            testIds.append(t)
            ts+=1
        else:
            tr+=1
    trainTexts=vector[trainIds]
    trainLabels=vlabels[trainIds]
    testTexts=vector[testIds]
    testLabels=vlabels[testIds]
    
    return trainTexts, trainLabels,testTexts,testLabels
