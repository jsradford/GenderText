'''
Created 8/21/2014

Houses Vectorization codes for taking a list of texts and translating them into various versions of
Doc x Feature Vectors
'''

from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.feature_extraction.text import CountVectorizer as CV
import numpy as np



def ExtractTexts(samples,essayDir=''):
    '''This code grabs the texts for each eligible essay selected by the subsamples within selected matched groups
    returns a dictionary of dict[project id]=text.  Type can be either "one2one" or "group."  Samples should be a
    the list of projectIDs from ExactSample functions'''
    print 'getting %d essays' % len(samples)
    texts={}
    er=0
    if essayDir=='':
        essayDir='C:\C_Backup\WorkSpace\DonorsChoose\Essays'
    for proj in samples: 
        fname=essayDir+'\\'+proj+'.txt'
        try:
            f=open(fname,'rb')
            texts[proj]=f.read()
            f.close()
        except:
            er+=1
            continue
    print '%d MISSING CASES' % er
    return texts

def tfidfVectorize(texts,genKey,vocabulary=None,stop_words=None,min_df=1,max_df=1.0,ngram_range=(1,1),max_features=None):
    '''This will likely require fixing so that I can pass some of the parameters into this function and keep the remaing functions
    unused - i.e. to have Vectorize(...vocabulary=someVocab,ngram_range=(1,3)) and Vectorize(...stopwords=someStops,max_features=1000)
    see options here: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    
    NOTE: Check by using an empty vocabulary first.  This should break.  Others are default values and should be okay.
    '''
    print 'vectorizing ', str(len(texts)),' texts'
    vectorizer= tfidf(texts.values(),stop_words=stop_words,vocabulary=vocabulary,min_df=min_df,ngram_range=ngram_range,max_features=max_features)
    vec=vectorizer.fit_transform(texts.values()) 
    labels=[]
    for k in texts.keys():
        labels.append(genKey[k])
    labels=np.asarray(labels)   
    return vec,labels,vectorizer

def countVectorize(texts,genKey,vocabulary=None,stop_words=None,min_df=1,max_df=1.0,ngram_range=(1,1),max_features=None,**kwargs):
    '''This will likely require fixing so that I can pass some of the parameters into this function and keep the remaing functions
    unused - i.e. to have Vectorize(...vocabulary=someVocab,ngram_range=(1,3)) and Vectorize(...stopwords=someStops,max_features=1000)
    see options here: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    
    NOTE: Check by using an empty vocabulary first.  This should break.  Others are default values and should be okay.
    '''
    print 'vectorizing ', str(len(texts)),' texts'
    vectorizer= CV(texts.values(),stop_words=stop_words,vocabulary=vocabulary,min_df=min_df,ngram_range=ngram_range,max_features=max_features,**kwargs)   
    vec=vectorizer.fit_transform(texts.values()) 
    labels=[]
    for k in texts.keys():
        labels.append(genKey[k])
    labels=np.asarray(labels)   
    return vec,labels,vectorizer

