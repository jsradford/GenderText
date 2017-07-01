'''Created 9-22-2014 using donorschoose data to test the implementation
Testing Notes

RESULTS:
The resulting data is collected by two operations: model=models.LdaModel() and document_scores = model[documents]
Both of these objects are class-based and difficult to index natively.
    Document_scores is an iterable with [(topic1, model weight1), (topicN, model weightN)].
    These results are limited to non-zero topics, so if a document doesn't have topicX, then there is no (topicX,model wieghtX)
    in document_scores.

    model - is the fitted topic model.  To get individual document scores, use -  for doc in corpus: doc_score = model[doc]
    


'''
import re
import numpy as np
import operator
#import DrawSamples
#import Vectorize
#import FeatureSelection
#import Classifiers
#import FeatureExtractors
#reload(DrawSamples)
#reload(Vectorize)
#reload(FeatureSelection)
#reload(Classifiers)
#reload(FeatureExtractors)

from gensim import corpora, models, similarities

def importData(n=100):
    sfile='C:\DonorsChoose\Data\DataRerun2009Original2.csv'
    matches=DrawSamples.getSample(sfile,categories=['male'])
    rmatches=DrawSamples.randomSample(matches, num=n)
    targetKey={}
    for cat,ids in rmatches['male'].iteritems():
        for idx in ids:
            targetKey[idx]=cat
    texts=Vectorize.ExtractTexts(targetKey.keys())
    return texts, targetKey,rmatches

def Tokenize(texts):
    toktexts=[]
    for txt in texts.values():
        text = txt.lower()
        text = re.findall(r"[\d\w']+",text)
        list_of_tokens = []
        for w in text:
            list_of_tokens.append(w)
        toktexts.append(list_of_tokens)
    return toktexts

def vocabulary(toktexts):
    dictionary = corpora.Dictionary(toktexts)
    return dictionary

def vectorize(toktexts,dictionary):
    corpus = [dictionary.doc2bow(text) for text in toktexts]
    tfidf = models.TfidfModel(corpus)
    tfidf_corpus=tfidf[corpus]
    return tfidf_corpus,tfidf,corpus

def topics(documents,dictionary,strategy='lda', num_topics=3,iterations=50,passes=1,**kwargs):
    """
    Strategies and best practices are:
    "lsi" - latent semantic indexing. Documents = tfidf_corpus. Num is 200-500 topics.
    "lda" - latent dirichlet analyisis. Documents = corpus. Num is expert driven.
    "rp" - Random projections. Documents = tfidf_corpus, Num is 100-10000
    "hdp" - Hierarchical Dirichlet Process = corpus. Num is not used.
    """
    if strategy == "lsi":
        model=models.LsiModel(documents, id2word=dictionary,  num_topics=num_topics,iterations=iterations,passes=passes,**kwargs)
        
    
    if strategy == "lda":
        model = models.LdaModel(documents, id2word=dictionary, num_topics=num_topics,iterations=iterations,passes=passes,**kwargs)
    
    if strategy == "rp":
        model = models.RpModel(documents,  num_topics=num_topics,iterations=iterations,passes=passes,**kwargs)
    
    if strategy == "hdp":
        model = models.HdpModel(documents, id2word=dictionary, **kwargs)
    results=model[documents]
    return model,results

def TopicsInDocuments(model,results,docCut=.90):
    '''
    Takes resulting model, corpus, and number of topics and outputs
    topDocs[topicNum] = Top Docs
    documentTopics[doc]= best topic
    '''
    
    topic_in_documents={}
    document_By_Topic=dict(zip(xrange(model.num_topics),[[] for x in xrange(model.num_topics)]))
    for i,doc in enumerate(results):
        res=sorted(doc, key=operator.itemgetter(1),reverse=True)[0]
        topic_in_documents[i] = res[0]
        #print res
        if res[1]>docCut:
            document_By_Topic[res[0]].append(i)

    return document_By_Topic, topic_in_documents

def wordsInTopics(model, numWords=10,strategy='lsa'):
    '''Takes the model object produced from topics() and the number of topwords to grab (i.e. numwords) and returns
    words[topicNum]= [word1...wordN} in order by weight'''
    #list of lists of len(num_topics) where each list is tuples (weight, word) of len(numWords)
    words={}
    for t in xrange(model.num_topics):
        words[t]=[wordscore[1] for wordscore in model.show_topic(t,topn=numWords)]

    return words



#texts, targetKey,rmatches=importData()
#toktexts=Tokenize(texts)
#dictionary=vocabulary(toktexts)
#tfidf_corpus,tfidf,corpus=vectorize(toktexts,dictionary)
#tester=['lda','lsi']
#for strategy in tester:
#    print 'Doing ', strategy
#    model,results=topics(corpus,dictionary,strategy=strategy, num=10)
#    document_By_Topic, topic_in_documents = TopicsInDocuments(model,results,docCut=.90)
#    print 'example topic in document: ', topic_in_documents[2]
#    words=wordsInTopics(model, numWords = 15)
#    print 'example top words: ',  words[2]

