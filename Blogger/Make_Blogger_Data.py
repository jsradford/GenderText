
#Blogger Steps

#unzip the folder - files are xml: <Date><<post>><Date><<post>> (some are not well structured,many missing data);
#filename has data: ID, sex, age, topic, astral sign

#unzip files and create meta-data while we're at it.

'''
To do:
Rerun Structure_Score using 100 words per group

Create a nonword_score

Individual Gender:
Create an Individual_score algorithm to detect when someone says, "I'm male/female" "as a woman" etc.
Could also do individual score as just using the words/phrases occuring after the phrase "I am a" or "I'm a."
Create a "My name is" version.  Watch out for "He said, 'my name is Nathan.'" It seems oddly frequent.


Download LIWC and generate scores for each corpus.


'''

import zipfile,os.path,sys,codecs,csv
import xml.etree.ElementTree as ET
from lxml import etree
import nltk
import re
import hashlib
import numpy as np
from itertools import groupby
import scipy


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


def RemoveStops(texts):
    stopwords=nltk.corpus.stopwords.words('english')
    for idx,txt in texts.iteritems():
        text = txt.lower()
        text = re.findall(r"[\d\w']+",text)
        list_of_tokens = []
        if stopwords != []:
            for w in text:
                if w not in stopwords:
                    list_of_tokens.append(w)
        else:
            for w in text:
                list_of_tokens.append(w)
        texts[idx]=' '.join(list_of_tokens)
    return texts
    
def importLIWC():
    '''This pulls in the LIWC data set and translates it into a dictionary for sklearn's vectorizer engine
    returns vocab {word: column number for vectorizer}
    and catIndex {LIWC Category: column number for vectorizer}'''
    vocab={}
    categories=[]
    filename='Datasets/LIWC_Features.txt'
    #f=open('C:\DonorsChoose\GenderText\\liwc\\features.txt','r')
    with open(filename,'r') as f:
        for i,line in enumerate(f.readlines()):
            items=line.split(', ')
            category=items[0]
            categories.append(items[0])
            for word in items[1:]:
                if word.endswith('*'):
                    word='\b'+word[:-1]
                else:
                    word='\b'+word+'\\b'
                if word in vocab.keys():
                    vocab[word].append(category)
                else:
                    vocab[word]=[category]
            
    return vocab,categories

def unzip(num=-1):
    '''
    returns
        metadata = {id1, [sex1, age1, topic1, sign1],id2:[data2]}
        texts={id1: "text1";id2: "text2"...} 
    '''
    source_filename='Datasets/Blogger/blogs.zip'
    metadata={}
    texts={}
    htexts={} #stands for hash texts, used to test for duplicate blogs.
    broken=[]
    parser = etree.XMLParser(encoding='utf8',recover=True)
    with zipfile.ZipFile(source_filename) as zf:
        for i,member in enumerate(zf.infolist()):
            if num>0:
                if i<1 or i>num:
                    continue
            else:
                if i<1:
                    continue
            #print i,member.filename
            words = member.filename.split('/')
            data=words[1].split('.')
            raw=zf.open(member)
            raw=raw.read().decode('utf8',errors='replace')
            raw=raw.replace('&', '&amp;')
            eraw=codecs.encode(raw,'utf8', 'replace')
            root=ET.fromstring(eraw,parser=parser)
            text=[]
            for child in root.iter('post'):
                try:
                    text.append(child.text.lstrip().rstrip())
                except:
                    print 'missed line in ', i
                    broken.append(data[0])
            text='\n '.join(text)
            t = text.lower()
            t = re.findall(r"[\d\w']+",t)
            if len(t)<100:
                continue
            ht=hashlib.md5(text.encode('utf-8')).digest()
            if ht not in htexts.values():
                metadata[data[0]]=data[1:len(data)-1]
                metadata[data[0]]=data[1:len(data)-1]
                texts[data[0]]=text
                htexts[data[0]]=ht
            else:
                broken.append(data[0])
            
                
    return metadata,texts

#metadata,texts=unzip(num=1000)

#
#Create Meta-Data file
#
#print 'saving meta-data file'
#with open('bloggerMeta.csv','wb') as f:
#    writer=csv.writer(f,delimiter=',')
#    writer.writerow(['id','sex','age','category','sign'])
#    for id, info in metadata.iteritems():
#        writer.writerow([id]+info)


def extractKBest(vec,labels,vectorizer,num,**kwargs):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    #runfile('Scripts/Generic/Vectorize.py')
    import Vectorize
    reload(Vectorize)
    data=[]
    #vec,labels,vectorizer=Vectorize.tfidfVectorize(texts,catlabels,**kwargs)
    a=SelectKBest(chi2, k=num)
    vec_new =a.fit_transform(vec, labels)
    vec_new=vec_new.astype('float')
    ids=[]
    for i,bl in enumerate(a.get_support()):
        if bl==True:
            ids.append(i)
    Features=[]
    features=vectorizer.get_feature_names()
    for i in ids:
        Features.append(features[i])
    data=[]
    for i in xrange(np.shape(vec_new)[0]):
        data.append(vec_new[i].todense().tolist()[0])
    return Features,data

##########
#Series of Vectorizing Functions and calls
##########
def MakeLIWCData(texts):
    '''This code calculates LIWC scores for every essay in the corpus and saves it as LIWCScores '''
    #import Scripts/Generic/Vectorize.py
    #import 'Scripts/Generic/FeatureSelection.py'
    #reload(FeatureSelection)
    #reload(Vectorize)
    #runfile('Scripts/Generic/FeatureSelection.py')
    #runfile('Scripts/Generic/Vectorize.py')
    #import FeatureSelection
    #import Vectorize
    #reload(FeatureSelection)
    #reload(Vectorize)
    print 'getting LIWC categories'
    vocab,categories=importLIWC()
    ltext={}
    for idx, txt in texts.iteritems():
        txt=txt.lower()
        catDict=dict(zip(categories, [0 for i in categories]))
        for word,cats in vocab.iteritems():
            res=re.findall(r''+word,txt)
            for cat in cats:
                catDict[cat]+=len(res)
        ltext[idx]=catDict.values()
    header='id,'+",".join(['_'.join(a) for a in zip(['LIWC' for i in catDict],catDict.keys())])+'\n'
    #outfile.write(header)
    print "writing LIWC data to file"
    with open('Blogger/Data/Blogger_LIWC_Scores.csv','wb') as f:
        writer=csv.writer(f,delimiter=',')
        writer.writerow(header)
        for idx,vals in ltext.iteritems():
            writer.writerow([idx]+[str(i) for i in vals])
    
    sys.stdout.flush()
    return

#MakeLIWCData(texts)

#top K Words
def MakeRawScore(texts,num_features=5000):
    #import 'C:\DonorsChoose\GenderText\Vectorize.py'
    #reload(Vectorize)
    print 'vectorizing text for raw scoring'
    #texts=RemoveStops(texts)
    vec,labels,vectorizer=Vectorize.tfidfVectorize(texts,dict([(k,'') for k in texts.keys()]),max_features=num_features)
    outfile=open('Blogger/Data/Blogger_Raw_Scores.txt','w')
    header='id,'+",".join(['_'.join(a) for a in zip(['Raw' for i in xrange(len(vectorizer.get_feature_names()))],vectorizer.get_feature_names())])+'\n'
    outfile.write(header)
    print "writing raw data to file"
    for i in xrange(np.shape(vec)[0]):
        row=vec[i].todense().tolist()[0]
        outfile.write(texts.keys()[i]+','+','.join(map(str,row))+'\n')
    sys.stdout.flush()
    outfile.close()
    return

#MakeRawScore(texts, num_features=5000)
def MakeKBestScore(texts,metadata,grams=4, num_features=5000,**tfkwargs):
    '''
    This code takes the texts and returns the top-n features that correlate with the labels
    '''
    print 'vectorizing corpus'
    inlabels=dict([(k,v[0]) for k,v in metadata.iteritems()])
    Words=[[] for x in texts]
    Scores=[[] for x in texts]
    for gram in xrange(grams):
        vec,outlabels,vectorizer=Vectorize.tfidfVectorize(texts,inlabels,ngram_range=(gram+1,gram+1),**tfkwargs)
        #vectorizer= tfidf(texts.values(),max_df=.9, min_df=10,**tfkwargs)
        #vec=vectorizer.fit_transform(texts.values()) 
        words,scores=extractKBest(vec,outlabels,vectorizer,int(round(num_features/grams)))
        for i,w in enumerate(words):
            Words[i]+=w
            Scores[i]+=scores[i]
            
    ids=texts.keys()
    dat=scipy.sparse.csr_matrix(Scores)
    with open('Blogger/Data/Blogger_KBest_Scores.pkl','wb') as f:
        cPickle.dump(dat,f)
    with open('Blogger/Data/Blogger_KBest_Names.pkl','wb') as f:
        cPickle.dump(Words,f)
    with open('Blogger/Data/Blogger_KBest_Ids.pkl','wb') as f:
        cPickle.dump(ids,f)
        
        
    #    sys.stdout.flush()
    #print "writing KBest scores for Blogger"
    ##header=['id']+[u'kbest_'+a.decode('utf-8').replace(' ','_').encode('utf-8') for a in Words]
    #with open('Blogger/Data/Blogger_KBest_Scores.csv','w') as f:
    #    writer=csv.writer(f,delimiter=',')
    #    writer.writerow(['id']+[u'kbest_'+a.decode('utf-8','replace').replace(' ','_').encode('utf-8','replace') for a in Words])
    #    for i,row in enumerate(Scores):
    #        writer.writerow([texts.keys()[i]]+[str(item) for item in row])
    return

def gensimGenerator(texts,dictionary):
    for text in TopicModels.Tokenize(texts):
        yield dictionary.doc2bow(text)


def getTopicsGenerator(model,texts,dictionary):
    for text in texts:
        bow = dictionary.doc2bow(TopicModels.Tokenize(text))
        topic_in_doc=dict(model.get_document_topics(corpus))
        yield topic_in_doc


def RawTopicScore(texts,numtopics=200,iterations=500,passes=10,name='Blogger_Raw',**exargs):
    '''
    This code runs a topic model on the texts and returns a vector of texts and proportions of topics in texts
    Input:
        texts = {id1: "text1",id2:'text2',...}
    
    '''
    from gensim import corpora
    print 'doing topic modelling on ', numtopics, ' topics'
    #runfile('C:\Users\Boss\Documents\Python Scripts\onlineldavb.py')
    print 'tokenizing ', name
    #texts=RemoveStops(texts)
    toktexts=TopicModels.Tokenize(texts)
    dictionary=TopicModels.vocabulary(toktexts)
    print 'original vocabulary size is ', len(dictionary)
    dictionary.filter_extremes(**exargs)#)
    print 'reduced vocabulary size is ',len(dictionary)
    dictionary.compactify()
    print 'reduced vocabulary size is ',len(dictionary)
    corpus = [dictionary.doc2bow(text) for text in TopicModels.Tokenize(texts)]
    corpusgenerator=gensimGenerator(texts,dictionary)
    corpora.MmCorpus.serialize('Blogger/Data/'+name+'_Corpus.mm', corpusgenerator) 
    print 'vectorizing ', name
    tfidf_corpus,tfidf,corpus=TopicModels.vectorize(toktexts,dictionary)
    print 'Doing lda ', name
    mm = corpora.MmCorpus('Blogger/Data/'+name+'_Corpus.mm',)
    model,topic_in_document=TopicModels.topics(mm,dictionary,strategy='lda', num_topics=numtopics,passes=passes,iterations=iterations) #passes=4
    topic_in_documents=[dict(res) for res in topic_in_document] #Returns list of lists =[[(top2, prob), (top8, prob8)],[top1,prob]]
    Data=[]
    for doc, resdict in enumerate(topic_in_documents):
        line=[texts.keys()[doc]]    #This should line up. If it doesn't the model results will be random noise
        for i in xrange(numtopics):
            if i in resdict.keys():
                line.append(resdict[i])
            else:
                line.append(0.0)
        Data.append(line)
    print "writing Document by Topic scores for  ", name
    with open('Blogger/Results/'+name+'_Topic_Scores.csv','wb') as f:
        writer=csv.writer(f,delimiter=',')
        writer.writerow(['id']+["Topic_"+str(n) for n in xrange(numtopics)])
        for info in Data:
            writer.writerow([str(i) for i in info])
                
    print 'writing topic words to Results Folder for ', name
    words=TopicModels.wordsInTopics(model, numWords = 25)
    with open('Blogger/Results/'+name+'_TopicsByWords_'+str(numtopics)+'.csv','wb') as f:
        writer=csv.writer(f,delimiter=',')
        for topic,wordlis in words.iteritems():
            writer.writerow([topic]+[" ".join(wordlis)])
    return

#exargs={'no_below':5,'no_above':.90,'keep_n':10000}
#RawTopicScore(texts,numtopics=100,iterations=500,passes=10,name='Raw',**exargs)

def subTopicsScore(texts,metadata,numtopics=10,name='Blogger'):
    '''This code runs topic models for matched sets of the confounding categories
    
    For other corpuses, adjust matchData=[[k,v[0],v[2]] for k,v in metadata.iteritems()] so that v[2] is the
    string of whatever the subtopic category is supposed to be - 'horrorMove','actionMovie'; 'HouseDemocrat','SenateRepublican',
    The remainder of the code should customizable from the function arguments.
    
    texts = {id1:"some text",}
    meta={id1, [sex1, age1, topic1, sign1],id2:[data2]}
    numtopics=[num1,num2...]
    
    Returns 
    matchDict = [CatName1: id1,id2...] '''
    #Note: match here is only within subject area, not age or sign -UPDATE v[2] FOR EACH DATA SET
    matchData=[[k,v[0],v[2]] for k,v in metadata.iteritems()]
    ##match is a set of the third postion
    matches=set([item[2] for item in matchData])
    match2NumDict=dict(zip(matches,[i for i in xrange(len(matches))]))
    matchData=[v+[match2NumDict[v[2]]] for v in matchData]    #this is now: id, sex, industry, matchCategory
    matchDict=dict([(v[2],[]) for v in matchData])
    for v in matchData:
        matchDict[v[2]].append(v[0]) #creates list of matchCategory: id1,id2,id3... [should be v[3], but on using one category - industry]
    for category,ids in matchDict.iteritems():
        #if category.lower() not in ['indunk']:#category in ['Arts','HumanResources','Publishing','IndUnk','Student'] or category.lower() in ['indunk']:
        #    continue
        matchtexts=dict([(idx,texts[idx]) for idx in ids])
        num=int(round(len(matchtexts)/10.0))
        if num>numtopics:
            num=numtopics
        if num<2:
            num=2
            exargs={'no_below':1}
        else:
            exargs={'no_below':5,'no_above':.95}
        
        RawTopicScore(matchtexts,numtopics=num,iterations=500,passes=10,name=name+'_'+str(category),**exargs)
    
    return #matchDict

#d=subTopicsScore(texts,metadata,numtopics=10,name='Blogger')


def MakeStructureScore(texts, metadata,num=10):
    '''
    Structure score is the model that best fits predictions of secondary infromation (in Blogger Data, this is age and industry)
    Do men and women talk about different parts of their industry?
    '''
    from gensim import corpora, models, similarities
    from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
    print 'running STRUCTURE analysis: which words most distinguish structural characteristics'
    matchData=[[k,v[0],v[1]] for k,v in metadata.iteritems()]
    #texts=RemoveStops(texts)
    kwargs={'binary':False}
    Head=[]
    Data=[[] for i in texts]
    print 'vectorizing corpus'
    vectorizer= tfidf(texts.values())
    vec=vectorizer.fit_transform(texts.values()) 
    for cat in set([line[2] for line in matchData]):
        print 'Selecting ', num, ' Best for ', cat
        catlabels=np.asarray([cat==line[2] for i,line in enumerate(matchData)])
        head,data=extractKBest(vec,catlabels,vectorizer,num)
        sys.stdout.flush()
        print "writing Structure scores for ", cat
        with open('Blogger/Data/Blogger_Structure_'+cat+'_Scores.csv','wb') as f:
            writer=csv.writer(f,delimiter=',')
            writer.writerow(['id']+['Struct_'+a for a in head])
            for i,row in enumerate(data):
                writer.writerow([texts.keys()[i]]+[str(item) for item in row])
    #Prior code for structure scoring no longer in use, but could roll back to
    #for cat in set([line[2] for line in matchData]):
    #    print 'Selecting ', num, ' Best for ', cat
    #    catlabels=dict([(texts.keys()[i],cat==line[2]) for i,line in enumerate(matchData)])
    #    head,data=extractKBest(texts,catlabels,num)
    #    Head+=head
    #    for i,row in enumerate(data):
    #        Data[i]+=row
    #    sys.stdout.flush()
    #print "writing STRUCTURE scores"
    #with open('Blogger/Data/Blogger_Structure_Scores.csv','wb') as f:
    #    writer=csv.writer(f,delimiter=',')
    #    writer.writerow(['id']+['Struc_'+a for a in Head])
    #    for i,row in enumerate(Data):
    #        writer.writerow([texts.keys()[i]]+[str(item) for item in row])  #I don't need to use str(item), but everything is written that way now so I'm keeping it
    sys.stdout.flush()
    return #Head, Data


#head,data=MakeStructureScore(texts, metadata, num=50)  #there are 35 unique categories, resulting in 35*num features


def makeBehaviorScore(texts, metadata,num=10):
    '''
    
    '''
    from gensim import corpora, models, similarities
    import sklearn
    print 'running BEHAVIOR analysis: which words most distinguish intra-structure behavioral characteristics'
    matchData=[[k,v[0],v[2]] for k,v in metadata.iteritems()]   #matchData is not necessary here since strucutre is only one variable, more useful when structure is a combination of variables
    #texts=RemoveStops(texts)
    kwargs={'binary':False}
    Head=[]
    categories=set([line[2] for line in matchData])
    print 'going through categories'
    for i,cat in enumerate(categories):
        subtexts=dict([(line[0],texts[line[0]]) for line in matchData if line[2]==cat])
        Data=dict([(k,[]) for k in subtexts.keys()])
        catlabels=dict([(line[0],line[1]) for line in matchData if line[2]==cat])
        men=sum([v=='male' for v in catlabels.values()])
        women=sum([v=='female' for v in catlabels.values()])
        m=min(men,women)
        #n=(m*2)/10.0
        #if n<5:
        #    num=5
        #elif n>num:
        #    num=50
        #else:
        #    n=num
        print 'Selecting ', num, ' Best for Males (', men,') and Females (',women,') in ', cat
        vec,labels,vectorizer=Vectorize.tfidfVectorize(subtexts,catlabels)
        head,data=extractKBest(vec,labels,vectorizer,num)
        for k,row in enumerate(data):
            Data[subtexts.keys()[k]]=row
        sys.stdout.flush()
        print "writing BEHAVIOR scores"
        with open('Blogger/Data/Blogger_Behavior_'+cat+'_Scores.csv','wb') as f:
            writer=csv.writer(f,delimiter=',')
            writer.writerow(['id']+['Behav_'+a for a in head])
            for k,row in Data.iteritems():
                writer.writerow([k]+[str(item) for item in row])
    sys.stdout.flush()
    return #head,Data

#makeBehaviorScore(texts, metadata,num=10)

def makeIndivScore(texts,metadata,window=30):
    '''
    A test of window sizes shows that there is a peak in accuracy around 20-30 and missing drops slowly off.  So, 30 seems safe.
    
    AT IT'S CURRENT SETTINGS: ACCURACY IS 82.2% with a missing rate of 88.6%
    '''
    #phrases=["I'm a woman", "I'm a male", "I'm female","I'm male", "I'm a girl","I'm a boy","I'm a guy","I'm a dude"]
    phrases=['i am a ',"i'm a ",  'i am an ',"i'm an "]
    femstems=['woman','female','girl','chick','lady','waitress','actress','housewife','wife','mother','daughter','aunt','niece','queen','widow','bachelorette','spinster','bride','mum','mom','mommy','goddess','mistress','princesss','fangirl','witch']
    malestems=['man','guy','male','boy','dude','bro','gentleman', 'waiter','actor','husband','father','son','uncle','nephew','king','widower','bachelor','groom','dad','daddy','god','prince','househusband','fanboy','wizard']
    res=[]
    sens=[]
    for k,t in texts.iteritems():
        t=t.lower()
        m=0
        f=0
        missing=''
        guess=''
        if sum([p in t for p in phrases])>0:
            missing=0
            for p in phrases[1:]:
                t=t.replace(p,phrases[0])
            #r=t.replace("i'm a ","i am a ")
            txt=re.split(phrases[0],t)
            if len(txt)>1:
                for instance in txt[1:]:
                    if len(instance)>window:
                        instance=instance[0:window]
                    clean=re.findall('[\w]+',instance)
                    sens.append(clean)
                    for s in malestems:
                        if s in clean:   #needs to allow "man." or "man,"
                            #sens.append(instance)
                            m+=1
                    for s in femstems:
                        if s in clean:
                            #sens.append(instance)
                            f+=1
        else:
            missing=1
            guess=''
        if m>f:
            guess=0
        if f>m:
            guess=1
        if f==0 and m==0:
            missing=1
        correctDict={'female':1,'male':0}
        correct=correctDict[metadata[k][0]]
        line=[k,missing, m, f, guess,correct]
        res.append(line)
    print 'Missing rate = ', sum([r[1]==1 for r in res])/float(len(res))
    print 'Accuracy rate = ', sum([r[5]==r[4] for r in res if r[4]!=''])/float(sum([r[1]==0 for r in res]))
    print 'writing to file'
    outfile=open('Blogger/Data/Blogger_Individual_Scores.txt','w')
    names=['missing','malecount','femalecount']
    header='id,'+",".join(['Indiv_'+a for a in names])+'\n' #zip(['Raw' for i in xrange(len(vectorizer.get_feature_names()))],vectorizer.get_feature_names())])+'\n'
    outfile.write(header)
    print "writing Individual Words data to file"
    for row in res:
        outfile.write(','.join(map(str,row[0:4]))+'\n')
    sys.stdout.flush()
    outfile.close()
    return #res


def makeWord2Vec(texts,num_features,**kwargs):
    from gensim.models import word2vec
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    print 'extracting sentences'
    #prepare texts
    sentences = []
    for idx,text in texts.iteritems():
        raw_sentences = tokenizer.tokenize(text.strip())
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(re.findall(r"[\d\w']+",raw_sentence))
    print 'training word2vec model'
    #train model
    model=word2vec.Word2Vec(sentences,**kwargs)
    model.init_sims(replace=True)
    
    #cache model. use Word2Vec.load(model_name)to import
    model_name = "blogger_300features_40minwords_10context"
    model.save(model_name)
    
    print 'vectorizing data'
    textFeatureVecs = np.zeros((len(texts),num_features),dtype="float32")
    counter=0
    for idx,text in texts.iteritems():
        words=re.findall(r"[\d\w']+",text)
        nwords=0
        featureVec = np.zeros((num_features,),dtype="float32")
        index2word_set = set(model.index2word)
        for word in words:
            if word in index2word_set: 
                nwords+=1
                featureVec = np.add(featureVec,model[word])
        textFeatureVecs[counter] = np.divide(featureVec,nwords)
        counter+=1
        #Turn text into list of words
        #For each word, 
    
    print 'writing word2vec scores'
    with open('GenderText\Blogger\Data\Blogger_Word2Vec_Scores.csv','wb') as f:
        writer=csv.writer(f,delimiter=',')
        writer.writerow(['id']+['w2v_'+str(i) for i,a in enumerate(textFeatureVecs[2])])
        for i,row in enumerate(textFeatureVecs):
            writer.writerow([texts.keys()[i]]+[str(v) for v in textFeatureVecs[i]])
    return 

def makeNonWordScore(texts,metadata):
    '''
    Features derived from:De Vel and Tweede (1998)
    at D:\Users\Administrator\Documents\Chicago\Research\Gender on DonorsChoose\Readings\Computational Gender
    
    #error at simp=sum([len(list(g))*(freq/f... 'float division by zero' I wonder if freq can't be calculated twice
    '''
    #metadata,texts=unzip(num=-1)
    from scipy import stats
    from collections import Counter
    #NUMCHAR == NUMWORDS HERE
    prints=[i*100 for i in xrange(int(len(texts)/100.0))]
    iterator=0
    errors=[]
    data={}
    for key,text in texts.iteritems():
        iterator+=1
        if iterator in prints:
            print iterator
        row=[]
        head=[]
        #GETTING WORD COUNT AVG WORD LENGTH
        txt = text.lower()
        t = re.findall(r"[\d\w']+",txt)
        if len(t)<10:
            print 'blog with little data'
            errors.append(key)
            continue
        head.append('numWords')
        numWords=float(len(t))
        row.append(len(t))
        head.append('avgWordLen')
        row.append(sum([len(i) for i in t])/float(len(t)))
        head.append('richness')
        row.append(len(set(t))/float(len(t)))
        head.append('shortwords')
        row.append(sum([len(i)<5 for i in t])/float(len(t)))
        
        #GETTING LEXICAL DIVERSITY MEASURES - FROM DE VEL, TWEEDIE & BAAYEN '98
        counts=Counter(t)
        head.append('hapaxlogema1')
        row.append(sum([1 for k,v in counts.iteritems() if v==1])/float(len(counts)))
        head.append('hapaxlogema2')
        row.append(sum([1 for k,v in counts.iteritems() if v==1])/float(len(t)))
        head.append('guiraud')
        row.append(float(len(counts))/np.sqrt(len(t)))
        head.append('herdanC')
        row.append(float(np.log(len(counts)))/np.log(len(t)))
        head.append('herdanV')
        hv=0.0
        for freq,g in groupby(sorted(counts.values())):
            l=float(len(list(g)))
            hv+=sum([l*((freq/float(len(t)))**2) - 1/l])
        row.append(hv*-1)
        head.append('rubet')
        row.append(float(np.log(len(counts)))/np.log(np.log(len(t))))
        head.append('maas')
        row.append(np.log(len(t)-np.log(len(counts)))/np.log2(len(t)))
        head.append('dugast')
        row.append(np.log2(len(t))/(np.log(len(t))-np.log(len(counts))))
        head.append('LnM')
        row.append(-1.0*(1.0-(len(counts)**2))/((len(counts)**2)*np.log(len(t))))
        head.append('brunet')
        row.append(pow(len(t),pow(len(counts),-.172)))
        head.append('honore')
        row.append(100.0*(np.log(len(t))/(1-(sum([1 for k,v in counts.iteritems() if v==1])/float(len(counts))))))
        head.append('sichel')
        row.append(sum([1 for k,v in counts.iteritems() if v==2])/float(len(counts)))
        head.append('yuleK')
        M1 = float(len(t))
        M2 = sum([len(list(g))*(freq**2) for freq,g in groupby(sorted(counts.values()))])
        row.append(10000*(M2-M1)/(M1*M1))
        head.append('simpson')
        simp=sum([len(list(g))*(freq/float(len(t)))*((freq-1)/(len(t)-1.0)) for freq,g in groupby(sorted(counts.values()))])
        row.append(simp)
        head.append('entropy')
        row.append(stats.entropy(counts.values()))
        
        #GETTING CHARACTER MEASURES - LARGELY FROM CHENG (DE VEL HAS A BIT MORE)
        #txt = text.lower()     #txt is called above
        head.append("numChar")
        C=float(len(txt))
        row.append(C)
        head.append("numLetters")
        t = re.findall(r'[a-z]',txt)
        row.append(len(t)/C)
        head.append("numUpper")
        t=re.findall(r'[A-Z]',text)
        row.append(len(t)/C)
        head.append("numDigits")
        t=re.findall(r'[0-9]',txt)
        row.append(len(t)/C)
        head.append("numWhitespace")
        t=re.findall(r'[\s]',txt)
        row.append(len(t)/C)
        head.append("numTabspace")
        t=re.findall(r'[\t]',txt)
        row.append(len(t)/C)
        #NUM SPECIAL CHARACTERS (sans syntactical characters used below)
        for ch in ['@','#','$','%','^','&','*','(',')','{','}','[',']','`','~','<','>','/','|','+','-','_']:
            t=re.findall(r'[\\'+ch+']',txt)
            row.append(len(t)/C)
            head.append('num'+ch)
        t=re.findall(r'[\\]',txt)
        row.append(len(t)/C)
        head.append('num'+'\\')
        
        #GETTING GENDER-PREFERENTIAL WORDS - FROM DE VEL
        #txt=text.lower()   #this is called above
        t = [r for r in re.findall(r"[\w]+",txt) if len(r)>3]
        head.append("ableWords")
        row.append(sum([w.endswith('able') for w in t])/numWords)
        head.append("alWords")
        row.append(sum([w.endswith('al') for w in t])/numWords)
        head.append("fulWords")
        row.append(sum([w.endswith('ful') for w in t])/numWords)
        head.append("ibleWords")
        row.append(sum([w.endswith('ible') for w in t])/numWords)
        head.append("icWords")
        row.append(sum([w.endswith('ic') for w in t])/numWords)
        head.append("iveWords")
        row.append(sum([w.endswith('ive') for w in t])/numWords)
        head.append("lessWords")
        row.append(sum([w.endswith('less') for w in t])/numWords)
        head.append("lyWords")
        row.append(sum([w.endswith('ly') for w in t])/numWords)
        head.append("ousWords")
        row.append(sum([w.endswith('ous') for w in t])/numWords)
        head.append("apologWords")
        row.append(sum([w.startswith('apolog') for w in t])/numWords)    
        head.append("sorryWords")
        row.append(sum([w in ['sorry'] for w in t])/numWords)    
        
        
        #GETTING SYNTACTICAL FEATURES   #IN DE VEL, THEY JUST USE NUM_PUNCTUATION
        head.append("numSingleQoutes")
        t=re.findall(r"[']",txt)
        row.append(len(t)/C)
        head.append("numPeriods")
        t=re.findall(r'[.]+',txt)
        row.append(sum([len(i)==1 for i in t])/C)
        head.append("numElipses")
        t=re.findall(r'[.]+',txt)
        row.append(sum([len(i)>1 for i in t])/C)
        head.append("numCommas")
        t=re.findall(r'[,]',txt)
        row.append(len(t)/C)
        head.append("numColons")
        t=re.findall(r'[:]',txt)
        row.append(len(t)/C)
        head.append("numSemicolons")
        t=re.findall(r'[;]',txt)
        row.append(len(t)/C)
        head.append("numQuestionMarks")
        t=re.findall(r'[?]+',txt)
        row.append(sum([len(i)==1 for i in t])/C)
        head.append("numMultQuestionMarks")
        row.append(sum([len(i)>1 for i in t])/C)
        head.append("numExclamationMarks")
        t=re.findall(r'[!]+',txt)
        row.append(sum([len(i)==1 for i in t])/C)
        head.append("numMultExclamationMarks")
        row.append(sum([len(i)>1 for i in t])/C)
        
        #GETTING STRUCTURAL FEATURES (see Cheng)
        txt=text.replace('!','').replace('?','.')
        pars=txt.split('\n')
        totsen=[]
        Pars=[]
        for par in pars:
            if '.' in par:
                sens=par.split('.')
                sens=[sen for sen in sens if len(sen.strip())>3]
                Pars.append(sens)
            else:
                if len(par.strip())<3:
                    continue
                else:
                    Pars.append([par])
        
        head.append('numParagraphs')
        row.append(len(Pars))
        head.append('numSentences')
        row.append(sum([len(sens) for sens in Pars]))
        head.append('sensPerParagraph')
        row.append(np.mean([len(sens) for sens in Pars]))
        head.append('wordsPerParagraph')
        row.append(numWords/len(Pars))
        head.append('charsPerParagraph')
        row.append(np.mean([sum([len(sen) for sen in sens]) for sens in Pars]))
        lensens=float(sum([len(sens) for sens in Pars]))
        head.append('wordsPerSentence')
        row.append(numWords/lensens)        
        head.append('upperCaseSentence')    #I was there.
        row.append(sum([sum([sen.strip()[0].isupper() for sen in sens]) for sens in Pars])/lensens)
        head.append('lowerCaseSentence')    #i was there.
        row.append(sum([sum([sen.strip()[0].islower() for sen in sens]) for sens in Pars])/lensens)
        #head.append('numBlankLines')   - Not relevant in blogs. Originally done for emails.
        data[key]=row
        
        #
    with open('Blogger/Data/Blogger_Nonword_Scores.csv','wb') as f:
        writer=csv.writer(f,delimiter=',')
        writer.writerow(['id']+head)
        for idx,row in data.iteritems():
            writer.writerow([idx]+[str(i) for i in row])
    return head,data

metadata,texts=unzip(num=100)
#MakeLIWCData(texts)
#head,data=makeNonWordScore(texts,metadata)
#makeIndivScore(texts,metadata,window=30)
tfkwargs={'max_df':.9,'min_df':5}
grams=4
MakeKBestScore(texts,metadata,num_features=10,grams=grams,**tfkwargs)

#texts=RemoveStops(texts)
#exargs={'no_below':5,'no_above':.90,'keep_n':100000}

#MakeRawScore(texts, num_features=10000)
#RawTopicScore(texts,numtopics=100,iterations=500,passes=10,name='Raw',**exargs)
#MakeStructureScore(texts, metadata, num=100)  #there are 35 unique categories, resulting in 35*num features
#makeBehaviorScore(texts, metadata,num=100)
#subTopicsScore(texts,metadata,numtopics=10,name='Blogger')
#num_features=300
#kwargs={'size':num_features,'min_count':2,'workers':4,'window':10,'sample':1e-3}
#res=makeWord2Vec(texts,num_features,**kwargs)

#
    
    #can use the list from studies of male/female titles
    #this needs to allow people who say "I'm a man" and "I'm a woman" as forms of error.
    #could use a "I am a " find the next noun - is it in femstems or malestems. [Or, could do a, in the next N words, do you see an fem/malestem]
    #The accuracy can be tested here.  The trade off is type 1/type 2 more so than noise.
    
#Create Vectorized Files
#
#Summary: each file contains tfidf of different features, TFIDF-LIWC, TFIDF-raw, TFIDF-category, TFIDF-matched, TFIDF-topics, TFIDF-subtopics, TFIDF-self
#   Doc1: liwc features, raw features, category features, matched features, ..., etc.
#   Doc2: liwc features, raw features, category features, matched features, ..., etc.
#Extract Features
#LIWC
#bag of words

#LIWC


