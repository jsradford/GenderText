    
#Twitter Steps

#unzip the folder - files are xml: <Date><<post>><Date><<post>> (some are not well structured,many missing data);
#filename has data: ID, sex, age, topic, astral sign

#unzip files and create meta-data while we're at it.

'''

This was copied from an outdated Make_Twitter_Data.py.  Update all functions to fit updated code from GenderText/Make_Twitter_Data.py 

'''

import zipfile,os.path,sys,codecs,csv
import xml.etree.ElementTree as ET
from lxml import etree
import nltk
import re
import hashlib
import numpy as np
from itertools import groupby
import requests

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

def importNameDict():
    
    namefile='Datasets/USNameGender.txt'
    names=dict(zip([chr(i) for i in range(97,123)],[{} for i in xrange(26)]))
    with open(namefile,'rb') as nfile:
        raw=nfile.readlines()
        for l in raw:
            l=l.strip().split(',')
            names[l[0].lower()[0]][l[0].lower()]=l
    return names

def EstimateName(name,namedict):
    #femstems=['woman','female','girl','chick','lady','waitress','actress','housewife','wife','mother','daughter','aunt','niece','queen','widow','bachelorette','spinster','bride','mum','mom','mommy','goddess','mistress','princesss','fangirl','witch','waitress','mrs','ms','miss','madame','mademoiselle','frau','seniora','seniorita','duchess','nun','priestes','governess','madam']
    #malestems=['man','guy','male','boy','dude','bro','gentleman', 'waiter','actor','husband','father','son','uncle','nephew','king','widower','bachelor','groom','dad','daddy','god','prince','househusband','fanboy','wizard','mr','mister','monsignor','senor','rabbi','cardinal','lord','duke','priest']
    
    genDict={'Male':'male','Female':'female','Unknown':'?'}
    if name in namedict[name[0]].keys():
        gender=genDict[namedict[name[0]][name][5]]
        #est=float(names[name][6])
        return gender
    else:
        return '?'
        #if name in femstems:
        #    return 'female'
        #elif name in malestems:
        #    return 'male'
        #else:
        #    #print 'could not find gender of ', name
        #    return '?'
 
   
def makeMetaData(num):
    '''metadata header is [party,DWNominate score, name, ICPSR, state code, State-District]
    '''
    texts={}
    source_filename='Datasets/Twitter/raw.zip'
    with zipfile.ZipFile(source_filename) as zf:
        for i,member in enumerate(zf.infolist()):
            if i<num or num == -1:
                name=member.filename.split('/')[1].split('.')[0]
                texts[name]=''
                raw=zf.open(member)
                data=csv.reader(raw)
                for j,line in enumerate(data):
                    if j>0:
                        texts[name]+=unicode(line[0], 'utf-8')+'\n'
                    
    metadata={}
    polKey={}
    missing=[]
    f=open('Datasets/Twitter/IdealPts_112_House-Twitter.csv')
    dta=csv.reader(f)
    for i,line in enumerate(dta):
        if i>0:
            if line[8]!='':
                if line[5]=='100':        #recode to 1,0 fits with previous code and keeps direction of analysis the sm
                    party=1
                elif line[5]=='200':
                    party=0
                else:
                    continue
                    party=''
                tid=line[8].lower()
                
                metadata[tid]=[party, line[7],line[6].lower(),1,line[1],line[2],line[4]+'-'+line[3]]#,numTxt[tid],numWds[tid]]
                polKey[tid]=party
                #else:
                #    missing.append([tid,'House member has Twitter but not in scraped data'])
    f.close()
    f=open('Datasets/Twitter/IdealPts_112_Senate-Twitter.csv')
    dta=csv.reader(f)
    for i,line in enumerate(dta):
        if i>0:
            if line[8]!='':
                if line[5]=='100':
                    party=1
                elif line[5]=='200':
                    party=0
                else:
                    continue
                    party=''
                tid=line[8].lower()
                #if tid in numTxt.keys():
                metadata[tid]=[party, line[7],line[6].lower(),1,line[1],line[2],line[4]]#,numTxt[tid],numWds[tid]]
                polKey[tid]=party
                #else:
                #    missing.append([tid,'Senator has Twitter but not in scraped data'])
    f.close()
    for k in texts.keys():
        if k not in metadata.keys():
            texts.pop(k)
    for k in metadata.keys():
        if k not in texts:
            metadata.pop(k)
    print 'saving meta-data file'
    with open('Twitter/Results/twitterMeta.csv','wb') as f:
        writer=csv.writer(f,delimiter=',')
        writer.writerow(['id','party','dw1', 'name', 'senate', 'ICPSR', 'stateCode', 'district'])
        for idx, info in metadata.iteritems():
            writer.writerow([idx]+info)
    
    
    return metadata,texts
    



##########
#Series of Vectorizing Functions and calls
##########
def MakeLIWCData(texts):
    '''This code calculates LIWC scores for every essay in the corpus and saves it as LIWCScores '''
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
    
    #print 'vectorizing'
    #vec,labels,vectorizer=Vectorize.countVectorize(ltext,dict([(k,'') for k in texts.keys()]))
    #outfile=open('Twitter/Data/Twitter_LIWC_Scores.csv','w')
    #header='id'+'_'.join(catDict.keys())
    header='id,'+",".join(['_'.join(a) for a in zip(['LIWC' for i in catDict],catDict.keys())])+'\n'
    #outfile.write(header)
    print "writing LIWC data to file"
    with open('Twitter/Data/Twitter_LIWC_Scores.csv','wb') as f:
        writer=csv.writer(f,delimiter=',')
        writer.writerow(header)
        for idx,vals in ltext.iteritems():
            writer.writerow([idx]+[str(i) for i in vals])
    
    sys.stdout.flush()
    return

#MakeLIWCData(texts)

#top K Words
def MakeRawScore(texts,num_features=5000):
    #import 'Scripts/Generic/Vectorize.py'
    #reload(Vectorize)
    #runfile('Scripts/Generic/Vectorize.py')
    import Vectorize
    reload(Vectorize)
    
    print 'vectorizing text for raw scoring'
    #texts=RemoveStops(texts)
    vec,labels,vectorizer=Vectorize.tfidfVectorize(texts,dict([(k,'') for k in texts.keys()]),max_features=num_features)
    outfile=open('Twitter/Data/Twitter_Raw_Scores.txt','w')
    header='id,'+",".join(['_'.join(a) for a in zip(['Raw' for i in xrange(len(vectorizer.get_feature_names()))],vectorizer.get_feature_names())])+'\n'
    outfile.write(header)
    print "writing raw data to file"
    for i in xrange(np.shape(vec)[0]):
        row=vec[i].todense().tolist()[0]
        outfile.write(texts.keys()[i]+','+','.join(map(str,row))+'\n')
    sys.stdout.flush()
    outfile.close()
    return

def gensimGenerator(texts,dictionary):
    for text in TopicModels.Tokenize(texts):
        yield dictionary.doc2bow(text)


def getTopicsGenerator(model,texts,dictionary):
    for text in texts:
        bow = dictionary.doc2bow(TopicModels.Tokenize(text))
        topic_in_doc=dict(model.get_document_topics(corpus))
        yield topic_in_doc


def RawTopicScore(texts,numtopics=200,iterations=500,passes=10,name='Twitter_Raw',**exargs):
    '''
    This code runs a topic model on the texts and returns a vector of texts and proportions of topics in texts
    Input:
        texts = {id1: "text1",id2:'text2',...}
    
    '''
    from gensim import corpora
    print 'doing topic modelling on ', numtopics, ' topics'
    #runfile('Scripts/Generic/TopicModels.py')
    import TopicModels
    reload(TopicModels)
    
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
    corpora.MmCorpus.serialize('Twitter/Data/'+name+'_Corpus.mm', corpusgenerator) 
    print 'vectorizing ', name
    tfidf_corpus,tfidf,corpus=TopicModels.vectorize(toktexts,dictionary)
    print 'Doing lda ', name
    mm = corpora.MmCorpus('Twitter/Data/'+name+'_Corpus.mm',)
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
    with open('Twitter/Data/'+name+'_Topic_Scores.csv','wb') as f:
        writer=csv.writer(f,delimiter=',')
        writer.writerow(['id']+["Topic_"+str(n) for n in xrange(numtopics)])
        for info in Data:
            writer.writerow([str(i) for i in info])
                
    print 'writing topic words to Results Folder for ', name
    words=TopicModels.wordsInTopics(model, numWords = 25)
    with open('Twitter/Results/'+name+'_TopicsByWords_'+str(numtopics)+'.csv','wb') as f:
        writer=csv.writer(f,delimiter=',')
        for topic,wordlis in words.iteritems():
            writer.writerow([topic]+[" ".join(wordlis)])
    return

#exargs={'no_below':5,'no_above':.90,'keep_n':10000}
#RawTopicScore(texts,numtopics=100,iterations=500,passes=10,name='Raw',**exargs)

def subTopicsScore(texts,metadata,numtopics=10,name='Twitter'):
    '''This code runs topic models for matched sets of the confounding categories
    
    For other corpuses, adjust matchData=[[k,v[0],v[2]] for k,v in metadata.iteritems()] so that v[2] is the
    string of whatever the subtopic category is supposed to be - 'horrorMove','actionMovie'; 'HouseDemocrat','SenateRepublican',
    The remainder of the code should customizable from the function arguments.
    
    texts = {id1:"some text",}
    meta={id1, [sex1, class1,class2,class3,...],id2:[sex2,class2,class2]}
    numtopics=[num1,num2...]
    
    Returns 
    matchDict = [CatName1: id1,id2...] '''
    #Note: match here is only within subject area, not age or sign -UPDATE v[2] FOR EACH DATA SET
    #matchData=[[k,v[0],v[1]] for k,v in metadata.iteritems()]
    ##match is a set of the third postion
    matchData=[[k]+v for k,v in metadata.iteritems()]
    #match2NumDict=dict(zip(matches,[i for i in xrange(len(matches))]))
    #matchData=[v+[match2NumDict[v[2]]] for v in matchData]    #this is now: id, sex, industry, matchCategory
    #matchDict=dict([(m,[]) for m in matches])
    matchDict={}
    for k,v in metadata.iteritems():
        for item in v[1:]:
            if item in matchDict.keys():
                matchDict[item].append(k) #creates list of ids matched by category: high_poverty:[id1,id2,id3...], excluding sex.
            else:
                matchDict[item]=[k]
    for category,ids in matchDict.iteritems():
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

#d=subTopicsScore(texts,metadata,numtopics=10,name='Twitter')

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


def MakeStructureScore(texts, metadata,num=10):
    '''
    Structure score is the model that best fits predictions of secondary infromation (in Twitter Data, this is age and industry)
    Do men and women talk about different parts of their industry?
    '''
    from gensim import corpora, models, similarities
    from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
    print 'running STRUCTURE analysis: which words most distinguish structural characteristics'
    matchData=[[k]+v for k,v in metadata.iteritems()]
    #texts=RemoveStops(texts)
    kwargs={'binary':False}
    Head=[]
    Data=[[] for i in texts]
    print 'vectorizing corpus'
    vectorizer= tfidf(texts.values(),max_df=.9, min_df=10)
    vec=vectorizer.fit_transform(texts.values()) 
    #vec,labels,vectorizer=Vectorize.tfidfVectorize(texts,{})
    for cats in xrange(len(matchData[0][2:])):
        for cat in set([line[cats+2] for line in matchData]):
            print 'Selecting ', num, ' Best for ', cat
            catlabels=np.asarray([cat==line[cats+2] for i,line in enumerate(matchData)])
            head,data=extractKBest(vec,catlabels,vectorizer,num)
            sys.stdout.flush()
            print "writing Structure scores for ", cat
            with open('Twitter/Data/Twitter_Structure_'+cat+'_Scores.csv','wb') as f:
                writer=csv.writer(f,delimiter=',')
                writer.writerow(['id']+['Struct_'+a for a in head])
                for i,row in enumerate(data):
                    writer.writerow([texts.keys()[i]]+[str(item) for item in row])
    sys.stdout.flush()
    return #Head, Data


#head,data=MakeStructureScore(texts, metadata, num=50)  #there are 35 unique categories, resulting in 35*num features


def makeBehaviorScore(texts, metadata,num=10):
    '''
    
    '''
    from gensim import corpora, models, similarities
    import sklearn
    print 'running BEHAVIOR analysis: which words most distinguish intra-structure behavioral characteristics'
    matchData=[[k]+v for k,v in metadata.iteritems()]   #matchData is not necessary here since strucutre is only one variable, more useful when structure is a combination of variables
    #texts=RemoveStops(texts)
    kwargs={'binary':False}
    Head=[]
    categories=set([line[2] for line in matchData])
    print 'going through categories'
    for cats in xrange(len(matchData[0][2:])):
        for cat in set([line[cats+2] for line in matchData]):
    #for i,cat in enumerate(categories):
            subtexts=dict([(line[0],texts[line[0]]) for line in matchData if line[cats+2]==cat])
            Data=dict([(k,[]) for k in subtexts.keys()])
            catlabels=dict([(line[0],line[1]) for line in matchData if line[cats+2]==cat])
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
            with open('Twitter/Data/Twitter_Behavior_'+cat+'_Scores.csv','wb') as f:
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
    outfile=open('Twitter/Data/Twitter_Individual_Scores.txt','w')
    names=['missing','malecount','femalecount']
    header='id,'+",".join(['Indiv_'+a for a in names])+'\n' #zip(['Raw' for i in xrange(len(vectorizer.get_feature_names()))],vectorizer.get_feature_names())])+'\n'
    outfile.write(header)
    print "writing Individual Words data to file"
    for row in res:
        outfile.write(','.join(map(str,row[0:4]))+'\n')
    sys.stdout.flush()
    outfile.close()
    return #res

def makeNonWordScore(texts,metadata):
    '''
    Features derived from:De Vel and Tweede (1998)
    at D:\Users\Administrator\Documents\Chicago\Research\Gender on Twitter\Readings\Computational Gender
    
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
        txt = text.lower()+' the the'
        t = re.findall(r"[\d\w']+",txt)
        if len(t)<10:
            print 'text with little data'
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
        if sum([len(re.findall('\w+',' '.join(p))) for p in Pars])<5:
            print Pars, 'text with little data'
            continue    #essentially, do you have more than five words longer than three letters.
        lensens=float(sum([len(sens) for sens in Pars]))
        if lensens==0:
            continue
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
        
        head.append('wordsPerSentence')
        row.append(numWords/lensens)        
        head.append('upperCaseSentence')    #I was there.
        row.append(sum([sum([sen.strip()[0].isupper() for sen in sens]) for sens in Pars])/lensens)
        head.append('lowerCaseSentence')    #i was there.
        row.append(sum([sum([sen.strip()[0].islower() for sen in sens]) for sens in Pars])/lensens)
        #head.append('numBlankLines')   - Not relevant in blogs. Originally done for emails.
        data[key]=row
            
        #
    with open('Twitter/Data/Twitter_Nonword_Scores.csv','wb') as f:
        writer=csv.writer(f,delimiter=',')
        writer.writerow(['id']+head)
        for idx,row in data.iteritems():
            writer.writerow([idx]+[str(i) for i in row])
    return #head,data

def makeTwitterData(metadata,texts):
    '''
    This code takes twitter-specific data and creates twitter-specific feature sets
    '''
    mentions={}
    hashtags={}
    
    
    for idx,text in texts.iteritems():
        print idx
        mentions[idx]=''
        hashtags[idx]=''
        words=re.findall(r"[\d\w#@_]+",text)
        for word in words:
            if word.startswith('@'):
                mentions[idx]+=' '+word[1:]
            if word.startswith('#'):
                hashtags[idx]+=' '+word[1:]
                
    #SAVE MENTIONS
    vec,labels,vectorizer=Vectorize.countVectorize(mentions,dict([(k,metadata[k][0]) for k in mentions.keys()]))
    outfile=open('Twitter/Data/Twitter_Mentions_Scores.txt','w')
    header='id,'+",".join(['_'.join(a) for a in zip(['Mentions' for i in xrange(len(vectorizer.get_feature_names()))],vectorizer.get_feature_names())])+'\n'
    outfile.write(header)
    print "writing twitter mentions data to file"
    for i in xrange(np.shape(vec)[0]):
        row=vec[i].todense().tolist()[0]
        outfile.write(texts.keys()[i]+','+','.join(map(str,row))+'\n')
    
    #Save Hashtags
    vec,labels,vectorizer=Vectorize.tfidfVectorize(hashtags,dict([(k,metadata[k][0]) for k in hashtags.keys()]))
    outfile=open('Twitter/Data/Twitter_Hashtags_Scores.txt','w')
    header='id,'+",".join(['_'.join(a) for a in zip(['Hashtags' for i in xrange(len(vectorizer.get_feature_names()))],vectorizer.get_feature_names())])+'\n'
    outfile.write(header)
    print "writing twitter hashtags data to file"
    for i in xrange(np.shape(vec)[0]):
        row=vec[i].todense().tolist()[0]
        outfile.write(texts.keys()[i]+','+','.join(map(str,row))+'\n')
    
    #Save URLs
    return

def makeURLData(metadata,texts):
    urlremove=['www','https','http','com','m']
    urls={}
    for idx,text in texts.iteritems():
        print 'getting urls for', idx
        urls[idx]=''
        words=re.findall(r"[\d\w#@.:/\\]+",text)
        for word in words:
            if word.startswith('http'):
                try:
                    r = requests.head(word, allow_redirects=True,timeout=5)
                except:
                    print 'found exception'
                    print r,word, r.url
                #except:
                #    print 'failed url ', word
                url=r.url
                url=re.findall(r"[\d\w]+",url)
                [url.remove(s) for s in urlremove if s in url]
                urls[idx]+=' '+' '.join(url)
    
    vec,labels,vectorizer=Vectorize.tfidfVectorize(urls,dict([(k,metadata[k]) for k in urls.keys()]))
    outfile=open('Twitter/Data/Twitter_URL_Scores.txt','w')
    header='id,'+",".join(['_'.join(a) for a in zip(['urls' for i in xrange(len(vectorizer.get_feature_names()))],vectorizer.get_feature_names())])+'\n'
    outfile.write(header)
    print "writing twitter mentions data to file"
    for i in xrange(np.shape(vec)[0]):
        row=vec[i].todense().tolist()[0]
        outfile.write(texts.keys()[i]+','+','.join(map(str,row))+'\n')
    
    return

def unitTest():
    
    metadata,texts=unzip(num=10)
    
    return


metadata,texts=makeMetaData(num=20)
##
makeTwitterData(metadata,texts)    #mentions and hashtags
#makeURLData(metadata,texts)        #any url content
MakeLIWCData(texts)                #LIWC features
makeNonWordScore(texts,metadata)   #features here may break, could be recreated for twitter-specific writing
#makeIndivScore(texts,metadata,window=30)
texts=RemoveStops(texts)
#
MakeRawScore(texts, num_features=5000)
exargs={'no_below':2,'no_above':.95,'keep_n':50000}
RawTopicScore(texts,numtopics=30,iterations=500,passes=10,name='Raw',**exargs)

#makeBehaviorScore(texts, metadata,num=100)
#MakeStructureScore(texts, metadata, num=100)  #there are m unique categories, resulting in 100*num features
#subTopicsScore(texts,metadata,numtopics=5,name='Twitter')
#
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


