'''
Created 8/21/2014
This code is a collection of vocabularies to be used in feature selection
It include LIWC
'''
import re

def tokenize(txt,stopwords=[]):
    '''Takes a string txt and returns it as a list of tokens '''
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
    return list_of_tokens

def LIWCVocab():
    '''This pulls in the LIWC data set and translates it into a dictionary for sklearn's vectorizer engine
    returns vocab {word: column number for vectorizer}
    and catIndex {LIWC Category: column number for vectorizer}'''
    vocab={}
    catIndex={}
    f=open('C:\DonorsChoose\GenderText\\liwc\\features.txt','r')
    for i,line in enumerate(f.readlines()):
        items=line.split(', ')
        category=items[0]
        for word in items[1:]:
            if word in vocab.keys():
                vocab[word].append(i)
            else:
                vocab[word]=[i]
        #categories.append(category)
        catIndex[i]=category
    return vocab,catIndex

def stemmatize(texts):
    '''takes a dictionary ids: textsStrings and returns ids: stemmed text strings'''
    from nltk.stem import PorterStemmer
    stemmer=PorterStemmer()
    stexts={}
    for t,txt in texts.iteritems():
        stexts[t]=' '.join([stemmer.stem(w) for w in tokenize(txt)])
    return stexts

def LIWCize(texts):
    '''This code takes a dictionary of raw string texts (texts={id1:'something',...}) and converts it to LIWC categories
    Returns texts with words as LIWC codes: {id1:"code1 code8 code1 code3..."}'''
    from nltk.stem import PorterStemmer
    stemmer=PorterStemmer()
    ltext={}
    lvocab,lcatIndex=LIWCVocab()
    for idx, txt in texts.iteritems():
        ltext[idx]=[]
        toks=tokenize(txt)
        stoks=[stemmer.stem(tok) for tok in toks]
        lists_of_LIWC_codes=[lvocab[stok] for stok in stoks if stok in lvocab.keys()]
        if len(lists_of_LIWC_codes)>0:
            for list_of_codes in lists_of_LIWC_codes:
                ltext[idx]+=[lcatIndex[code] for code in list_of_codes]
            ltext[idx]=" ".join(ltext[idx])
        else:
            print 'found an essay without any LIWC features'
            ltext[idx]=''
        
    return ltext
