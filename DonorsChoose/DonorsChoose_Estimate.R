#setwd('C:/Data/GenderText')

#source('C://DonorsChoose//Stratification//WindowFunctions.R')

#library(MASS)
#library(Zelig)

#fails:
    #subtopic
meta<-read.table('DonorsChoose/Results/DonorsChooseMeta.csv',sep=',',header=T)
#GENERAL ANALYSIS FOR ALL DATA
beh<-read.table('DonorsChoose/rf/Behavior_Preds.csv',sep=',' ,header=T)
raw<-read.table('DonorsChoose/rf/Raw_Preds.csv',sep=',' ,header=T)
rt<-read.table('DonorsChoose/rf/Raw_Topic_Preds.csv',sep=',' ,header=T)
struct<-read.table('DonorsChoose/rf/Structure_Preds.csv',sep=',' ,header=T)
st<-read.table('DonorsChoose/rf/SubTopic_Preds.csv',sep=',' ,header=T)
liwc<-read.table('DonorsChoose/rf/LIWC_Preds.csv',sep=',' ,header=T)
w2v<-read.table('DonorsChoose/rf/Word2Vec_Preds.csv',sep=',' ,header=T)
kb<-read.table('DonorsChoose/rf/KBest_Preds.csv',sep=',' ,header=T)
nw<-read.table('DonorsChoose/rf/Nonwords_Preds.csv',sep=',' ,header=T)
ind<-read.table('DonorsChoose/rf/Individual_Preds.csv',sep=',' ,header=T)   #only 77 people had data


library(plyr)
dta<-join_all(list(meta,beh,raw,rt,struct,st,liwc,w2v,kb,nw,ind), by = 'id', type = 'full')
colnames(dta)<-c(names(meta),'beh','raw','rawtopic','struct','subtopic','liwc','w2v','kbest','nonwords','ind')



#########################################
##       DIRECT MODEL SKELETON
#########################################

model<-glm(sex~raw,data=dta,family=binomial)
print(summary(model))
model<-glm(sex~raw+rawtopic+struct+subtopic+beh+nonwords,data=dta,family=binomial)
print(summary(model))
model<-glm(sex~raw+rawtopic+struct+subtopic+beh+nonwords+liwc+w2v+kbest,data=dta,family=binomial)
print(summary(model))
model<-glm(sex~raw+ind+rawtopic+struct+subtopic+beh+nonwords,data=dta,family=binomial)
print(summary(model))

accuracies<-function(dta,model){
    dta$preds<-predict(model,dta,type='response',na.action = na.pass)
    #print length(dta$preds)
    num<-sum((dta$preds>.5 & dta$sex=='male') | (dta$preds<.5& dta$sex=='female'),na.rm=T)
    denom<-sum(is.na(dta$preds)==F)
    pct<-num/denom
    print(c('Accuracy for model is ',pct))
}

saveImportanceChart<-function(fit,file){
    mat<-fit$importance
    vals<-mat[,'MeanDecreaseAccuracy']
    vars<-rownames(mat)
    jpeg(file)
    barplot(vals, main="DonorsChoose Variable Importance",xlab='Model',ylab='Change in Accuracy',names.arg=vars)
    dev.off()
}

names<-c('raw','rawtopic','w2v','kbest','liwc','nonword','behav.','struct.','subtop.','indiv.')
counts<-c(.71,.70,.62,.72,.51,.62,.68,.60,.69,0)
jpeg('DonorsChoose/Results/DonorsChoose_Accuracy_Bar.jpg')
barplot(counts, main="DonorsChoose Classifier Accuracy",ylim=c(0,1),xlab='Model',ylab='Accuracy', cex.names=.6,
  names.arg=names)
dev.off()
#
#axis(side=1, at=1:length(counts), labels=c('Raw','LIWC','Nonword','Raw Topic','Category','Subtopic','Behavior','Individual'),cex.axis=1)
#
#########################################
##       VOTING MODEL
#########################################
#
#dta$braw<-as.numeric(dta$raw>0)
#dta$brawtopic<-as.numeric(dta$rawtopic>0)
#dta$bind<-as.numeric(dta$ind>0)
#dta$bbeh<-as.numeric(dta$beh>0)
#dta$bstruct<-as.numeric(dta$struct>0)
#dta$bsub<-as.numeric(dta$subtopic>0)
dta$bnw<-as.numeric(dta$nonword>0)
dta$bliwc<-as.numeric(dta$liwc>0)
dta$bw2v<-as.numeric(dta$w2v>0)
dta$bkbest<-as.numeric(dta$kbest>0)
#
#dta$count<-dta$braw+dta$brawtopic+dta$bind+dta$bbeh+dta$bnw+dta$bliwc+dta$bw2v+dta$bkbest
#
#cats<-as.numeric()
#idx<-0
#tidx<-0
#tnames<-colnames(prop.table(table(dta$sex,dta$category),2))
#for (line in prop.table(table(dta$sex,dta$category),2)){
#    idx<-idx+1
#    if (idx %% 2 == 1){
#        tidx<-tidx+1
#        }
#    if (line>.7){
#        cats<-c(cats,tnames[tidx])
#    }
#}
#dta$sexist<-as.integer(dta$cat %in% cats)
#
#model<-glm(sex~braw+brawtopic+bbeh,data=dta,family=binomial)
#accuracies(dta,model)
#
#
#set.seed(415)
#fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize +
#                    FamilyID2, data=train, importance=TRUE, ntree=2000)



#########################################
##       PREDICTION MODEL
#########################################
#

library(caret)
form<-as.formula('sex~raw+ind+rawtopic+struct+subtopic+beh')

fitsvm<-function(dta,formula,num){
    library(e1071)
    library(gbm)
    ids<-sample(nrow(dta), num)
    train<- dta[ids, ]
    test<- dta[-ids, ]
    fit<-svm(formula,data=train)
    print(varImp(fit,scale=F))
    Prediction<-predict(fit,test)
    print(table(Prediction,test[['sex']])
    return(fit)
}
form<-as.formula('sex~raw+rawtopic+struct+subtopic+beh+nonwords')
sample<-500
fit<-fitsvm(dta,form,sample)


fitRF<-function(dta,formula,num){
    library(randomForest)
    ids<-sample(nrow(dta), num)
    train<- dta[ids, ]
    test<- dta[-ids, ]
    fit<-randomForest(formula,data=train,importance=T,ntree=1000,na.action=na.omit)
    varImpPlot(fit)
    Prediction <- predict(fit, test)
    print(table(Prediction,test[['sex']]))
    return(fit)
}

#form<-as.formula('sex~raw+rawtopic+struct+subtopic+beh+nonwords+w2v+liwc+kbest')
##form<-as.formula('sex~braw+brawtopic+bstruct+bsub+bbeh+bnw')
#sample<-1000
#fit<-fitRF(dta,form,sample)
#
#
#fitParty<-function(dta,formula,num){
#    library(party)
#    ids<-sample(nrow(dta), num)
#    train<- dta[ids, ]
#    test<- dta[-ids, ]
#    fit <- ctree(formula,data=train)
#    #fit<-randomForest(formula,data=train,importance=T,ntree=1000,na.action=na.omit)
#    #varImpPlot(fit)
#    Prediction <- predict(fit, test)
#    print(table(Prediction,test[['sex']]))
#    return(fit)
#}
#form<-as.formula('sex~braw+brawtopic+bstruct+bsub+bbeh')
#form<-as.formula('sex~raw+rawtopic+struct+subtopic+beh')
#sample<-1000
#fit<-fitParty(dta,form,sample)
#
#
#fitGBM<-function(dta,formula,num){
#    library(caret)
#    library(gbm)
#    ids<-sample(nrow(dta), num)
#    tr<- dta[ids, ]
#    test<- dta[-ids, ]
#    fit <- train(formula, data=tr, method="gbm", distribution="bernoulli") 
#    print(varImp(fit,scale=F))    #scale prevents normalization
#    Prediction <- predict(fit, test)
#    #print(Prediction)
#    print(table(Prediction,test[['sex']]))
#    return(fit)
#}
#
#form<-as.formula('as.factor(sex)~raw+rawtopic+struct+subtopic+beh')
#form<-as.formula('as.factor(sex)~braw+brawtopic+bstruct+bsub+bbeh')
#sample<-500
#fit<-fitGBM(dta,form,sample)
#
#
#fitADA<-function(dta,formula,include,num){
#    library(caret)
#    library(gbm)
#    wanted = colnames(dta) %in% include 
#    ids<-sample(nrow(dta), num)
#    tr<- dta[ids,wanted ]
#    test<- dta[-ids,wanted ]
#    fit <- train(formula, data=tr, method="ada")
#    #fit <- train(formula, data=tr, method="gbm", distribution="bernoulli") 
#    print(varImp(fit,scale=F))    #scale prevents normalization
#    #Prediction <- predict(fit, test)
#    #print(Prediction)
#    #print(table(Prediction,test[['sex']]))
#    return(fit)
#}
#
#form<-as.formula('sex~raw+rawtopic+struct+subtopic+beh')
#include<-c('sex','raw','rawtopic','struct','subtopic','beh')
#form<-as.formula('as.factor(sex)~braw+brawtopic+bstruct+bsub+bbeh')
#include<-c('sex','braw','brawtopic','bstruct','bsub','bbeh')
#
#sample<-500
#fit<-fitADA(dta,form,include,sample)
#



#########################################
##       FACTOR ANALYSIS MODEL
#########################################
#
# Summary: this section checks whether the variables that are supposed to be related to one
# antoher are correlated with one another.
# [raw, raw topic, kbest, and structure]
# [behavior and subtopic] 
#########################################
#

#################
##       CORRELATIONS
#################

cols=c(seq(6,length(names(dta))-1,1))   #the length-1 throws out "individual"
sdta=dta[,cols]
cor(sdta)

#################
##       CRONBACH'S ALPHA
#################
library(psych)

institutions<-c(2,3,7,8)    #raw, raw topic, w2v, kbest
alpha(sdta[,institutions])
interactions<-c(1,4,5)      #beh, struct, subtopic
alpha(sdta[,interactions])


#################
##       Principal Components
#################

f<-princomp(na.omit(sdta),cor=T)
f<-principal(na.omit(sdta),nfactors=5,rotate='varimax')
print(f)