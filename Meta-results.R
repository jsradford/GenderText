avg<-c(0.696,0.646,0.662,0.714,0.586,0.6325,0.704,0.62,0.62,0.885)
names<-c('raw','rawtopic','w2v','kbest','liwc','nonwrd','behav.','struct.','subtop.','indiv.')
jpeg('Model_Average_Accuracy.jpg')
barplot(avg, main="Average Accuracy",ylim=c(0,1),xlab='Model',ylab='Accuracy', cex.names=.6,
  names.arg=names)
dev.off()

    
rank<-c(2.8,4.8,5.0,2.6,7.4,7.2,3.4,6.8,6.8,6.2)
names<-c('raw','rawtopic','w2v','kbest','liwc','nonwrd','behav.','struct.','subtop.','indiv.')
jpeg('Model_Average_Rank.jpg')
barplot(rank, main="Models by their Average Rank",ylim=c(0,8),xlab='Model',ylab='Average Rank', cex.names=.6,
  names.arg=names)
dev.off()
