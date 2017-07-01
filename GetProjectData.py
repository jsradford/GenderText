import cPickle
import csv
import numpy as np

print 'importing previous data'     #eligible.pkl contains all projects either completed or expired.
f=open('C:\Users\jsradford\Desktop\Workspace\DonorsChoose\Data\Eligible.pkl','r')
eligible=cPickle.load(f)
f.close()

eligibleProjects=[]
for i in eligible:
    eligibleProjects.append(i[0])

#f=open('C:\Users\jsradford\Desktop\Workspace\DonorsChoose\BAD.pkl','r')
#bad=cPickle.load(f)
#f.close()
#bads=[]
#for i in bad:
#    bads.append(i[0])

#f=open('C:\Users\jsradford\Desktop\Workspace\DonorsChoose\Scripts\existingCases.pkl','r')
#existingCases=cPickle.load(f)
#f.close()
#
#eligibleProjects=[]
#for i in eligibles:
#    if i not in bads and i not in existingCases:
#        eligibleProjects.append(i)

examine=[]
Years=range(2002,2012)
for year in Years:
    print 'calculating project data for %d' % year
    newCases=set()
    Syear=str(year)
    Data=[]
    f=open('C:\\Users\jsradford\Desktop\Workspace\DonorsChoose\Data\Projects\Projects.csv', 'rb')
    data = csv.reader(f, delimiter=',', quotechar='"')
    for i,line in enumerate(data):
        if i!=0 and line[42].split('-')[0]==Syear:
            if line[0] in eligibleProjects:
                newCases.update([line[0]])
                state =line[7]
                zip =line[8]
                if len(line[8])>3:
                    shortzip=line[8][0:3]
                else:
                    shortzip=''
                if line[9]=='rural':
                    rural=1
                else:
                    rural=0
                if line[9]=='suburban':
                    suburban=1
                else:
                    suburban=0
                if line[9]=='urban':
                    urban=1
                else:
                    urban=0
                charter=line[12]
                magnet=line[13]
                YearRound=line[14]
                nlns=line[15]
                kipp=line[16]
                promise=line[17]
                if 'true' in line[12:18]:
                    altSchool=1
                else:
                    altSchool=0
                if line[18]=='Mr.':
                    male =1
                elif line[18]=='Ms.' or line[18]=='Mrs.':
                    male=0
                else:
                    male=''
                subject1=line[21] #longer, more fine grained list of subjects. The area variable is a DonorsChoose grouping of these subjects into a smaller set
                subject2=line[23]
                area1=line[22] #applied learning, health and sports, history and civics, literacy and language, math & science, music & the arts, special needs (capitalized)
                area2=line[24]
                if line[22]=='Applied Learning' or line[24]=='Applied Learning':
                    applied=1
                else:
                    applied=0
                if line[22]=='Health & Sports' or line[24]=='Health & Sports':
                    sports=1
                else:
                    sports=0
                if line[22]=='History & Civics' or line[24]=='History & Civics':
                    socsci=1
                else:
                    socsci=0
                if line[22]=='Literacy & Language' or line[24]=='Literacy & Language':
                    language=1
                else:
                    language=0
                if line[22]=='Math & Science' or line[24]=='Math & Science':
                    STEM=1
                else:
                    STEM=0
                if line[22]=='Music & The Arts' or line[24]=='Music & The Arts':
                    arts=1
                else:
                    arts=0
                if line[22]=='Special Needs' or line[24]=='Special Needs':
                    special=1
                else:
                    special=0
                if line[22]=='' and line[24]=='':
                    areamiss=1
                else:
                    areamiss=0
                if line[25]== 'essential':   #essential vs. enrichment
                    usage=1
                else:
                    usage=0
                if line[26]=='Books':
                    resbook=1
                else:
                    resbook=0
                if line[26]=='Technology':
                    restech=1
                else:
                    restech=0
                if line[26]=='Visitors':
                    resvis=1
                else:
                    resvis=0
                if line[26]=='Supplies':
                    ressupplies=1
                else:
                    ressupplies=0
                if line[26]=='Trips':
                    restrips=1
                else:
                    restrips=0
                if line[26]=='Other' or line[26]=='':
                    resother=1
                else:
                    resother=0
                restype=line[26] #books, technology, visitors, supplies, trips, others (capitalized)
                if line[27]=='minimal':
                    poverty=0
                elif line[27]=='low':
                    poverty=1
                elif line[27]=='high':
                    poverty=2
                else:
                    poverty=''
                if line[27]=='unknown':
                    povertymiss=1
                else:
                    povertymiss=0
                #poverty=line[27] #minimal low high unknown
                if line[28]=='Grades PreK-2':
                    gradeK=1
                else:
                    gradeK=0
                if line[28]=='Grades 3-5':
                    gradeElem=1
                else:
                    gradeElem=0
                if line[28]=='Grades 6-8':
                    gradeMid=1
                else:
                    gradeMid=0
                if line[28]=='Grades 9-12':
                    gradeHigh=1
                else:
                    gradeHigh=0
                gradelevel=line[28] #<2, 3-5, 6-8,9-12
                price=line[33]
                price2=line[34]
                reach=line[35]
                if line[36]=='true':
                    future=1
                else:
                    future=0
                if line[37]!='':
                    TotDonations=line[37]
                else:
                    TotDonations=0
                if line[38] != '':
                    NumDonors=line[38]
                else:
                    NumDonors=0
                DoubleMatch=line[39]
                HomeMatch=line[40]
                if line[39] =='true' or line[40]=='true':
                    Match=1
                else:
                    Match=0        
                ###The following calculations create the Fundrate variable, which was ultimately shown to be unpredictable.  Some missing and corrupt date values
                ###required several conditional loops to clean them out.
                year1=line[42].split('-')
                Year=year1[0]
                MonthPosted=year1[1]
                if line[41]=='completed':
                    Funded=1
                    year2=line[43].split('-')
                    TotTime=int(year2[1])-int(year1[1])+12*(int(year2[0])-int(year1[0]))+1  #measures tottime in months
                elif line[41]=='expired':
                    Funded=0
                    if line[45]!='':
                        year2=line[45].split('-')
                        TotTime=int(year2[1])-int(year1[1])+12*(int(year2[0])-int(year1[0]))+1 #plus 1 at end is to adjust for one month completion = 0.
                    if line[45]=='':
                        TotTime=''
                        if NumDonors!='':
                            FundRate=0
                        else:
                            FundRate=''
                if TotTime>=1 and TotTime!='' and TotDonations!='':
                    FundRate=float(TotDonations)/float(TotTime)
                elif TotTime==0 or TotTime=='':
                    FundRate=''
                    examine.append([line[0],line[37:46]])
                case=[line[0],line[1],line[2], state, zip, shortzip, rural, suburban, urban, charter, magnet, YearRound, nlns, kipp, promise, altSchool, male,
                          subject1, subject2, area1, area2, applied, sports, socsci,language, STEM, arts, special, areamiss, usage, resbook,
                          restech, resvis, ressupplies, restrips, resother, poverty, povertymiss, gradeK,gradeElem, gradeMid, gradeHigh, price, price2, reach, future,
                          TotDonations, NumDonors, DoubleMatch, HomeMatch, Match, Funded, Year, MonthPosted, TotTime, FundRate]
                for i,x in enumerate(case):
                    if x=='true':
                        case[i]=1
                    if x=='false':
                        case[i]=0
                Data.append(case)
    f.close()
    
    print '%d cases in %d' % (len(Data),year)

    print 'calculating donation data for %d' % year
    f=open('C:\\Users\jsradford\Desktop\Workspace\DonorsChoose\Data\donations\donations.csv', 'rb')
    data = csv.reader(f, delimiter=',', quotechar='"')
    
    #creates an indexed dictionary for pulling out the cases that in the random samples
    database={}
    for index,line in enumerate(data):
        if line[1] in newCases:
            projectid=line[1]
            database[projectid]=[]
    f.close()
    f=open('C:\\Users\jsradford\Desktop\Workspace\DonorsChoose\Data\donations\donations.csv', 'rb')
    data = csv.reader(f, delimiter=',', quotechar='"')
    for index,line in enumerate(data):
        if line[1] in newCases:
            ### pulls out raw variables to be re-computed in the next for loop.
            projectid=line[1]
            givingpage=line[15]
            donorstate=line[5]
            if line[9]=='under_10':
                amount=0
            if line[9]=='10_to_100':
                amount=1
            if line[9]=='100_and_up':
                amount=2
            if line[13] == 'true' or line[14] == 'true':
                if line[11]=='no_cash_received':
                    giftcard=1
            else:
                giftcard=0
            teacherdonor=line[7]
            case=[projectid, donorstate,amount,giftcard,teacherdonor,givingpage]
            for i,x in enumerate(case):
                if x=='true':
                    case[i]=1
                if x=='false':
                    case[i]=0
            
            database[projectid].append(case[1:len(case)])
    f.close()
    
    Donations=[]
    for ids,lis in database.iteritems():
        giftwork=[]
        #statework=[]
        amountwork=[]
        #teacherwork=[]
        givpage=[]
        for case in lis:
            giftwork.append(case[2])
            amountwork.append(case[1])
            #teacherwork.append(case[3])
            givpage.append(case[4])
        giftcards=0
        for i in giftwork:
            if i==1:
                giftcards+=1
        percentgiftcards=float(giftcards)/float(len(lis))
        perGivePage=float(np.sum(givpage))/float(len(lis))
        if perGivePage>0:
            givePage=1
        else:
            givePage=0
        low=0
        mid=0
        high=0
        for am in amountwork:
            if am==0:
                low+=1
            if am==1:
                mid+=1
            if am==2:
                high+=1
        percenthigh=float(high)/float(len(lis))
        percentunder100=float(low+mid)/float(len(lis))
        Donations.append([ids,percentgiftcards,percenthigh,percentunder100,perGivePage,givePage])
        ###To note, the percenthigh, 100, and givepage, were ultimately not used in the analysis.
    
    nums=range(1,6)
    for j in Data:
        num=range(len(j),len(j)+5)
    for j in Data:
        for n in nums:
            j.append(0)
    for i in Donations:
        for j in Data:
            if i[0]==j[0]:
                for id,n in enumerate(nums):
                    for d,nu in enumerate(num):
                        if id==d:
                            j[nu]=i[n]
    
    print 'writing new data for %d' % year
    
    header=['projectid', 'teacherid', 'schoolid', 'state', 'zip', 'shortzip', 'rural', 'suburban', 'urban', 'charter', 'magnet', 'YearRound', 'nlns', 'kipp', 'promise', 'altSchool',
                      'male', 'subject1', 'subject2', 'area1', 'area2', 'applied', 'sports', 'socsci', 'language', 'STEM', 'arts', 'special', 'areamiss', 'usage',
                      'resbooks', 'restech', 'resvis', 'ressupplies', 'restrips', 'resother', 'poverty', 'povertymiss', 'gradeK','gradeElem', 'gradeMid', 'gradeHigh',
                      'price', 'price2', 'reach', 'future', 'TotDonations', 'NumDonors', 'DoubleMatch', 'HomeMatch', 'Match', 'Funded', 'Year','MonthPosted',
                      'TotTime','FundRate','PercentGiftCards','PercentHigh','PercentUnder100', 'perGivePage','givePage']
    fn=open('DataRerun'+Syear+'.csv','wb')
    writeit=csv.writer(fn,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    print 'writing it to csv'
    writeit.writerow(header)
    for v in Data:
        writeit.writerow(v)
    fn.close()
    
        
    fn=open("DataRerun"+Syear+'.pkl','wb')
    cPickle.dump(Data,fn)
    fn.close()
    