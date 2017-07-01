import cPickle
import csv
import numpy as np

#print 'importing previous data'     #eligible.pkl contains all projects either completed or expired.
#f=open('C:\Users\jsradford\Desktop\Workspace\DonorsChoose\Data\Eligible.pkl','r')
#eligible=cPickle.load(f)
#f.close()
#
#eligibleProjects=[]
#for i in eligible:
#    eligibleProjects.append(i[0])

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


Data=[]
f=open('C:\DonorsChoose Archive\Original Data\Projects\Projects.csv', 'rb')
data = csv.reader(f, delimiter=',', quotechar='"')
for i,line in enumerate(data):
    if i!=0:
        if line[18]=='Mr.':
            male=1
        elif line[18]=='Ms.' or line[18]=='Mrs.':
            male=0
        else:
            continue        #skip projects w/o author gender
        if line[27]=='':
            continue        #skip projects w/o registered poverty level
        poverty=line[27]+'_poverty'
        #if line[0] in eligibleProjects:
        metro=line[9]
        if 'true' in line[12:18]:
            altSchool='altSchool'
        else:
            altSchool='normSchool'
        
        if line[22]=='Applied Learning' or line[24]=='Applied Learning':
            applied='applied'
        else:
            applied='applied_null'
        if line[22]=='Health & Sports' or line[24]=='Health & Sports':
            sports='sports'
        else:
            sports='sports_null'
        if line[22]=='History & Civics' or line[24]=='History & Civics':
            socsci='socsci'
        else:
            socsci='socsci_null'
        if line[22]=='Literacy & Language' or line[24]=='Literacy & Language':
            language='ela'
        else:
            language='ela_null'
        if line[22]=='Math & Science' or line[24]=='Math & Science':
            STEM='stem'
        else:
            STEM='stem_null'
        if line[22]=='Music & The Arts' or line[24]=='Music & The Arts':
            arts='arts'
        else:
            arts='arts_null'
        if line[22]=='Special Needs' or line[24]=='Special Needs':
            special='special'
        else:
            special='special_null'
        gradelevel=line[28]
        case=[line[0],male,metro,altSchool,applied, sports, socsci, language, STEM, arts, special, gradelevel,poverty]
        Data.append(case)

head=['projectid','male','metro','altSchool','applied','sports','socsci','language','STEM','arts','special','areamiss','gradelevel','poverty']
f.close()
with open('DonorsChoose\DonorsChooseClassData.csv','wb') as fn:
    writeit=csv.writer(fn,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    print 'writing it to csv'
    writeit.writerow(head)
    for v in Data:
        writeit.writerow(v)
fn.close()
