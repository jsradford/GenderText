import tarfile
from contextlib import closing
import gzip

tar='C:/Data/GenderText/Datasets/OpenLibrary/ol_dump_authors_2016-02-29.txt.gz'
noauthors=[]
raw=[]
metadata={}
texts={}
with gzip.open(tar, mode="r") as f:
#    for f in archive:
#        with closing(archive.extractfile(f)) as data:
    for i,line in enumerate(f.readlines()):
        if i<5:
            raw.append(line)
        else:
            with open('C:/Data/GenderText/Datasets/OpenLibrary/authors5.txt','w') as f:
                for line in raw:
                    f.write(line)
            quit()
            #continue

#print i, ' total number of authors'

