# GenderText

## About
This is a code repository for my current research building generic modelling strategies for identifying gender across corpora.
The data are available upon request where needed (see below). It is not meant to be downloaded and run, but to be borrowed from 
(how do you implement LDA in python?) and for other researchers to keep abreast of and perhaps contribute to this project as 
it develops. The repository is structured as follows:

## Coming Soon...
The biggest change I plan to make is the set of features I pull out representing gendered behavior (aka "doing gender"). I've implemented the theory incorrectly. The current feature selection strategies approach gender as a series of nested structures (i.e. people writing within gender-segregated subfields within further gender-segregated fields). Gender system theory suggests instead that there is a single-layered structure with gendered behavior cutting across it (i.e. people doing gendered things within gender-segregated fields). 

## Code
The **Generic** folder is contains the basic set of scripts I pull from for each analysis in each corpus. Each folder corresponds to a different corpus being used. Within each folder are localized versions of the **Generic** scripts adapted to the particularities of that corpus 

The remaining folders contain three corpus-specific scripts that clean the data and generate the feature sets (Make_[corpus]_data.py], train the classification models based on the feature sets (analyze_[corpus].py), and then run prediction and reporting ([corpus]_estimate.R).

## Data Sources
The data for this project are gathered from a variety of corpora containing text tagged with the gender of its author or speaker.

- **Abstracts**: Under preparation, the data [here](https://www.cs.cornell.edu/projects/kddcup/datasets.html) are the abstracts 
that were part of the KDD Cup for 2003.
- **Blogger**: This is a [dataset]() of posts from 19K bloggers at Blogger.
- **Brown**: This is the standard Brown corpus that comes with NLTK.
- **DonorsChoose**: This is the basis for my original study in this area. Updated data can be found at [data.DonorsChoose]
(https://data.donorschoose.org/)
- **IMBD**: The Movie Dialogue corpus [available here](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) 
I look to classify the gender of the speaker.
- **OpenLibrary**: This is a [corpus of books](https://archive.org/details/ol_exports&tab=collection) provided by the Internet 
Archive's [Open Library](https://openlibrary.org/). It's some 1GB zipped (6.9 million authors), so I haven't worked on it yet.
- **Reuters**: This is another classic corpus of news articles 
[available](https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html) as an old KDD Cup
- **Twitter**: This is a homebrew corpus of the tweets of all Members of the U.S. Congress from 2011-2013. Just ask if 
you want this data.
