# Search Engine and Inverted List Compression

This project consists of 3 parts:

1. Web crawler
2. Inverted List creation and compression
3. Search Queries

## Prerequisites

In order to run the files, you will need to install the following packages in Python 2:

```
pip install bs4
pip install nltk
```

## Usage

### Web Crawler

```
crawler.py -u URL [-m MAXPAGES]
```

The parameter `URL` determines the seed for the web crawler and the optional parameter `MAXPAGES` limits the number of pages the crawler can access (default: infinite).
    
The crawler is independent from the other two files.


### Inverted File

For the creation of the inverted file we have used this corpus:

https://github.com/logicx24/Text-Search-Engine/tree/master/corpus

The creation and compression of the inverted file can be done by writing:
```
BuildIndex.py [-I INPUTDIR -O OUTPUTDIR]
```
    
Where the optional parameter `INPUTDIR` sets the directory of the corpus (default: books) and the optional parameter `OUTPUTDIR` sets the directory for the inverted file to be saved (default: same directory).
    

### Search Queries

Before using the search engine, the inverted file has to be created first by running the `BuildIndex.py` file.

Then, we can initiate the search engine by writing:
```
QueryIndex.py [-I INPUTFILE]
```

Where the optional parameter `INPUTFILE` sets the name of the inverted file (default: inverted_file.txt).

The user will be then asked to give queries. There are two types of queries supported:
- Standard Query: where at least one of the words in the query appears in the document
- Phrase Query: where all the words in the query appear in the document in the same order.

This implementation supports both queries by using double quotation marks.
That is, Phrase Query is applied to the quoted string(s) and Standard Query to the remaining words.

The top 10 results are shown to the user, along with their logarithmically scaled tf-idf scores.

## Examples

### Crawler
```
> python crawler.py -u http://www.example.com -m 2
# 0 processing:  http://www.example.com
        Found:
         http://www.iana.org/domains/example
# 1 processing:  http://www.iana.org/domains/example
        Found:
         http://www.iana.org/
         http://www.iana.org/domains
         http://www.iana.org/numbers
         [...]
DONE...
```

### Inverted List
```
> python BuildIndex.py
Processing: C:\Search-Engine\books\1399-0.txt (295.16 sec)
Processing: C:\Search-Engine\books\pg11.txt ( 27.82 sec)
Processing: C:\Search-Engine\books\pg1232.txt ( 39.08 sec)
Processing: C:\Search-Engine\books\pg1342.txt ( 99.49 sec)
Processing: C:\Search-Engine\books\pg135.txt (503.16 sec)
Processing: C:\Search-Engine\books\pg1661.txt (100.57 sec)
Processing: C:\Search-Engine\books\pg174.txt ( 70.94 sec)
Processing: C:\Search-Engine\books\pg30601.txt ( 74.74 sec)
Processing: C:\Search-Engine\books\pg5200.txt ( 20.08 sec)
Processing: C:\Search-Engine\books\pg74.txt ( 73.99 sec)
Processing: C:\Search-Engine\books\pg76.txt ( 96.97 sec)
Processing: C:\Search-Engine\books\pg98.txt (127.02 sec)
Total time:1537.10 sec
```

### Search Query
```
> python QueryIndex.py

Give your queries.
Press ctrl-c for exit.
 > "full of joy" "robin hood" alice
Querying:    8 : pg76,pg74,1399-0,pg11,pg1661,pg174,pg135,pg1342

 1                 pg74       2.24
 2                 pg11       2.21
 3                pg135       1.78
 4               pg1661       1.59
 5                pg174       1.14
 6               1399-0       0.54
 7               pg1342       0.08
 8                 pg76       0.07
```
