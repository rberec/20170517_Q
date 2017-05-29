# 20170517_Q

https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question

Download data
```
mkdir input
cd input
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
python -m nltk.downloader stopwords
cd ..
```


Generate features
```
python feature_engineering.py
python feature_engineering_test.py
```
