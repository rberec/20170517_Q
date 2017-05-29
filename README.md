# 20170517_Q

```
wget https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh
bash Anaconda3-4.3.1-Linux-x86_64.sh

conda install py-xgboost
conda install gensim
conda install tqdm
conda install -c conda-forge fuzzywuzzy
conda install -c conda-forge pyemd
```

https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question

Download data
```
mkdir input
cd input
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
python -m nltk.downloader stopwords
cd ..
```

Download nltk data
```
echo "import nltk;nltk.download('all')" | python
```

Generate features
```
nohup python feature_engineering.py &
nohup python feature_engineering_test.py &
```
