# hatEval

This repository is the code of the proceeding ["Atalaya at SemEval 2019 task 5: Robust embeddings for tweet classification." ](https://www.aclweb.org/anthology/S19-2008.pdf)

# Instructions

1. Get data

```
./bin/get_data.sh
```

2. Get Elmo

```
./bin/get_elmo.sh
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Install this package in editable mode

```
pip install -e .
```

5. Run tests
```
python setup.py test
```
