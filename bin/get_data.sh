TRIAL_DIR="data/trial/"

echo "Getting trial data"

wget https://raw.githubusercontent.com/msang/hateval/master/SemEval2019-Task5/datasets/trial/trial_en.tsv -P $TRIAL_DIR
wget https://raw.githubusercontent.com/msang/hateval/master/SemEval2019-Task5/datasets/trial/trial_es.tsv -P $TRIAL_DIR
