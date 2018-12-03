TRIAL_DIR="data/trial/"
DATA_DIR="data"


SPANISH_PASS="kNHJ6IxG3t6BSXT0"
ENGLISH_PASS="aLlTm9faipJpbKsg"
URL_SPANISH_A="https://competitions.codalab.org/my/datasets/download/b75028db-dc0c-4f81-bc3c-abf7a54d9cf3"
URL_ENGLISH_A="https://competitions.codalab.org/my/datasets/download/70f24fdb-2df2-4a9c-9bec-2ec00e2f5554"
echo "Getting trial data"

wget https://raw.githubusercontent.com/msang/hateval/master/SemEval2019-Task5/datasets/trial/trial_en.tsv -P $TRIAL_DIR
wget https://raw.githubusercontent.com/msang/hateval/master/SemEval2019-Task5/datasets/trial/trial_es.tsv -P $TRIAL_DIR

echo "Getting development data"

#Spanish A

SPANISH_A_ZIP="$DATA_DIR/dev_spanish_a.zip" 
ENGLISH_A_ZIP="$DATA_DIR/dev_english_a.zip"

wget $URL_SPANISH_A -O $SPANISH_A_ZIP
wget $URL_ENGLISH_A -O $ENGLISH_A_ZIP

7z x $SPANISH_A_ZIP -o$DATA_DIR -p$SPANISH_PASS
7z x $ENGLISH_A_ZIP -o$DATA_DIR -p$ENGLISH_PASS

mv $DATA_DIR/public_development_es $DATA_DIR/dev_es
mv $DATA_DIR/public_development_en $DATA_DIR/dev_en

rm $SPANISH_A_ZIP
rm $ENGLISH_A_ZIP

#wget https://competitions.codalab.org/my/datasets/download/70f24fdb-2df2-4a9c-9bec-2ec00e2f5554 -O "$DATA_DIR/dev_english_a.zip"
