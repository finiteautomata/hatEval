from hate.corpus import CorpusReader

corpora = {
    'trial_en': CorpusReader('HATEVAL/public_trial/trial_en.tsv'),
    'trial_es': CorpusReader('HATEVAL/public_trial/trial_en.tsv'),
    'train_en': CorpusReader('HATEVAL/A/public_development_en/train_en.tsv'),
    'dev_en': CorpusReader('HATEVAL/A/public_development_en/dev_en.tsv'),
    'train_es': CorpusReader('HATEVAL/A/public_development_es/train_es.tsv'),
    'dev_es': CorpusReader('HATEVAL/A/public_development_es/dev_es.tsv'),
}
