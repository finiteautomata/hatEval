from os.path import join

from hate.corpus import CorpusReader


# Tree tagger
tree_tagger_path = '/home/francolq/tass2018/tree-tagger'
tree_tagger_params_path ={
    # http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/spanish-ancora.par.gz
    'es': join(tree_tagger_path, 'spanish-ancora-par-linux-3.2-utf8.bin'),
    # http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english.par.gz
    'en': join(tree_tagger_path, 'english.par'),
}

try:
    corpora_root = '/home/francolq/hatEval/HATEVAL'
    corpora = {
        'trial_en': CorpusReader(join(corpora_root, 'public_trial/trial_en.tsv')),
        'trial_es': CorpusReader(join(corpora_root, 'public_trial/trial_es.tsv')),
        'train_en': CorpusReader(join(corpora_root, 'A/public_development_en/train_en.tsv')),
        'dev_en': CorpusReader(join(corpora_root, 'A/public_development_en/dev_en.tsv')),
        'train_es': CorpusReader(join(corpora_root, 'A/public_development_es/train_es.tsv')),
        'dev_es': CorpusReader(join(corpora_root, 'A/public_development_es/dev_es.tsv')),
        'test_en': CorpusReader(join(corpora_root, 'public_test_en/test_en.tsv')),
        'test_es': CorpusReader(join(corpora_root, 'public_test_es/test_es.tsv')),
    }
except:
    corpora = {}
