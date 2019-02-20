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
    corpora_root = '/home/francolq/hatEval/data'
    corpora = {
        'trial_en': CorpusReader(join(corpora_root, 'trial/trial_en.tsv')),
        'trial_es': CorpusReader(join(corpora_root, 'trial/trial_es.tsv')),
        'train_en': CorpusReader(join(corpora_root, 'en/train_en.tsv')),
        'dev_en': CorpusReader(join(corpora_root, 'en/dev_en.tsv')),
        'train_es': CorpusReader(join(corpora_root, 'es/public_development_es/train_es.tsv')),
        'dev_es': CorpusReader(join(corpora_root, 'es/public_development_es/dev_es.tsv')),
        'test_en': CorpusReader(join(corpora_root, 'en/reference_en.tsv'), header=False),
        'test_es': CorpusReader(join(corpora_root, 'es/reference_es.tsv'), header=False),
    }
except:
    corpora = {}
