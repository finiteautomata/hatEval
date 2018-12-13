from csv import DictReader


class CorpusReader:
    
    def __init__(self, filename):
        self.filename = filename

        file = open(filename)
        reader = DictReader(file, delimiter='\t')
        self.entries = list(reader)

    def X(self):
        for e in self.entries:
            yield e['text']
    
    def y(self):
        for e in self.entries:
            yield e['HS']
