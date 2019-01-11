from csv import DictReader, QUOTE_NONE


class CorpusReader:
    
    def __init__(self, filename):
        self.filename = filename

        file = open(filename)
        reader = DictReader(file, delimiter='\t', quoting=QUOTE_NONE)
        self.entries = list(reader)

    def X(self):
        for e in self.entries:
            yield e['text']
    
    def y(self):
        for e in self.entries:
            yield e['HS']
