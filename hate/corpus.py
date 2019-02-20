from csv import DictReader, QUOTE_NONE


class CorpusReader:
    
    def __init__(self, filename, header=True):
        self.filename = filename

        file = open(filename)
        if not header:
            fns = 'id text HS TR AG'.split()
        else:
            fns = None
        reader = DictReader(file, fieldnames=fns, delimiter='\t', quoting=QUOTE_NONE)
        self.entries = list(reader)

    def X(self):
        for e in self.entries:
            yield e['text']
    
    def y(self):
        for e in self.entries:
            yield e['HS']

    def Xy(self):
        return list(self.X()), list(self.y())
