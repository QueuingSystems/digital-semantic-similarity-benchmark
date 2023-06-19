
class ParamsParser:
    def __init__(self, filename):
        self._file = open(filename, "r")

    def read(self, read_from=1):
        d = {}
        for line in self._file.readlines()[read_from:]:
            l = line.strip()
            if l != '':
                key = l
                value = [int(i) if i.isdigit() else i for i in l.split('-')]
                d[key] = value
        return d.items()
