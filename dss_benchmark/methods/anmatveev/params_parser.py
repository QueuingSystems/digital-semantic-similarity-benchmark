
class ParamsParser:
    def __init__(self, filename=None):
        if filename:
            self._file = open(filename, "r")

    def read(self, read_from=1, split_into_groups=False):
        if split_into_groups:
            d = {}
        else:
            d = []
        i = 1
        for line in self._file.readlines()[read_from:]:
            l = line.strip()
            if l != '':
                key = l
                value = [int(i) if i.isdigit() else i for i in l.split('-')]
                if split_into_groups:
                    d.setdefault(str(i), []).append((key, value))
                else:
                    d.append((key, value))
            else:
                i += 1

        return d

    def read_one(self, model_name):
        l = model_name.split('-')
        for i in range(1, len(l)):
            l[i] = int(l[i])
        return tuple(l)

