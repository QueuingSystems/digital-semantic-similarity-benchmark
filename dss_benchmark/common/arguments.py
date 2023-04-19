import re

__all__ = ["parse_arbitrary_arguments"]


def parse_arbitrary_arguments(args):
    if len(args) == 0:
        return {}

    result = {}
    args = list(args)
    while len(args) > 0:
        arg = args.pop(0)

        if arg.startswith("--"):
            arg = arg[2:]
        else:
            raise ValueError("Invalid argument: {}".format(arg))

        value = None
        if re.match(r"^[\w\-_]+=", arg):
            arg, value = arg.split("=")
        else:
            if len(args) == 0:
                raise ValueError("Invalid argument: {}".format(arg))
            value = args.pop(0)

        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass

        if value == 'true':
            value = True
        elif value == 'false':
            value = False
        result[arg] = value
    return result
