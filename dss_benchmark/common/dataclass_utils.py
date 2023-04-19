import dataclasses

import tabulate

__all__ = ["print_dataclass"]


def print_dataclass(dataclass_):
    fields = dataclasses.fields(dataclass_)
    has_values = False
    try:
        values = dataclasses.asdict(dataclass_)
        has_values = True
    except TypeError:
        values = {}

    params = []
    for field in fields:
        name = field.name
        type = field.type
        default = field.default
        if default == dataclasses.MISSING:
            default = "MISSING"
        help = field.metadata.get("help", "-")

        row = {
            "name": name,
            "type": type,
            # "default": str(default),
            "help": help,
        }
        if has_values:
            row["value"] = str(values[name])
        else:
            row["default"] = str(default)
        params.append(row)
    print(
        tabulate.tabulate(
            params, tablefmt="simple_outline", headers="keys", disable_numparse=True
        )
    )
