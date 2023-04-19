import dataclasses

import tabulate

__all__ = ["print_dataclass"]


def print_dataclass(dataclass_):
    fields = dataclasses.fields(dataclass_)
    params = []
    for field in fields:
        name = field.name
        type = field.type
        default = field.default
        if default == dataclasses.MISSING:
            default = "MISSING"
        help = field.metadata.get("help", "-")
        params.append([name, type, default, help])
    print(tabulate.tabulate(params, headers=["name", "type", "default", "help"]))
