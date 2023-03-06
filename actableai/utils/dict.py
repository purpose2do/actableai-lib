from typing import Any, Dict, Union, List


def to_dict_recursive(obj: Any) -> Union[Dict, List]:
    """
    TODO write documentation
    """
    if hasattr(obj, "__dict__"):
        return dict(obj)
    if isinstance(obj, dict):
        return {k: to_dict_recursive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict_recursive(e) for e in obj]

    return obj
