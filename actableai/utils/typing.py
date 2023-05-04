from typing import Union, List, Dict

JSONType = Union[None, int, str, bool, List["JSONType"], Dict[str, "JSONType"]]
