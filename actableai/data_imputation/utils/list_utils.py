import itertools
from typing import TypeVar, List, Set

T = TypeVar("T")


def all_possible_pairs(*list_values: Set[T]) -> List[List[T]]:
    return list(map(list, itertools.product(*list_values)))
