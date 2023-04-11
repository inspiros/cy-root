from typing import Sequence, Union

import numpy as np

__all__ = [
    'ArrayLike',
]

Real = Union[int, float]
sr1 = Sequence[Real]
sr2 = Sequence[sr1]
sr3 = Sequence[sr2]
sr4 = Sequence[sr3]

SequenceNDReals = Union[sr1, sr2, sr3, sr4]

ArrayLike = Union[np.ndarray, SequenceNDReals]
