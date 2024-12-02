from __future__ import annotations

import pickle
import threading
from abc import ABC, abstractmethod
from typing import Optional, Union

import semlaflow.util.functional as smolF

indicesT = Union[list[int], list[list[int]]]


PICKLE_PROTOCOL = 4


# *** Util functions ***


def _check_unique(obj_list, name="objects"):
    if len(obj_list) != len(set(obj_list)):
        raise RuntimeError(f"{name} cannot contain duplicates")


def _check_type_all(obj_list, exp_type, name="list"):
    for obj in obj_list:
        if not isinstance(obj, exp_type):
            raise TypeError(f"all objects in {name} must be instances of {exp_type}")


# *** Tokeniser Interface ***


class Tokeniser(ABC):
    """Interface for tokeniser classes"""

    @abstractmethod
    def tokenise(self, sentences: list[str]) -> Union[list[str], list[int]]:
        pass

    @classmethod
    @abstractmethod
    def from_vocabulary(cls, vocab: Vocabulary) -> Tokeniser:
        pass


# *** Tokeniser Implementations ***

# TODO


# *** Vocabulary Implementations ***


class Vocabulary:
    """Vocabulary class which maps tokens <--> indices"""

    def __init__(self, tokens: list[str]):
        _check_unique(tokens, "tokens list")

        token_idx_map = {token: idx for idx, token in enumerate(tokens)}
        idx_token_map = {idx: token for idx, token in enumerate(tokens)}

        self.token_idx_map = token_idx_map
        self.idx_token_map = idx_token_map

        # Just to be certain that vocab objects are thread safe
        self._vocab_lock = threading.Lock()

        # So that we can save this object without assuming the above dictionaries are ordered
        self._tokens = tokens

    @property
    def size(self) -> int:
        return len(self)

    def __len__(self) -> int:
        with self._vocab_lock:
            length = len(self.token_idx_map)

        return length

    def contains(self, token: str) -> bool:
        with self._vocab_lock:
            contains = token in self.token_idx_map

        return contains

    def tokens_from_indices(self, indices: list[int]) -> list[str]:
        _check_type_all(indices, int, "indices list")
        with self._vocab_lock:
            tokens = [self.idx_token_map[idx] for idx in indices]

        return tokens

    def indices_from_tokens(self, tokens: list[str], one_hot: Optional[bool] = False) -> indicesT:
        _check_type_all(tokens, str, "tokens list")

        with self._vocab_lock:
            indices = [self.token_idx_map[token] for token in tokens]

        if not one_hot:
            return indices

        one_hots = smolF.one_hot_encode(indices, len(self)).tolist()
        return one_hots

    def to_bytes(self) -> bytes:
        with self._vocab_lock:
            obj_bytes = pickle.dumps(self._tokens, protocol=PICKLE_PROTOCOL)

        return obj_bytes

    @staticmethod
    def from_bytes(data: bytes) -> Vocabulary:
        tokens = pickle.loads(data)
        return Vocabulary(tokens)


# class AtomVocabulary(Vocabulary):
#     """Vocabulary which only uses atoms and allows converting from atomic numbers"""

#     # TODO add more atoms?
#     ATOMIC_MAP = {
#         1: "H",
#         6: "C",
#         7: "N",
#         8: "O",
#         9: "F",
#         15: "P",
#         16: "S",
#         17: "Cl",
#         35: "Br"
#     }

#     def __init__(self, extra_atom_map: dict[int, str]):
#         all_atomics = list(AtomVocabulary.ATOMIC_MAP.keys()) + list(extra_atom_map.keys())
#         all_tokens = list(AtomVocabulary.ATOMIC_MAP.values()) + list(extra_atom_map.values())

#         _check_unique(all_atomics, "extended atomic numbers list")
#         _check_unique(all_tokens, "extended tokens list")

#         extended_map = dict(list(AtomVocabulary.ATOMIC_MAP.items()) + list(extra_atom_map.items()))

#         atomic_tokens = list(extended_map.items())
#         sorted_atomic_tokens = sorted(atomic_tokens, key=lambda a_t: a_t[0])
#         sorted_atomics, sorted_tokens = tuple(zip(*sorted_atomic_tokens))

#         atomic_token_map = dict(zip(sorted_atomics, sorted_tokens))
#         token_atomic_map = dict(zip(sorted_tokens, sorted_atomics))

#         self.atomic_token_map = atomic_token_map
#         self.token_atomic_map = token_atomic_map

#         super().__init__(sorted_tokens)

#     def tokens_from_atomic(self, atomics: list[int]) -> list[str]:
#         _check_type_all(atomics, int, "atomic numbers list")

#         with self._vocab_lock:
#             tokens = [self.atomic_token_map[atom] for atom in atomics]

#         return tokens

#     def atomic_from_tokens(self, tokens: list[str]) -> list[int]:
#         _check_type_all(tokens, str, "tokens list")

#         with self._vocab_lock:
#             atomics = [self.token_atomic_map[token] for token in tokens]

#         return atomics

#     def indices_from_atomic(self, atomics: list[int], one_hot: Optional[bool] = False) -> indicesT:
#         tokens = self.tokens_from_atomic(atomics)
#         indices = self.indices_from_tokens(tokens, one_hot=one_hot)
#         return indices

#     def atomic_from_indices(self, indices: list[int]) -> list[int]:
#         tokens = self.tokens_from_indices(indices)
#         atomics = self.atomic_from_tokens(tokens)
#         return atomics

#     def to_bytes(self) -> bytes:
#         with self._vocab_lock:
#             obj_bytes = pickle.dumps(self.atomic_token_map, protocol=PICKLE_PROTOCOL)

#         return obj_bytes

#     @staticmethod
#     def from_bytes(data: bytes) -> AtomVocabulary:
#         full_map = pickle.loads(data)
#         atomic_nums = set(AtomVocabulary.ATOMIC_MAP.keys())
#         extra_map = {atom: token for atom, token in full_map if atom not in atomic_nums}
#         return AtomVocabulary(extra_map)
