"""timing.placement: data_ptr classes of the buffers a timed call touches, ab placement warning."""
from __future__ import annotations

import types

from kdev.timing import placement as P


class FakeT:
    is_cuda = True

    def __init__(self, ptr: int, n: int) -> None:
        self._p, self._n = ptr, n

    def data_ptr(self) -> int:
        return self._p

    def numel(self) -> int:
        return self._n

    def element_size(self) -> int:
        return 2


def test_describe_classes_and_walk():
    mod = types.ModuleType("sub")
    mod.W = FakeT(4 << 20, 1 << 20)             # 2 MiB aligned -> A
    mod.torch = types                           # imported module: not walked
    state = {"w2": FakeT((4 << 20) + (64 << 10), 1 << 19), "lst": [FakeT((8 << 20) + 256, 10)]}
    named = P.tensors_in(state, "state") + P.tensors_in(mod, "", depth=2)
    d = P.describe(named)
    assert d["class"] == "ABC" and d["tensors"][0]["name"] == "W"
    assert d["tensors"][1]["ptr_mod_2m"] == 64 << 10 and d["tensors"][2]["ptr_mod_64k"] == 256


def test_compare_warns_on_class_difference():
    procs = [("A", {"weights_class": "AA", "outputs_class": "A"}), ("B", {"weights_class": "AB", "outputs_class": "A"}),
             ("B", {"weights_class": "AB", "outputs_class": "A"}), ("A", {"weights_class": "AA", "outputs_class": "A"})]
    w = P.compare(procs, ["A", "B"])
    assert len(w) == 1 and "weights class differs" in w[0]
    assert P.compare([(lb, {"weights_class": "A", "outputs_class": "B"}) for lb in "ABBA"], ["A", "B"]) == []
