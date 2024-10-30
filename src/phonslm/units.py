import numpy as np

from .data import ALIGNMENT_FREQ, MODEL_FREQ


def _build_count_matrix(phones: dict[str, list[str]], units: dict[str, list[int]]) -> tuple[np.ndarray, list[str]]:
    num_phones = len(np.unique([p for phone in phones.values() for p in phone]))
    num_units = max([max(unit) for unit in units.values()]) + 1
    count = np.zeros((num_phones, num_units), dtype=int)

    dictionnary, idx = {}, 0
    for fileid, unit in units.items():
        unit = np.repeat(unit, ALIGNMENT_FREQ // MODEL_FREQ)
        phone = phones[fileid]
        assert abs(len(phone) - len(unit)) <= 2, (len(phone), len(unit), fileid)
        for p, u in zip(phone, unit):
            if p not in dictionnary:
                dictionnary[p] = idx
                idx += 1
            count[dictionnary[p], u] += 1

    most_frequent_phones = np.argsort(count.sum(axis=1))[::-1]
    phone_order = [{v: k for k, v in dictionnary.items()}[idx] for idx in most_frequent_phones]
    count = count[most_frequent_phones]
    return count, phone_order


class DiscreteUnits:
    def __init__(self, phones: dict[str, list[str]], units: dict[str, list[int]]) -> None:
        """Compute the quality of the discrete units.
        Because the forced alignment is done at a frequency of 100Hz and the model
        has a frequency of 50Hz, each unit is repeated 2 times.

        Parameters
        ----------
        phones : dict[str, list[str]]
            Dictionnary mapping from fileid to list of phones.
        units : dict[str, list[int]]
            Dictionnary mapping from fileid to list of units.
        """
        self.count, self.phone_order = _build_count_matrix(phones, units)

    def phone_purity(self) -> float:
        proba = self.count / self.count.sum()
        return proba.max(axis=0).sum()

    def cluster_purity(self) -> float:
        proba = self.count / self.count.sum()
        return proba.max(axis=1).sum()

    def pnmi(self) -> float:
        proba = self.count / self.count.sum()
        px = proba.sum(axis=1, keepdims=True)
        py = proba.sum(axis=0, keepdims=True)
        mutual_info = (proba * np.log(proba / (px @ py + 1e-8) + 1e-8)).sum()
        entropy_x = (-px * np.log(px + 1e-8)).sum()
        return mutual_info / entropy_x

    def proba_phone_code(self) -> tuple[np.ndarray, list[int]]:
        count_by_code = self.count.sum(axis=0, keepdims=True)
        proba = np.divide(
            self.count,
            count_by_code,
            out=np.zeros_like(self.count, dtype="float64"),
            where=count_by_code != 0,
        )
        assert not np.any(np.isnan(proba))
        units_order, argmax = [], proba.argmax(axis=0)
        for phone_index in range(len(self.count)):
            indices = np.where(argmax == phone_index)[0]
            units_order.extend(indices[np.argsort(proba[phone_index, indices])[::-1]])
        return proba[:, units_order], units_order

    def describe(self) -> None:
        print("Phone purity:\t", self.phone_purity())
        print("Cluster purity:\t", self.cluster_purity())
        print("PNMI:\t\t", self.pnmi())
