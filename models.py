import typing, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import mysql_conn as db

class Normalize:
    @classmethod
    def normalize(cls, arr:typing.List[float]) -> typing.List[float]:
        mean = sum(arr)/len(arr)
        std = pow(sum(pow(i - mean, 2) for i in arr)/len(arr), 0.5)
        return [(i - mean)/std for i in arr]


def perform_tests() -> None:
    with db.MySQL(database = "atlas_stuff") as conn:
        metrics = db.MySQL.metrics_to_list(conn._metrics())
        indices = db.MySQL.col_indices_to_list(conn.get_columns("test_stuff"))
        print(Normalize.normalize([*indices, *metrics]))


if __name__ == '__main__':
    perform_tests()
        