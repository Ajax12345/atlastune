import typing, ddpg
import mysql_conn
import numpy as np

to_array = np.array

class CDB_Wrapper:
    

def train(config:dict) -> None:
    knob_values = self.conn.reset_knob_configuration()

    metrics = db.MySQL.metrics_to_list(self.conn._metrics())
    indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())

    state = to_array([*(indices if is_marl else []), *metrics])

