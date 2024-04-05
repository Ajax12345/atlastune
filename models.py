import typing, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import mysql_conn as db

class Normalize:
    @classmethod
    def normalize(cls, arr:typing.List[float]) -> typing.List[float]:
        mean = sum(arr)/len(arr)
        std = pow(sum(pow(i - mean, 2) for i in arr)/len(arr), 0.5)
        return [(i - mean)/std for i in arr]

class Atlas_Index_Critic(nn.Module):
    def __init__(self, state_num:int, action_num:int, val_num:int) -> None:
        super().__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.val_num = val_num
        self.state_input = nn.Linear(self.state_num, 128)
        self.action_input = nn.Linear(self.action_num, 128)
        self.act = nn.Tanh()
        self.layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
        )
    
    def forward(self, state, action) -> typing.Any:
        state = self.act(self.state_input(state))
        action = self.act(self.action_input(action))
        return self.layers(torch.cat([state, action], dim = 1))


class Atlas_Index_Actor(nn.Module):
    def __init__(self, state_num:int, index_num:int) -> None:
        super().__init__()
        self.state_num = state_num
        self.index_num = index_num
        self.layers = nn.Sequential(
            nn.Linear(self.state_num, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.BatchNorm1d(64),
        )
        self.out_layer = nn.Linear(64, self.index_num)
        self.act = nn.Sigmoid()

    def forward(self, x) -> torch.tensor:
        return self.act(self.out_layer(self.layers(x)))


class Atlas_Index_Tune:
    def __init__(self, database:str, conn = None) -> None:
        self.database = database
        self.conn = conn
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None

    def mount_entities(self) -> None:
        if self.conn is None:
            self.conn = db.MySQL(database = self.database)

    def __enter__(self) -> 'Atlas_Index_Tune':
        self.mount_entities()
        return self

    def tune(self) -> None:
        metrics = db.MySQL.metrics_to_list(self.conn._metrics())
        indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())
        start_state = torch.tensor([Normalize.normalize(state:=[*indices, *metrics])], requires_grad = True)
        state_num, action_num = len(state), len(indices)

        self.actor = Atlas_Index_Actor(state_num, action_num)
        self.actor_target = Atlas_Index_Actor(state_num, action_num)
        self.critic = Atlas_Index_Critic(state_num, action_num, 1)
        self.critic_target = Atlas_Index_Critic(state_num, action_num, 1)
        self.actor.eval()
        p_action = self.actor(start_state)
        print(p_action)
        self.critic.eval()
        print(self.critic(start_state, p_action))
        #print(db.MySQL.activate_index_actor_outputs(self.actor(start_state).tolist()))
        #print(self.conn.workload_cost())

    def __exit__(self, *_) -> None:
        if self.conn is not None:
            self.conn.__exit__()


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(database="{self.database}")'


if __name__ == '__main__':
    a = Atlas_Index_Tune('tpcc100')
    a.mount_entities()
    a.tune()
        