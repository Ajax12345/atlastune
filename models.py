import typing, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import mysql_conn as db, random
import copy, json

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
    def __init__(self, database:str, conn = None, config = {
            'alr':0.00001,
            'clr':0.00001,
            'tau':0.00001
        }) -> None:
        self.database = database
        self.conn = conn
        self.config = config
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.loss_criterion = None
        self.experience_replay = []

    def mount_entities(self) -> None:
        if self.conn is None:
            self.conn = db.MySQL(database = self.database)

    def __enter__(self) -> 'Atlas_Index_Tune':
        self.mount_entities()
        return self

    def generate_experience_replay(self, indices:typing.List[int], metrics:typing.List, iterations:int, from_buffer:bool = False) -> None:
        if from_buffer:
            with open('experience_replay.json') as f:
                self.experience_replay = json.load(f)
            
            return

        self.experience_replay = [[[*indices, *metrics], None, None, None, self.conn.workload_cost()]]
        for _ in range(iterations):
            _indices = copy.deepcopy(indices)
            for i in random.sample([*range(len(indices))], random.choice([*range(1, 5)])):
                _indices[i] = int(not _indices[i])
        
            self.conn.apply_index_configuration(_indices)
            self.experience_replay.append([[*indices, *metrics], _indices, 
                self.compute_step_reward(self.experience_replay[-1][-1], 
                        w2:=self.conn.workload_cost()), 
                [*_indices, *metrics], w2])
            indices = _indices

        with open('experience_replay.json', 'w') as f:
            json.dump(self.experience_replay, f)

    def compute_step_reward(self, w1:dict, w2:dict) -> float:
        k = [(float(w1[a]['cost']) - float(b['cost']))/w1[a]['cost'] for a, b in w2.items()]
        return -0.5 if not (l:=(sum(k)/len(k))*10) else l


    def _test_experience_replay(self) -> None:
        with open('experience_replay.json') as f:
            data = json.load(f)

        for i in range(len(data)-1):
            print(self.compute_step_reward(data[i][-1], data[i+1][-1]))
            print('-'*20)      

    def init_models(self, state_num:int, action_num:int) -> None:
        self.actor = Atlas_Index_Actor(state_num, action_num)
        self.actor_target = Atlas_Index_Actor(state_num, action_num)
        self.critic = Atlas_Index_Critic(state_num, action_num, 1)
        self.critic_target = Atlas_Index_Critic(state_num, action_num, 1)
        self.loss_criterion = nn.MSELoss()
        self.actor_optimizer = optimizer.Adam(lr=self.config['alr'], params=self.actor.parameters(), weight_decay=1e-5)
        self.critic_optimizer = optimizer.Adam(lr=self.config['clr'], params=self.critic.parameters(), weight_decay=1e-5)

    def tune(self) -> None:
        metrics = db.MySQL.metrics_to_list(self.conn._metrics())
        indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())
        self.generate_experience_replay(indices, metrics, 50, from_buffer = True)
        for i in self.experience_replay:
            print(i[2])
        '''
        start_state = torch.tensor([Normalize.normalize(state:=[*indices, *metrics])], requires_grad = True)
        state_num, action_num = len(state), len(indices)
        self.init_models(state_num, action_num)
        self.actor.eval()
        p_action = self.actor(start_state)
        print(p_action)
        self.critic.eval()
        print(self.critic(start_state, p_action))
        #print(db.MySQL.activate_index_actor_outputs(self.actor(start_state).tolist()))
        #print(self.conn.workload_cost())
        '''

    def __exit__(self, *_) -> None:
        if self.conn is not None:
            self.conn.__exit__()


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(database="{self.database}")'


if __name__ == '__main__':
    
    a = Atlas_Index_Tune('tpcc100')
    a.mount_entities()
    a.conn.drop_all_indices()
    a.tune()
    