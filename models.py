import typing, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import mysql_conn as db, random
import copy, json, datetime
import collections, time
import matplotlib.pyplot as plt
import statistics

class Normalize:
    @classmethod
    def normalize(cls, arr:typing.List[float]) -> typing.List[float]:
        mean = sum(arr)/len(arr)
        std = pow(sum(pow(i - mean, 2) for i in arr)/len(arr), 0.5)
        if std:
            return [(i - mean)/std for i in arr]

        return [i - mean for i in arr]

    @classmethod
    def split_normalize(cls, ind:int, arr:typing.List[float]) -> typing.List[float]:
        return [*map(float, arr[:ind])] + cls.normalize(arr[ind:]) 

    @classmethod
    def add_noise(cls, inds:typing.List[typing.List[float]], noise_scale:float) -> typing.List[typing.List[float]]:
        return [[X + Y for X, Y in zip(ind, np.random.randn(len(ind))*noise_scale)] 
                    for ind in inds]

class Atlas_Knob_Critic(nn.Module):
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
        #self._init_weights()

    def _init_weights(self):
        self.state_input.weight.data.normal_(0.0, 1e-2)
        self.state_input.bias.data.uniform_(-0.1, 0.1)

        self.action_input.weight.data.normal_(0.0, 1e-2)
        self.action_input.bias.data.uniform_(-0.1, 0.1)

        for m in self.layers:
            if type(m) == nn.Linear:
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, state, action) -> typing.Any:
        state = self.act(self.state_input(state))
        action = self.act(self.action_input(action))
        return self.layers(torch.cat([state, action], dim = 1))

class Atlas_Knob_Actor(nn.Module):
    def __init__(self, state_num:int, action_num:int) -> None:
        super().__init__()
        self.state_num = state_num
        self.action_num = action_num
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
        self.out_layer = nn.Linear(64, self.action_num)
        self.act = nn.Sigmoid()
        #self._init_weights()

    def _init_weights(self):

        for m in self.layers:
            if type(m) == nn.Linear:
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, x) -> torch.tensor:
        return self.act(self.out_layer(self.layers(x)))

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
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        #self._init_weights()

    def _init_weights(self):
        self.state_input.weight.data.normal_(0.0, 1e-2)
        self.state_input.bias.data.uniform_(-0.1, 0.1)

        self.action_input.weight.data.normal_(0.0, 1e-2)
        self.action_input.bias.data.uniform_(-0.1, 0.1)

        for m in self.layers:
            if type(m) == nn.Linear:
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

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
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.out_layer = nn.Linear(64, self.index_num)
        self.act = nn.Sigmoid()
        #self._init_weights()

    def _init_weights(self):

        for m in self.layers:
            if type(m) == nn.Linear:
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, x) -> torch.tensor:
        return self.act(self.out_layer(self.layers(x)))

class Atlas_Index_QNet(nn.Module):
    def __init__(self, state_num:int, action_num:int) -> None:
        super().__init__()
        self.state_num = state_num 
        self.action_num = action_num
        #TODO: perhaps try 128 instead of 64?
        self.layers = nn.Sequential(
            nn.Linear(self.state_num, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_num)
        )

    def forward(self, x) -> torch.tensor:
        return self.layers(x)


class Atlas:
    def compute_delta_min_reward(self, experience_replay:typing.List[tuple], costs, w2:dict) -> float:
        '''knob tuning reward function'''
        w1 = experience_replay[0][-1]['knob']
        w2 = w2['knob']
        return min((w1['latency'] - w2['latency'])/w1['latency'],
            (w2['throughput'] - w1['throughput'])/w1['throughput'])

    def compute_delta_avg_reward(self, experience_replay:typing.List[tuple], costs, w2:dict) -> float:
        '''knob tuning reward function'''
        w1 = experience_replay[0][-1]['knob']
        w2 = w2['knob']
        return ((w1['latency'] - w2['latency'])/w1['latency'] + 
            (w2['throughput'] - w1['throughput'])/w1['throughput'])/2


    def workload_cost(self, w:dict) -> float:
        return sum(w[i]['cost'] for i in w)

    def compute_cost_delta(self, experience_replay:typing.List[dict], costs:typing.List[dict], w2:dict) -> float:
        '''index selection reward function'''
        j = self.workload_cost(experience_replay[0][-1]['index']) 
        k = self.workload_cost(w2['index'])
        if j > k:
            return 5
        
        if j < k:
            return -5

        return 1

    def compute_min_step_cost(self, experience_replay:typing.List[dict], costs:typing.List[dict], w2:dict) -> float:
        '''index selection reward function'''
        return min(map(self.workload_cost, costs[-50:])) - self.workload_cost(w2['index'])

    def compute_increment_step_cost(self, experience_replay:typing.List[dict], costs:typing.List[dict], w2:dict) -> float:
        '''index selection reward function'''
        return (self.workload_cost(costs[-1]) - self.workload_cost(w2['index']))/self.workload_cost(costs[0])

    def compute_cost_delta_per_query(self, experience_replay:typing.List[dict], costs:typing.List[dict], w2:dict) -> float:
        '''index selection reward function'''
        w1 = experience_replay[0][-1]['index']
        k = [(float(w1[a]['cost']) - float(b['cost']))/w1[a]['cost'] for a, b in w2['index'].items()]
        return max(min((sum(k)/len(k))*10, 10), -10)

    def compute_cost_delta_per_query_unscaled(self, experience_replay:typing.List[dict], costs:typing.List[dict], w2:dict) -> float:
        '''index selection reward function'''
        w1 = experience_replay[0][-1]['index']
        k = [(float(w1[a]['cost']) - float(b['cost']))/w1[a]['cost'] for a, b in w2['index'].items()]
        return sum(k)/len(k)

    def compute_total_cost_reward(self, _, costs:typing.List[dict], w:dict) -> float:
        '''index selection reward function'''
        return -1*self.workload_cost(w['index'])

    def compute_ranking_reward(self, experience_replay:typing.List[list], costs:typing.List[dict], w2:dict) -> float:
        '''index selection reward function'''
        c = [self.workload_cost(i[-1]['index']) for i in experience_replay]
        w2_c = self.workload_cost(w2['index'])
        '''
        s1, s2 = sum(w2_c > i for i in c), sum(w2_c <= i for i in c)
        if s1 > s2:
            return 5
        
        if s1 < s2:
            return -5
        
        return 1
        '''
        return sum(w2_c < i for i in c)/len(c)

    def compute_step_reward(self, w1:dict, w2:dict) -> float:
        '''index selection reward function'''
        w1 = w1['index']
        k = [j for a, b in w2['index'].items() if (j:=((float(w1[a]['cost']) - float(b['cost']))/w1[a]['cost']))]
        if not k:
            return 1

        return max(min((sum(k)/len(k))*10, 10), -10)
        '''
        return -0.5 if not (l:=(sum(k)/len(k))*10) else l
        '''

    def compute_team_reward_avg(self, experience_replay:typing.List[dict], costs:typing.List[dict], w2:dict) -> float:
        '''MARL team reward'''
        w1 = experience_replay[0][-1]['knob']
        w2_knob = w2['knob']
        d = [
            self.compute_cost_delta_per_query_unscaled(experience_replay, costs, w2),
            (w1['latency'] - w2_knob['latency'])/w1['latency'],
            (w2_knob['throughput'] - w1['throughput'])/w1['throughput']
        ]
        return sum(d)/len(d)

    def compute_team_reward_min(self, experience_replay:typing.List[dict], costs:typing.List[dict], w2:dict) -> float:
        '''MARL team reward'''
        w1 = experience_replay[0][-1]['knob']
        w2_knob = w2['knob']
        d = [
            self.compute_cost_delta_per_query_unscaled(experience_replay, costs, w2),
            (w1['latency'] - w2_knob['latency'])/w1['latency'],
            (w2_knob['throughput'] - w1['throughput'])/w1['throughput']
        ]
        return min(d)

    def compute_team_reward_scaled(self, experience_replay:typing.List[dict], costs:typing.List[dict], w2:dict) -> float:
        w1 = experience_replay[0][-1]['knob']
        w2_knob = w2['knob']
        c = self.compute_cost_delta_per_query_unscaled(experience_replay, costs, w2)*0.5
        if w1['latency'] < w2_knob['latency']:
            c -= 3
        else:
            c += 3
        
        if w2_knob['throughput'] < w1['throughput']:
            c -= 3
        else:
            c += 3

        return c

class Atlas_Knob_Tune(Atlas):
    def __init__(self, database:str, conn = None, config = {
            'alr':0.001,
            'clr':0.001,
            'gamma':0.9,
            'noise_scale':0.5,
            'noise_decay':0.001,
            'replay_size':50,
            'batch_size':20,
            'tpcc_time':4,
            'tau':0.9999
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
        self.tuning_log = None
        self.experience_replay = []

    def mount_entities(self) -> None:
        if self.conn is None:
            self.conn = db.MySQL(database = self.database)

    def __enter__(self) -> 'Atlas_Knob_Tune':
        self.mount_entities()
        self.tuning_log = open(f"logs/knob_tuning_{str(datetime.datetime.now()).replace(' ', '').replace('.', '')}.txt", 'a')
        self.conn.set_log_file(self.tuning_log)

        return self

    def update_config(self, **kwargs) -> None:
        self.config.update(kwargs)

    def init_models(self, state_num:int, action_num:int) -> None:
        self.actor = Atlas_Knob_Actor(state_num, action_num)
        self.actor_target = Atlas_Knob_Actor(state_num, action_num)
        self.critic = Atlas_Knob_Critic(state_num, action_num, 1)
        self.critic_target = Atlas_Knob_Critic(state_num, action_num, 1)
        self.loss_criterion = nn.MSELoss()
        self.actor_optimizer = optimizer.Adam(lr=self.config['alr'], params=self.actor.parameters(), weight_decay=1e-5)
        self.critic_optimizer = optimizer.Adam(lr=self.config['clr'], params=self.critic.parameters(), weight_decay=1e-5)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def update_target_weights(self, target_m, m) -> None:
        for target_param, param in zip(target_m.parameters(), m.parameters()):
            target_param.data.copy_(
                target_param.data * self.config['tau'] + param.data * (1-self.config['tau'])
            )

    def log_message(self, message:str) -> None:
        self.tuning_log.write('\n'+'#'*16+message+'#'*16)

    def build_workload_payload(self, is_marl:bool = False) -> dict:
        d = {'knob': self.conn.tpcc_metrics(self.config['tpcc_time'])}
        if getattr(self, 'ENABLE_MARL_REWARD', True) or (is_marl and not getattr(self, 'DISABLE_MARL_REWARD', False)):
            d['index'] = self.conn.workload_cost()

        return d

    def tune(self, iterations:int, reward_func:str = 'compute_delta_min_reward', reset_knobs:bool = True, is_marl:bool = False) -> dict:
        if reset_knobs:
            self.log_message('Resetting knobs')
            self.conn.reset_knob_configuration()

        metrics = db.MySQL.metrics_to_list(self.conn._metrics())
        indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())

        state = [*(indices if is_marl else []), *metrics]
        print('length of state', len(state))
        start_state = torch.tensor([Normalize.normalize(state)], requires_grad = True)
        state_num, action_num = len(state), db.MySQL.knob_num

        self.init_models(state_num, action_num)

        '''
        self.log_message('Getting default metrics')
        with open('experience_replay/ddpg_knob_tune/tpcc_1000_2024-04-1815:57:59283850.json') as f:
            self.experience_replay = json.load(f)
            skip_experience = len(self.experience_replay)
        '''
        self.experience_replay = [[state, None, None, None, self.build_workload_payload(is_marl)]]
        rewards = []
        skip_experience = self.config['replay_size']
        noise_scale = self.config['noise_scale']
        for i in range(iterations):
            print('iteration', i+1)
            self.log_message(f'Iteration {i+1}')
            self.actor.eval()
            self.actor_target.eval()
            self.critic.eval()
            self.critic_target.eval()

            knob_activation_payload = {
                'memory_size':(mem_size:=self.conn.memory_size('b')[self.database]*4),
                'memory_lower_bound':min(4294967168, mem_size)
            }
            [selected_action] = Normalize.add_noise(self.actor(start_state).tolist(), noise_scale)
            chosen_knobs, knob_dict = db.MySQL.activate_knob_actor_outputs(selected_action, knob_activation_payload)
            
            t = time.time()
            self.conn.apply_knob_configuration(knob_dict)
            #print('configuration completed', time.time()-t)
            self.experience_replay.append([state, selected_action, 
                reward:=getattr(self, reward_func)(self.experience_replay, None, w2:=self.build_workload_payload(is_marl)),
                [*(indices if is_marl else []), *(metrics:=db.MySQL.metrics_to_list(self.conn._metrics()))],
                w2
            ])
            #print('new stats computed')
            rewards.append(reward)
            state = [*(indices if is_marl else []), *metrics]
            print('noise scale:', noise_scale)
            noise_scale -= noise_scale*self.config['noise_decay']
            start_state = torch.tensor([Normalize.normalize(state)], requires_grad = True)

            if len(self.experience_replay) >= self.config['replay_size']:
                '''
                with open(e_f_file:=f"experience_replay/ddpg_knob_tune/{self.conn.database}_{str(datetime.datetime.now()).replace(' ', '').replace('.', '')}.json", 'a') as f:
                    json.dump(self.experience_replay, f)

                print('experience replay saved to:', e_f_file)
                '''
                inds = random.sample([*range(1,len(self.experience_replay))], self.config['batch_size'])
                _s, _a, _r, _s_prime, w2 = zip(*[self.experience_replay[i] for i in inds])
                s = torch.tensor([Normalize.normalize(i) for i in _s])
                a = torch.tensor([[float(j) for j in i] for i in _a])
                s_prime = torch.tensor([Normalize.normalize(i) for i in _s_prime])
                r = torch.tensor([[float(i)] for i in Normalize.normalize(_r)])

                target_action = self.actor_target(s_prime)

                target_q_value = self.critic_target(s_prime, target_action)
                next_value = r + self.config['gamma']*target_q_value

                current_value = self.critic(s, a)
                
                u = self.actor(s)
                predicted_q_value = self.critic(s, u)

                self.actor.train()
                self.actor_target.train()
                self.critic.train()
                self.critic_target.train()

                loss = self.loss_criterion(current_value, next_value)
                print(loss)
            
                
                self.critic_optimizer.zero_grad()
                loss.backward()

                pqv = -1*predicted_q_value
                actor_loss = pqv.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                self.critic_optimizer.step()
                self.actor_optimizer.step()

                self.update_target_weights(self.critic_target, self.critic)
                self.update_target_weights(self.actor_target, self.actor)

            if i and not i%self.config['marl_step']:
                yield {
                'experience_replay':self.experience_replay,
                'rewards':rewards[skip_experience:], 
                'metrics':[i[-1]['knob'] 
                    for i in self.experience_replay[skip_experience:]]}, False
                
        yield {
            'experience_replay':self.experience_replay,
            'rewards':rewards[skip_experience:], 
            'metrics':[i[-1]['knob'] 
                for i in self.experience_replay[skip_experience:]]}, True


    def __exit__(self, *_) -> None:
        if self.conn is not None:
            self.conn.__exit__()
        
        if self.tuning_log is not None:
            self.tuning_log.close()



class Atlas_Index_Tune(Atlas):
    def __init__(self, database:str, conn = None, config = {
            'alr':0.0001,
            'clr':0.0001,
            'gamma':0.9,
            'tau':0.9999
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

    def save_experience_replay(self, f_name = None) -> None:
        f_name = f_name if f_name is not None else f"experience_replay/ddpg_index_tune/experience_replay_{self.conn.database}_{str(datetime.datetime.now()).replace(' ', '').replace('.', '')}.json"
        with open(f_name, 'a') as f:
            json.dump(self.experience_replay, f)

    def generate_random_index(self, indices:typing.List[int], bound:int = 6) -> typing.List[int]:
        _indices = copy.deepcopy(indices)
        for i in random.sample([*range(len(indices))], random.choice([*range(1, bound)])):
            _indices[i] = int(not _indices[i])

        return _indices  

    def __generate_experience_replay(self, indices:typing.List[int], metrics:typing.List, iterations:int, from_buffer:bool = False) -> None:
        if from_buffer:
            with open(from_buffer) as f:
                self.experience_replay = json.load(f)
            
            return

        self.experience_replay = [[[*indices, *metrics], None, None, None, self.conn.workload_cost()]]
        for _ in range(iterations):
            _indices = self.generate_random_index(indices)
        
            self.conn.apply_index_configuration(_indices)
            self.experience_replay.append([[*indices, *metrics], _indices, 
                self.compute_step_reward(self.experience_replay[-1][-1], 
                        w2:=self.conn.workload_cost()), 
                [*_indices, *metrics], w2])
            indices = _indices

        self.save_experience_replay('experience_replay/ddpg_index_tune/custom_exper_repr.json')

    def generate_experience_replay(self, indices:typing.List[int], metrics:typing.List, iterations:int, from_buffer:bool = False) -> None:
        if from_buffer:
            with open(from_buffer) as f:
                self.experience_replay = json.load(f)
            
            return

        self.experience_replay = [[[*indices, *metrics], None, None, None, self.conn.workload_cost()]]
        for _ in range(iterations):
            _indices = self.generate_random_index(indices)
        
            self.conn.apply_index_configuration(_indices)
            self.experience_replay.append([[*indices, *metrics], _indices, 
                self.compute_cost_delta_per_query(self.experience_replay, 
                        w2:=self.conn.workload_cost()), 
                [*_indices, *metrics], w2])
            indices = _indices

        self.save_experience_replay(f1_name:=f"experience_replay/ddpg_index_tune/custom_exprience_replay{str(datetime.datetime.now()).replace('.','').replace(' ','')}.json")
        print('experience replay saved to: ', f1_name)

    def _test_experience_replay(self) -> None:
        with open('experience_replay/ddpg_index_tune/experience_replay3.json') as f:
            data = json.load(f)

        '''
        for i in range(len(data)-1):
            w1, w2 = data[i][-1], data[i+1][-1]
            #print([(w1[a]['cost'], w1[a]['cost'] - b['cost']) for a, b in w2.items()])
            print(self.compute_step_reward(data[i][-1], data[i+1][-1]))
            #print('-'*20)    
        '''
        for i in data:
            print(i[2])  
       

    def init_models(self, state_num:int, action_num:int) -> None:
        self.actor = Atlas_Index_Actor(state_num, action_num)
        self.actor_target = Atlas_Index_Actor(state_num, action_num)
        self.critic = Atlas_Index_Critic(state_num, action_num, 1)
        self.critic_target = Atlas_Index_Critic(state_num, action_num, 1)
        self.loss_criterion = nn.MSELoss()
        self.actor_optimizer = optimizer.Adam(lr=self.config['alr'], params=self.actor.parameters(), weight_decay=1e-5)
        self.critic_optimizer = optimizer.Adam(lr=self.config['clr'], params=self.critic.parameters(), weight_decay=1e-5)

    def update_target_weights(self, target_m, m) -> None:
        for target_param, param in zip(target_m.parameters(), m.parameters()):
            target_param.data.copy_(
                target_param.data * self.config['tau'] + param.data * (1-self.config['tau'])
            )

    def tune(self, iterations:int, with_epoch:bool = False) -> typing.List[float]:
        torch.autograd.set_detect_anomaly(True)
        self.conn.drop_all_indices()
        metrics = db.MySQL.metrics_to_list(self.conn._metrics())
        indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())
        #self.generate_experience_replay(indices, metrics, 100)
        if not with_epoch or not self.experience_replay:
            self.generate_experience_replay(indices, metrics, 50, from_buffer = 'experience_replay/ddpg_index_tune/custom_exprience_replay2024-04-0913:18:31438505.json')
        '''
        for i in self.experience_replay:
            print(i[2])
        
        '''
        #return 
        self.conn.drop_all_indices()

        state = [*indices, *metrics]
        start_state = torch.tensor([Normalize.normalize(state)], requires_grad = True)
        state_num, action_num = len(state), len(indices)
        if not with_epoch or self.actor is None:
            self.init_models(state_num, action_num)
        
        rewards = []
        reward_sum = 0
        for _ in range(iterations):
            print(_)
            self.actor.eval()
            self.actor_target.eval()
            self.critic.eval()
            self.critic_target.eval()

            '''
            if random.random() < 0.75:
                new_indices = self.generate_random_index(indices)
            
            else:
                [new_indices] = db.MySQL.activate_index_actor_outputs(self.actor(start_state).tolist())
                print(new_indices)
            '''
            #[new_indices] = db.MySQL.activate_index_actor_outputs(self.actor(start_state).tolist())
            
            selected_action = Normalize.add_noise(self.actor(start_state).tolist(), 0.05)
            [new_indices] = db.MySQL.activate_index_actor_outputs(selected_action)
            
            t = time.time()
            self.conn.apply_index_configuration(new_indices)
            #print(time.time() - t, new_indices)
            self.experience_replay.append([state, new_indices, 
                reward:=self.compute_cost_delta_per_query(self.experience_replay, None,
                        w2:=self.conn.workload_cost()), 
                [*new_indices, *metrics], w2])
            reward_sum += reward
            rewards.append(reward)
            indices = new_indices
            state = [*indices, *metrics]
            start_state = torch.tensor([Normalize.normalize(state)])
            

            inds = random.sample([*range(1,len(self.experience_replay))], 20)
            _s, _a, _r, _s_prime, w2 = zip(*[self.experience_replay[i] for i in inds])
            s = torch.tensor([Normalize.normalize(i) for i in _s])
            a = torch.tensor([[float(j) for j in i] for i in _a])
            s_prime = torch.tensor([Normalize.normalize(i) for i in _s_prime])
            r = torch.tensor([[float(i)] for i in Normalize.normalize(_r)])
            #r = torch.tensor([[float(i)] for i in _r])

            u_prime = self.actor_target(s_prime).tolist()
            target_action = torch.tensor([[float(j) for j in i] for i in db.MySQL.activate_index_actor_outputs(u_prime)])
            
            target_q_value = self.critic_target(s_prime, target_action)
            next_value = r + self.config['gamma']*target_q_value


            current_value = self.critic(s, a)

            u = self.actor(s).tolist()
            predicted_action = torch.tensor([[float(j) for j in i] for i in db.MySQL.activate_index_actor_outputs(u)])
            predicted_q_value = self.critic(s, predicted_action)

            self.actor.train()
            self.actor_target.train()
            self.critic.train()
            self.critic_target.train()

            loss = self.loss_criterion(current_value, next_value)
            print(loss)
        
            
            self.critic_optimizer.zero_grad()
            loss.backward()

            pqv = -1*predicted_q_value
            actor_loss = pqv.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            self.critic_optimizer.step()
            self.actor_optimizer.step()

            self.update_target_weights(self.critic_target, self.critic)
            self.update_target_weights(self.actor_target, self.actor)

    
        #self.save_experience_replay()
        return rewards

    def tune_random(self, iterations:int) -> typing.List[float]:
        self.conn.drop_all_indices()
        indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())
        _experience_replay = [[indices, None, self.conn.workload_cost()]]
        rewards = []
        for _ in range(iterations):
            print(_)
            indices = self.generate_random_index(indices, 2)
            print(indices)
            self.conn.apply_index_configuration(indices)
            _experience_replay.append([indices, reward:=self.compute_cost_delta_per_query(_experience_replay, None,
                        w2:=self.conn.workload_cost()), w2])
            rewards.append(reward)
        
        return rewards

    def __exit__(self, *_) -> None:
        if self.conn is not None:
            self.conn.__exit__()


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(database="{self.database}")'


class Atlas_Index_Tune_DQN(Atlas_Index_Tune, Atlas):
    def __init__(self, database:str, conn = None, config = {
            'lr':0.0001,
            'gamma':0.9,
            'weight_copy_interval':10,
            'tau':0.9999,
            'epsilon':1,
            'epsilon_decay':0.001,
            'batch_sample_size':50
        }) -> None:

        self.database = database
        self.conn = conn
        self.config = config
        self.q_net = None
        self.q_net_target = None
        self.loss_func = None
        self.optimizer = None
        self.experience_replay = []

    def update_config(self, **kwargs) -> None:
        self.config.update(kwargs)

    def reset_target_weights(self) -> None:
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def init_models(self, state_num:int, action_num:int) -> None:
        self.q_net = Atlas_Index_QNet(state_num, action_num)
        self.q_net_target = Atlas_Index_QNet(state_num, action_num)
        self.reset_target_weights()
        self.loss_func = nn.MSELoss() #TODO: try Huber loss
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = self.config['lr'])
        
    def random_action(self, indices:typing.List[int]) -> typing.Tuple[int, typing.List[int]]:
        ind = random.choice([*range(len(indices)+1)])
        _indices = copy.deepcopy(indices)
        if ind < len(_indices):
            _indices[ind] = int(not _indices[ind])

        return ind, _indices

    def build_workload_payload(self, is_marl:bool = False) -> dict:
        d = {'index': self.conn.workload_cost()}
        if getattr(self, 'ENABLE_MARL_REWARD', True) or (is_marl and not getattr(self, 'DISABLE_MARL_REWARD', False)):
            d['knob'] = self.conn.tpcc_metrics(self.config['tpcc_time'])

        return d

    def generate_experience_replay(self, indices:typing.List[int], metrics:typing.List[int], iterations:int, reward_func:str, is_marl:bool, from_buffer:typing.Union[bool, str] = False) -> None:
        print('experience replay reward func:', reward_func)
        if from_buffer:
            print('loading experience replay from buffer ', from_buffer)
            with open(from_buffer) as f:
                self.experience_replay = json.load(f)
            
            return

        self.experience_replay = [[[*indices, *(metrics if is_marl else [])], None, None, None, w2_o:=self.build_workload_payload(is_marl)]]
        costs = [w2_o['index']]
        for _ in range(iterations):
            ind, _indices = self.random_action(indices)

            self.conn.apply_index_configuration(_indices)
            self.experience_replay.append([[*indices, *(metrics if is_marl else [])], ind, 
                getattr(self, reward_func)(self.experience_replay, costs,
                        w2:=self.build_workload_payload(is_marl)), 
                [*_indices, *(metrics if is_marl else [])], w2])
            indices = _indices
            costs.append(w2['index'])

        print('experience replay saved to:')
        print(self.save_experience_replay())


    def save_experience_replay(self, f_name = None) -> str:
        f_name = f_name if f_name is not None else f"experience_replay/dqn_index_tune/experience_replay_{self.conn.database}_{str(datetime.datetime.now()).replace(' ', '').replace('.', '')}.json"
        with open(f_name, 'a') as f:
            json.dump(self.experience_replay, f)

        return f_name

    def tune(self, iterations:int, reward_func:str = 'compute_cost_delta_per_query', from_buffer = None, is_epoch:bool = False, is_marl:bool = False) -> typing.Iterator:
        print('tuning reward function:', reward_func)
        self.conn.drop_all_indices()
        metrics = db.MySQL.metrics_to_list(self.conn._metrics())
        indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())

        self.generate_experience_replay(indices, metrics, 150, reward_func, is_marl, from_buffer = from_buffer)
        state = [*indices, *(metrics if is_marl else [])]
        print('length of state in index tune', len(state))
        start_state = torch.tensor([Normalize.normalize(state)], requires_grad = True)
        
        action_num, state_num = len(indices)+1, len(state)

        if not is_epoch or self.q_net is None:
            print('setting weights')
            self.init_models(state_num, action_num)

        else:
            print('skipping weight setting')

        self.conn.drop_all_indices()
        
        rewards, costs = [], [self.conn.workload_cost()]
        delta_rewards = []
        epsilon = self.config['epsilon']
        for iteration in range(iterations):
            if random.random() < epsilon:
                print('random')
                ind, _indices = self.random_action(indices)

            else:
                print('q_val')
                with torch.no_grad():
                    ind = self.q_net(start_state).max(1)[1].item()
                    _indices = copy.deepcopy(indices)
                    if ind < len(_indices):
                        _indices[ind] = int(not _indices[ind])
                
            
            self.conn.apply_index_configuration(_indices)
            self.experience_replay.append([state, ind, 
                reward:=getattr(self, reward_func)(self.experience_replay, costs,
                        w2:=self.build_workload_payload(is_marl)), 
                [*_indices, *(metrics if is_marl else [])], w2])

            rewards.append(reward)
            delta_rewards.append(self.compute_cost_delta_per_query(self.experience_replay, costs, w2))
            costs.append(w2['index'])
            indices = _indices
            state = [*indices, *(metrics if is_marl else [])]
            start_state = torch.tensor([Normalize.normalize(state)])
            epsilon -= epsilon*self.config['epsilon_decay']

            inds = random.sample([*range(1,len(self.experience_replay))], self.config['batch_sample_size'])
            _s, _a, _r, _s_prime, w2 = zip(*[self.experience_replay[i] for i in inds])
            s = torch.tensor([Normalize.normalize(i) for i in _s], requires_grad = True)
            a = torch.tensor([[i] for i in _a])
            s_prime = torch.tensor([Normalize.normalize(i) for i in _s_prime], requires_grad = True)
            r = torch.tensor([[float(i)] for i in Normalize.normalize(_r)], requires_grad = True)

            with torch.no_grad():
                q_prime = self.q_net_target(s_prime).max(1)[0].unsqueeze(1)
                q_value = self.q_net(s).gather(1, a)
            
            target_q_value = r + self.config['gamma']*q_prime

            loss = self.loss_func(q_value, target_q_value)
            #print(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if iteration and not iteration%self.config['weight_copy_interval']:
                self.reset_target_weights()

            if iteration and not iteration%self.config['marl_step']:
                yield {'experience_replay':self.experience_replay, 
                    'rewards':rewards, 
                    'delta_rewards':delta_rewards, 
                    'costs':costs}, False

        yield {'experience_replay':self.experience_replay, 
                    'rewards':rewards, 
                    'delta_rewards':delta_rewards, 
                    'costs':costs}, True
    
    def mount_entities(self) -> None:
        if self.conn is None:
            self.conn = db.MySQL(database = self.database)


    def __enter__(self) -> 'Atlas_Index_Tune':
        self.mount_entities()
        return self

    def __exit__(self, *_) -> None:
        if self.conn is not None:
            self.conn.__exit__()


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(database="{self.database}")'

def atlas_index_tune_ddpg() -> None:
    with Atlas_Index_Tune('tpcc100') as a:
        #a.tune_random(300)
        rewards = []
        
        for _ in range(1):
            a.conn.drop_all_indices()
            rewards.append(a.tune(50))
            with open(f'outputs/rl_ddpg14.json', 'a') as f:
                json.dump(rewards, f)
       
        plt.plot([*range(1,len(rewards[0])+1)], [sum(i)/len(i) for i in zip(*rewards)], label="ddpg")
        '''
        
        '''
        
        '''
        with open(f'outputs/rl_ddpg12.json') as f:
            rewards = json.load(f)

        
        #print(len(rewards))
        plt.plot([*range(1,len(rewards[0])+1)], [sum(i)/len(i) for i in zip(*rewards)], label="ddpg_main")
    
        with open('outputs/rl_ddpg2.json') as f:
            rewards = json.load(f)
            plt.plot([*range(1,len(rewards[0])+1)], [sum(i)/len(i) for i in zip(*rewards)], label="ddpg")

        
        with open('outputs/rl_ddpg3.json') as f:
            rewards = json.load(f)
            plt.plot([*range(1,len(rewards[0])+1)], [sum(i)/len(i) for i in zip(*rewards)], label="ddpg1")

        with open('outputs/random_control.json') as f:
            #rewards = [i[:d] for i in json.load(f)]
            rewards = json.load(f)
            plt.plot([*range(1,len(rewards[0])+1)], [sum(i)/len(i) for i in zip(*rewards)], label="random")
        

        '''
        plt.title("reward at each iteration (5 epochs)")
        plt.legend(loc="lower right")

        plt.xlabel("iteration")
        plt.ylabel("reward")
        plt.show()


def display_tuning_results(f_name:str) -> None:
    with open(f_name) as f:
        tuning_data = json.load(f)

    rewards = [i['rewards'] for i in tuning_data]
    costs = [[sum(j[k]['cost'] for k in j) for j in i['costs']] for i in tuning_data]
    if 'delta_rewards' in tuning_data[0]:
        fig, [a1, a2, a3] = plt.subplots(nrows=1, ncols=3)
        delta_rewards = [i['delta_rewards'] for i in tuning_data]
        a3.plot([*range(1,len(delta_rewards[0])+1)], [sum(i)/len(i) for i in zip(*delta_rewards)], label = 'dqn')
        a3.title.set_text("Absolute Difference in Initial Cost")
        a3.legend(loc="lower right")

        a3.set_xlabel("iteration")
        a3.set_ylabel("Absolute delta cost")

    else:
        fig, [a1, a2] = plt.subplots(nrows=1, ncols=2)

    a1.plot([*range(1,len(rewards[0])+1)], [sum(i)/len(i) for i in zip(*rewards)], label = 'dqn')

    a2.plot([*range(1,len(costs[0])+1)], [sum(i)/len(i) for i in zip(*costs)], label="dqn")

    

    a1.title.set_text("Reward at each iteration")
    a1.legend(loc="lower right")

    a1.set_xlabel("iteration")
    a1.set_ylabel("reward")

    a2.title.set_text("Total workload cost at each iteration")
    a2.legend(loc="lower right")

    a2.set_xlabel("iteration")
    a2.set_ylabel("Workload cost")

    plt.show()

def display_knob_tuning_results(f_name:str) -> None:
    with open(f_name) as f:
        tuning_data = json.load(f)

    fig, [a1, a2, a3] = plt.subplots(nrows=1, ncols=3)

    
    a1.plot([*range(1, len(tuning_data['rewards'])+1)], tuning_data['rewards'], label = 'reward')

    a1.title.set_text("Reward")
    a1.legend(loc="lower right")

    a1.set_xlabel("iteration")
    a1.set_ylabel("Reward")

    a2.plot([*range(1, len(tuning_data['metrics'])+1)], [i['latency'] for i in tuning_data['metrics']], label = 'latency', color = 'orange')
    a2.title.set_text("Latency")
    a2.legend(loc="upper right")

    a2.set_xlabel("iteration")
    a2.set_ylabel("latency")

    a3.plot([*range(1, len(tuning_data['metrics'])+1)], [i['throughput'] for i in tuning_data['metrics']], label = 'throughput', color = 'green')
    a3.title.set_text("Throughput")
    a3.legend(loc="lower right")

    a3.set_xlabel("iteration")
    a3.set_ylabel("throughput")

    plt.show()

def atlas_index_tune_dqn() -> None:
    with Atlas_Index_Tune_DQN('tpcc_30') as a_index:
        a_index.DISABLE_MARL_REWARD = True
        a_index.conn.reset_knob_configuration()
        tuning_data = []
        for i in range(2):
            #print(i + 1)
            a_index.update_config(**{'weight_copy_interval':10, 'epsilon':1, 'epsilon_decay':0.003, 'marl_step':50})
            a_index_prog = a_index.tune(400, reward_func = 'compute_cost_delta_per_query_unscaled', is_marl = True)
            while True:
                payload, flag = next(a_index_prog)
                if flag:
                    tuning_data.append(payload)
                    break
        
        with open(f'outputs/tuning_data/rl_dqn20.json', 'a') as f:
            json.dump(tuning_data, f)
        
        
        display_tuning_results('outputs/tuning_data/rl_dqn20.json')

def atlas_knob_tune() -> None:
    with Atlas_Knob_Tune('tpcc_30') as a_knob:
        a_knob.update_config(**{'replay_size':50, 'noise_scale':1.5, 'noise_decay':0.006, 'batch_size':40, 'tpcc_time':4, 'marl_step':50})
        a_knob_prog = a_knob.tune(200, reward_func = 'compute_delta_avg_reward', is_marl = True)
        while True:
            results, flag = next(a_knob_prog)
            if flag:
                break

        with open('outputs/knob_tuning_data/rl_ddpg14.json', 'a') as f:
            json.dump(results, f)
        
        display_knob_tuning_results('outputs/knob_tuning_data/rl_ddpg14.json')

def display_marl_results(f_name:str) -> None:
    with open(f_name) as f:
        data = json.load(f)

    data = data['db_stats'][0]
    fig, [a1, a2, a3] = plt.subplots(nrows=1, ncols=3)
    a2.plot([*range(1, len(data)+1)], [i['knob']['latency'] for i in data], label = 'latency', color = 'orange')
    a2.title.set_text("Latency")
    a2.legend(loc="upper right")

    a2.set_xlabel("iteration")
    a2.set_ylabel("latency")

    a3.plot([*range(1, len(data)+1)], [i['knob']['throughput'] for i in data], label = 'throughput', color = 'green')
    a3.title.set_text("Throughput")
    a3.legend(loc="lower right")

    a3.set_xlabel("iteration")
    a3.set_ylabel("throughput")

    plt.show()


def file_agg_results(files:typing.List[str], agg_func:typing.Callable = statistics.median) -> typing.List[list]:
    latency, throughput = [], []
    for i in files:
        with open(i) as f:
            data = json.load(f)['db_stats'][0]
            latency.append([j['knob']['latency'] for j in data])
            throughput.append([j['knob']['throughput'] for j in data])

    agg_latency = [agg_func(i) for i in zip(*latency)]
    agg_throughput = [agg_func(i) for i in zip(*throughput)]

    return agg_latency, agg_throughput 

def marl_agg_stats(file_blocks:typing.List[typing.List[str]], agg_func:typing.Callable = statistics.median) -> None:
    fig, [a2, a3] = plt.subplots(nrows=1, ncols=2)

    for label, f_block in file_blocks:
        agg_latency, agg_throughput = file_agg_results(f_block, agg_func)
        a2.plot([*range(1, len(agg_latency)+1)], agg_latency, label = f'latency {label}')
        
        a3.plot([*range(1, len(agg_throughput)+1)], agg_throughput, label = f'throughput {label}')
    

    a2.title.set_text("Latency")
    a2.legend(loc="upper right")

    a2.set_xlabel("iteration")
    a2.set_ylabel("latency")

    a3.title.set_text("Throughput")
    a3.legend(loc="lower right")

    a3.set_xlabel("iteration")
    a3.set_ylabel("throughput")

    plt.show()

def atlas_marl_tune(database:str, marl_step:int, epochs:int, iterations:int, is_marl:bool) -> None:
    with Atlas_Index_Tune_DQN(database) as a_index:
        with Atlas_Knob_Tune(database, conn = a_index.conn) as a_knob:
            a_index.ENABLE_MARL_REWARD = True
            a_knob.ENABLE_MARL_REWARD = True
            a_index.conn.reset_knob_configuration()
            index_results, knob_results = [], []
            db_stats = []
            for _ in range(epochs):
                a_index.update_config(**{'weight_copy_interval':10, 'epsilon':1, 'epsilon_decay':0.003, 'tpcc_time':4, 'marl_step':marl_step})
                a_index_prog = a_index.tune(iterations, 
                    reward_func = 'compute_delta_avg_reward',
                    #from_buffer = 'experience_replay/dqn_index_tune/experience_replay_tpcc_30_2024-04-2914:23:35380520.json',
                    is_marl = is_marl)

                a_knob.update_config(**{'replay_size':50, 'noise_scale':1.5, 'noise_decay':0.006, 'batch_size':40, 'tpcc_time':4, 'marl_step':marl_step})
                a_knob_prog = a_knob.tune(iterations, 
                    reward_func = 'compute_delta_avg_reward', 
                    is_marl = is_marl)

                iteration_db_stats = []
                while True:
                    index_payload, index_flag = next(a_index_prog)
                    ep = index_payload['experience_replay'][-1*marl_step:]
                    iteration_db_stats.extend([i[-1] for i in ep])
                    best_index_config = max(ep, key=lambda x:x[2])[3][:a_index.conn.db_column_count]
                    print('best index config')
                    print(best_index_config)
                    a_index.conn.apply_index_configuration(best_index_config)

                    knob_payload, knob_flag = next(a_knob_prog)
                    ep = knob_payload['experience_replay'][-1*marl_step:]
                    iteration_db_stats.extend([i[-1] for i in ep])
                    best_knob_config = max(ep, key=lambda x:x[2])[1]
                    knob_activation_payload = {
                        'memory_size':(mem_size:=a_index.conn.memory_size('b')[a_index.database]*4),
                        'memory_lower_bound':min(4294967168, mem_size)
                    }
                    chosen_knobs, knob_dict = db.MySQL.activate_knob_actor_outputs(best_knob_config, knob_activation_payload)
                    a_index.conn.apply_knob_configuration(knob_dict)
                    print('chosen knob config', knob_dict)

                    if knob_flag:
                        index_results.append(index_payload)
                        knob_results.append(knob_payload)
                        break


                db_stats.append(iteration_db_stats)

            with open('outputs/marl_tuning_data/marl14.json', 'a') as f:
                json.dump({'index_results':index_results, 'knob_results':knob_results, 'db_stats':db_stats}, f)


            display_marl_results('outputs/marl_tuning_data/marl14.json')

if __name__ == '__main__':
    #display_marl_results('outputs/marl_tuning_data/marl1.json')
    atlas_marl_tune('tpcc_30', 50, 1, 300, True)
    '''
    marl_agg_stats([('MARL', [
        'outputs/marl_tuning_data/marl5.json',
        'outputs/marl_tuning_data/marl6.json',
        'outputs/marl_tuning_data/marl7.json',
        'outputs/marl_tuning_data/marl8.json',
        'outputs/marl_tuning_data/marl9.json'
    ]), 
        ('Default', [
            'outputs/marl_tuning_data/marl11.json',
            'outputs/marl_tuning_data/marl12.json',
    ])], statistics.mean)
    '''
    #display_marl_results('outputs/marl_tuning_data/marl6.json')
    #atlas_knob_tune()
    #print(Normalize.add_noise([[1, 1, 1, 1, 1, 1, 1, 1, 1]], 0.1))


