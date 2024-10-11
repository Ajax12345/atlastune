import typing, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import mysql_conn as db, random
import copy, json, datetime
import collections, time
import matplotlib.pyplot as plt
import statistics, os, re
import whittaker_eilers
from scipy.signal import savgol_filter
import math, ddpg

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
        
        '''
        self.layers = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        '''
        self._init_weights()

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
        '''
        self.layers = nn.Sequential(
            nn.Linear(self.state_num, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        '''
        self.out_layer = nn.Linear(64, self.action_num)
        self.act = nn.Sigmoid()
        self._init_weights()

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


class Atlas_Rewards:
    def compute_delta_min_reward(self, experience_replay:typing.List[tuple], w2:dict) -> float:
        '''knob tuning reward function'''
        w1 = experience_replay[0][-1]['knob']
        w2 = w2['knob']
        return min((w1['latency'] - w2['latency'])/w1['latency'],
            (w2['throughput'] - w1['throughput'])/w1['throughput'])

    def compute_delta_avg_reward(self, experience_replay:typing.List[tuple], w2:dict) -> float:
        '''knob tuning reward function'''
        w1 = experience_replay[0][-1]['knob']
        w2 = w2['knob']
        return ((w1['latency'] - w2['latency'])/w1['latency'] + 
            (w2['throughput'] - w1['throughput'])/w1['throughput'])/2


    def workload_cost(self, w:dict) -> float:
        return sum(w[i]['cost'] for i in w)

    def compute_cost_delta(self, experience_replay:typing.List[dict], w2:dict) -> float:
        '''index selection reward function'''
        j = self.workload_cost(experience_replay[0][-1]['index']) 
        k = self.workload_cost(w2['index'])
        if j > k:
            return 5
        
        if j < k:
            return -5

        return 1

    def compute_cost_delta_per_query(self, experience_replay:typing.List[dict], w2:dict) -> float:
        '''index selection reward function'''
        w1 = experience_replay[0][-1]['index']
        k = [(float(w1[a]['cost']) - float(b['cost']))/w1[a]['cost'] for a, b in w2['index'].items()]
        return max(min((sum(k)/len(k))*10, 10), -10)

    def compute_cost_delta_per_query_unscaled(self, experience_replay:typing.List[dict], w2:dict) -> float:
        '''index selection reward function'''
        w1 = experience_replay[0][-1]['index']
        k = [(float(w1[a]['cost']) - float(b['cost']))/w1[a]['cost'] for a, b in w2['index'].items()]
        return sum(k)/len(k)

    def compute_cost_delta_per_query_unscaled_geometric(self, experience_replay:typing.List[dict], w2:dict) -> float:
        '''index selection reward function'''
        w1 = experience_replay[0][-1]['index']
        k = [(float(w1[a]['cost']) - float(b['cost']))/w1[a]['cost'] for a, b in w2['index'].items()]
        pos, neg = [], []
        for i in k:
            if i > 0:
                pos.append(i)
            elif i < 0:
                neg.append(abs(i))

        return (0 if not pos else statistics.geometric_mean(pos)) - (0 if not neg else statistics.geometric_mean(neg))

    def compute_total_cost_reward(self, _, w:dict) -> float:
        '''index selection reward function'''
        return -1*self.workload_cost(w['index'])

    def compute_ranking_reward(self, experience_replay:typing.List[list], w2:dict) -> float:
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

    def compute_team_reward_avg(self, experience_replay:typing.List[dict], w2:dict) -> float:
        '''MARL team reward'''
        w1 = experience_replay[0][-1]['knob']
        w2_knob = w2['knob']
        d = [
            self.compute_cost_delta_per_query_unscaled(experience_replay, w2),
            (w1['latency'] - w2_knob['latency'])/w1['latency'],
            (w2_knob['throughput'] - w1['throughput'])/w1['throughput']
        ]
        return sum(d)/len(d)

    def compute_team_reward_min(self, experience_replay:typing.List[dict], w2:dict) -> float:
        '''MARL team reward'''
        w1 = experience_replay[0][-1]['knob']
        w2_knob = w2['knob']
        d = [
            self.compute_cost_delta_per_query_unscaled(experience_replay, w2),
            (w1['latency'] - w2_knob['latency'])/w1['latency'],
            (w2_knob['throughput'] - w1['throughput'])/w1['throughput']
        ]
        return min(d)

    def compute_team_reward_scaled(self, experience_replay:typing.List[dict], w2:dict) -> float:
        w1 = experience_replay[0][-1]['knob']
        w2_knob = w2['knob']
        c = self.compute_cost_delta_per_query_unscaled(experience_replay, w2)*0.5
        if w1['latency'] < w2_knob['latency']:
            c -= 3
        else:
            c += 3
        
        if w2_knob['throughput'] < w1['throughput']:
            c -= 3
        else:
            c += 3

        return c

    def compute_tpch_qph_reward(self, experience_replay:typing.List[dict], w2:dict) -> float:
        o_qph = experience_replay[0][-1]['qph']
        return (w2['qph'] - o_qph)/o_qph

    def compute_sysbench_reward(self, experience_replay:typing.List[dict], w2:dict) -> float:
        return min([
            (experience_replay[0][-1]['latency'] - w2['latency'])/experience_replay[0][-1]['latency'],
            (w2['throughput'] - experience_replay[0][-1]['throughput'])/experience_replay[0][-1]['throughput']
        ])

    def scale_num(self, n:int) -> int:
        n_b = int((s:=str(n)[0])+'0'*(L:=len(str(n))-1))
        if n >= n_b + 5*10**(L-1):
            return n_b + 10**L
        return n_b

    def compute_sysbench_reward_throughput_scaled(self, experience_replay:typing.List[dict], w2:dict) -> float:
        t = self.scale_num(int(experience_replay[0][-1]['throughput']))
        return (self.scale_num(int(w2['throughput'])) - t)/t

    def compute_sysbench_reward_throughput_discrete(self, experience_replay:typing.List[dict], w2:dict) -> float:
        d1 = self.scale_num(int(experience_replay[0][-1]['throughput']))
        d2 = self.scale_num(int(w2['throughput']))
        dt = max(d1, d2)/min(d1, d2)
        return math.ceil(dt)*[-1, 1][d2 > d1]

    def compute_sysbench_reward_throughput_qtune(self, experience_replay:typing.List[dict], w2:dict) -> float:
        mt, m0 = w2['throughput'], experience_replay[0][-1]['throughput']
        mt1 = experience_replay[-1][-1]['throughput']
        dt0, dt1 = (mt - m0)/m0, (mt - mt1)/mt1
        if dt0 > 0:
            return ((1 + dt1)**2 - 1)*abs(1 + dt0)
        
        return -1*((1 - dt1)**2 - 1)*abs(1 - dt0)


    def compute_sysbench_reward_throughput(self, experience_replay:typing.List[dict], w2:dict) -> float:
        return (w2['throughput'] - experience_replay[0][-1]['throughput'])/experience_replay[0][-1]['throughput']

    def compute_sysbench_reward_throughput_delta(self, experience_replay:typing.List[dict], w2:dict) -> float:
        return (w2['throughput'] - experience_replay[-1][-1]['throughput'])/experience_replay[-1][-1]['throughput']

    def compute_sysbench_reward_throughput_raw(self, experience_replay:typing.List[dict], w2:dict) -> float:
        return w2['throughput']

    def compute_sysbench_reward_throughput_max_adjust(self, experience_replay:typing.List[dict], w2:dict) -> float:
        m = max([i[-1]['throughput'] for i in experience_replay])
        w2_t = w2['throughput']
        if (w2_t - m)/m >= -0.3:
            return w2_t
        
        return w2_t - m
    

    def compute_sysbench_weighted_avg_reward(self, experience_replay:typing.List[dict], w2:dict) -> float:
        lt = (experience_replay[0][-1]['latency'] - w2['latency'])/experience_replay[0][-1]['latency']
        th = (w2['throughput'] - experience_replay[0][-1]['throughput'])/experience_replay[0][-1]['throughput']
        return (0.01*lt + th)/2

    def compute_sysbench_discounted_latency_reward(self, experience_replay:typing.List[dict], w2:dict) -> float:
        lt = (experience_replay[0][-1]['latency'] - w2['latency'])/experience_replay[0][-1]['latency']
        th = (w2['throughput'] - experience_replay[0][-1]['throughput'])/experience_replay[0][-1]['throughput']
        return 0.001*lt + th

class Atlas_Reward_Signals:
    def basic_query_workload_cost(self, *args, **kwargs) -> dict:
        full_query_cost = self.conn.workload_cost()
        return {'index': full_query_cost,
            'params': {'geo_avg_workload_cost':statistics.geometric_mean([full_query_cost[i]['cost'] for i in full_query_cost])}}

    def tpch_queries_per_hour(self, *args, **kwargs) -> dict:
        qph = self.conn.tpch_qphH_size()
        return {'qph': qph,
            'params': {
                'qph': qph
            }
        }

    def sysbench_latency_throughput(self, seconds:int = 10) -> dict:
        d = self.conn.sysbench_metrics(seconds)
        return {
            'latency': d['latency_max'],
            'throughput': d['throughput'],
            'params': {
                'latency': d['latency_max'],
                'throughput': d['throughput']
            }
        }

class Atlas_Environment:
    def sysbench_env_reset(self) -> None:
        self.conn.sysbench_cleanup_benchmark()
        self.conn.sysbench_prepare_benchmark()


class ClusterQueue:
    def __init__(self, f:str = 'cosine', dist:float = 0.002) -> None:
        def normalize(v:typing.List[float]) -> typing.List[float]:
            s = sum(v)
            return [i/s for i in v]

        def clip(v:typing.List[float]) -> typing.List[float]:
            return [min(max(0, i), 1) for i in v]

        def cosine(v1:typing.List[float], v2:typing.List[float]) -> float:
            v1, v2 = normalize(clip(v1)), normalize(clip(v2))
            return 1 - sum(a*b for a, b in zip(v1, v2))/(pow(sum(a**2 for a in v1), 0.5) * pow(sum(b**2 for b in v2), 0.5))

        def euclidian(v1:typing.List[float], v2:typing.List[float]) -> float:
            v1, v2 = normalize(clip(v1)), normalize(clip(v2))
            return pow(sum((a - b)**2 for a, b in zip(v1, v2)), 0.5)

        self.f, self.dist = f, dist
        self.clusters = []
        self.f_map = {'cosine': cosine, 'euclidian': euclidian}
        self.ind_count = 1
        self.access_cache = []

    def __len__(self) -> int:
        return len(self.clusters)

    def add_action(self, action:typing.List[float]) -> None:
        if (cluster:=[i for i in self.clusters \
                if all(self.f_map[self.f](action, j) <= self.dist for _, j in i)]):
            c = random.choice(cluster)
            c.append((self.ind_count, action))

        else:
            self.clusters.append([(self.ind_count, action)])
            self.access_cache.append(None)

        self.ind_count += 1

    def sample(self, num:int) -> typing.List[int]:
        assert num <= len(self.clusters)

        results = []
        for i in random.sample([*range(len(self.clusters))], num):
            if self.access_cache[i] is None:
                self.access_cache[i] = random.choice(self.clusters[i])[0]

            results.append(self.access_cache[i])

        return results

class ClusterCache(ClusterQueue):
    def __init__(self, f:str = 'cosine', dist:float = 0.002) -> None:
        super().__init__(f = f, dist = dist)
        self.storage = []

    def __getitem__(self, action:typing.List[float]) -> typing.Union[None, dict]:
        if (options:=[(a, b, score) for a, b in self.storage \
                if (score:=self.f_map[self.f](action, a)) <= self.dist]):
            return sorted(options, key=lambda x:x[-1])[0][1]

    def add_entry(self, action:typing.List[float], data:dict) -> None:
        self.storage.append((action, data))


class Atlas_Knob_Tune(Atlas_Rewards, Atlas_Reward_Signals, 
        Atlas_Environment):
    def __init__(self, database:str, conn = None, config = {
            'alr':0.001,
            'clr':0.001,
            'gamma':0.9,
            'noise_scale':0.5,
            'noise_decay':0.001,
            'replay_size':50,
            'batch_size':20,
            'workload_exec_time':4,
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
        self.actor_lr_scheduler = None
        self.critic_lr_scheduler = None
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
        print('actor lr:', self.config['alr'])
        print('critic lr:', self.config['clr'])
        print('weight decay', self.config['weight_decay'])
        self.actor_optimizer = optimizer.Adam(lr=self.config['alr'], params=self.actor.parameters(), weight_decay=self.config['weight_decay'])
        self.critic_optimizer = optimizer.Adam(lr=self.config['clr'], params=self.critic.parameters(), weight_decay=self.config['weight_decay'])
        
        #self.actor_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=lambda x:0.97 ** x)
        #self.critic_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=lambda x:0.97 ** x)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def update_target_weights(self, target_m, m) -> None:
        for target_param, param in zip(target_m.parameters(), m.parameters()):
            target_param.data.copy_(
                target_param.data * self.config['tau'] + param.data * (1-self.config['tau'])
            )

    def log_message(self, message:str) -> None:
        self.tuning_log.write('\n'+'#'*16+message+'#'*16)

    def tune(self, iterations:int, 
            reward_func:str = None,
            reward_signal:str = None, 
            reset_knobs:bool = True, 
            is_marl:bool = False, 
            is_epoch:bool = False) -> dict:

        if reset_knobs or is_epoch:
            print('Resetting knobs')
            knob_values = self.conn.reset_knob_configuration()

        experience_replay_f_name = f'experience_replay/ddpg_knob_tune/er_{str(time.time()).replace(".", "")}.json'

        print('update number specified', self.config['updates'])

        #metrics = db.MySQL.metrics_to_list(self.conn._metrics())
        metrics = knob_values
        indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())

        state = [*(indices if is_marl else []), *metrics]
        print('length of state', len(state))
        start_state = F.normalize(torch.tensor([[*map(float, state)]], requires_grad = True))
        state_num, action_num = len(state), db.MySQL.knob_num

        if not is_epoch or self.actor is None:
            self.init_models(state_num, action_num)

        '''
        self.log_message('Getting default metrics')
        with open('experience_replay/ddpg_knob_tune/tpcc_1000_2024-04-1815:57:59283850.json') as f:
            self.experience_replay = json.load(f)
            skip_experience = len(self.experience_replay)
        '''

        if not self.experience_replay:
            self.experience_replay = [[state, None, None, None, getattr(self, reward_signal)(self.config['workload_exec_time'])]]
        

        cq = ClusterQueue(dist = self.config['cluster_dist'])

        c_cache = ClusterCache(dist = self.config['cluster_dist'])

        rewards = []
        skip_experience = self.config['replay_size']
        noise_scale = self.config['noise_scale']
        for i in range(iterations + self.config['replay_size']):

            if self.config.get('terminate_after') is not None and self.config['noise_eliminate'] + self.config['replay_size'] + self.config['terminate_after'] <= i:
                print('terimate_after reached, training halted')
                break

            print(f'iteration {i+1} of {iterations + self.config["replay_size"]}')
            #self.log_message(f'Iteration {i+1}')
            with open(experience_replay_f_name, 'w') as f:
                json.dump([{'experience_replay':self.experience_replay, 
                    'rewards':rewards}], f)

            if (env_reset:=self.config.get('env_reset')) is not None:
                if i and not i%env_reset['steps']:
                    print('resetting environment')
                    getattr(self, env_reset['func'])()


            knob_activation_payload = {
                'memory_size':(mem_size:=self.conn.memory_size('b')[self.database]*4),
                'memory_lower_bound':min(4294967168, mem_size)
            }

            clipped_noise_scale = max(noise_scale, noise_scale if (mns:=self.config.get('min_noise_scale')) is None else mns)
            self.actor.eval()
            [selected_action] = Normalize.add_noise(self.actor(start_state).tolist(), clipped_noise_scale)
            self.actor.train()
            chosen_knobs, knob_dict = db.MySQL.activate_knob_actor_outputs(selected_action, knob_activation_payload)
            
            knob_values = self.conn.apply_knob_configuration(knob_dict)


            cq.add_action(selected_action)

            if (w2:=c_cache[selected_action]) is None:
                self.experience_replay.append([state, selected_action, 
                    reward:=getattr(self, reward_func)(self.experience_replay, w2:=getattr(self, reward_signal)(self.config['workload_exec_time'])),
                    [*(indices if is_marl else []), *(metrics:=knob_values)],
                    w2
                ])
                c_cache.add_entry(selected_action, w2)

            else:
                self.experience_replay.append([state, selected_action, 
                    reward:=getattr(self, reward_func)(self.experience_replay, w2),
                    [*(indices if is_marl else []), *(metrics:=knob_values)],
                    w2
                ])
            
            '''
            self.experience_replay.append([state, selected_action, 
                reward:=getattr(self, reward_func)(self.experience_replay, w2:=getattr(self, reward_signal)(self.config['workload_exec_time'])),
                [*(indices if is_marl else []), *(metrics:=knob_values)],
                w2
            ])
            '''
            rewards.append(reward)
            state = [*(indices if is_marl else []), *metrics]

            if (noise_eliminate:=self.config.get('noise_eliminate')) is None or i < noise_eliminate:
                noise_scale -= noise_scale*self.config['noise_decay']

            else:
                noise_scale = 0

            print('noise scale:', noise_scale)

            start_state = F.normalize(torch.tensor([[*map(float, state)]], requires_grad = True))

            if len(self.experience_replay) >= self.config['replay_size']:
                '''
                with open(e_f_file:=f"experience_replay/ddpg_knob_tune/{self.conn.database}_{str(datetime.datetime.now()).replace(' ', '').replace('.', '')}.json", 'a') as f:
                    json.dump(self.experience_replay, f)

                print('experience replay saved to:', e_f_file)
                '''
                for _ in range(self.config['updates']):
                 
                    batch_size = min(self.config['batch_size'], len(self.experience_replay)-1)
                    inds = random.sample([*range(1, len(self.experience_replay))], batch_size)
                    
                    #batch_size = min(self.config['batch_size'], len(self.experience_replay))
                    #inds = cq.sample(batch_size)

                    _s, _a, _r, _s_prime, w2 = zip(*[self.experience_replay[i] for i in inds])
                    #s = torch.tensor([Normalize.normalize(i) for i in _s])
                    s = F.normalize(torch.tensor([[*map(float, i)] for i in _s]))
                    a = torch.tensor([[float(j) for j in i] for i in _a])
                    #s_prime = torch.tensor([Normalize.normalize(i) for i in _s_prime])
                    s_prime = F.normalize(torch.tensor([[*map(float, i)] for i in _s_prime]))
                    #r = torch.tensor([[float(i)] for i in Normalize.normalize(_r)])
                    r = torch.tensor([[float(i)] for i in _r])

                    target_action = self.actor_target(s_prime).detach()

                    target_q_value = self.critic_target(s_prime, target_action).detach()
                    
                    current_value = self.critic(s, a)
                    next_value = r + self.config['gamma']*target_q_value

                    loss = self.loss_criterion(current_value, next_value)
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    self.critic_optimizer.step()

                    self.critic.eval()
                    u = self.actor(s)
                    predicted_q_value = self.critic(s, u)

                    pqv = -1*predicted_q_value
                    actor_loss = pqv.mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()

                    self.actor_optimizer.step()
                    
                    self.critic.train()

                    self.update_target_weights(self.critic_target, self.critic)
                    self.update_target_weights(self.actor_target, self.actor)

                #self.actor_lr_scheduler.step()
                #self.critic_lr_scheduler.step()
            
            
            if i and not i%self.config['marl_step']:
                yield {
                'experience_replay':self.experience_replay,
                'rewards':rewards}, False
                
        yield {
            'experience_replay':self.experience_replay,
            'rewards':rewards}, True


    def __exit__(self, *_) -> None:
        if self.conn is not None:
            self.conn.__exit__()
        
        if self.tuning_log is not None:
            self.tuning_log.close()


class CDB_Wrapper(Atlas_Knob_Tune):
    def tune(self) -> None:

        is_marl = self.config['is_marl']

        print('reseting knobs')
        knob_values = self.conn.reset_knob_configuration()
        metrics = db.MySQL.metrics_to_list(self.conn._metrics())
        indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())
        state = np.array([*(indices if is_marl else []), *metrics])
        state_num, action_num = len(state), db.MySQL.knob_num
        noisy = self.config['noisy']
        reward_signal = self.config['reward_signal']
        reward_func = self.config['reward_func']

        ddpg_opt = {
            'tau': 0.00001,
            'alr': 0.00001,
            'clr': 0.00001,
            'model':  '',
            'gamma': 0.9,
            'memory_size': 10000000,
            'batch_size': self.config['batch_size'],
        }

        model = ddpg.DDPG(
            n_states=state_num,
            n_actions=action_num,
            opt=ddpg_opt,
            mean_var_path='mean_var.pkl',
            ouprocess=not noisy
        )

        self.experience_replay = [[state.tolist(), None, None, None, getattr(self, reward_signal)(self.config['workload_exec_time'])]]
        rewards = []

        knob_activation_payload = {
            'memory_size':(mem_size:=self.conn.memory_size('b')[self.database]*4),
            'memory_lower_bound':min(4294967168, mem_size)
        }

        for iteration in range(self.config['iterations']):
            print(f"iteration {iteration + 1} of {self.config['iterations']}")
            if self.config['noisy']:
                model.sample_noise()

            action = model.choose_action(state)

            print('action chosen', action)
            chosen_knobs, knob_dict = db.MySQL.activate_knob_actor_outputs(action.tolist(), knob_activation_payload)
            
            knob_values = self.conn.apply_knob_configuration(knob_dict)

            self.experience_replay.append([state.tolist(), action.tolist(), 
                reward:=getattr(self, reward_func)(self.experience_replay, w2:=getattr(self, reward_signal)(self.config['workload_exec_time'])),
                [*(indices if is_marl else []), *(metrics:=db.MySQL.metrics_to_list(self.conn._metrics()))],
                w2
            ])

            rewards.append(reward)
            #reward, state_, done, score, metrics, restart_time = env.step(current_knob)
            
            next_state = np.array([*(indices if is_marl else []), *metrics])


            model.add_sample(state, action, reward, next_state, 0)

            state = next_state

            if len(model.replay_memory) > self.config['batch_size']:
                for _ in range(2):
                    model.update()

    
        return {
            'experience_replay':self.experience_replay,
            'rewards':rewards
        }       


class Atlas_Index_Tune(Atlas_Rewards):
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


class Atlas_Index_Tune_DQN(Atlas_Index_Tune, Atlas_Rewards, Atlas_Reward_Signals):
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
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = self.config['lr'])
        
    def random_action(self, indices:typing.List[int]) -> typing.Tuple[int, typing.List[int]]:
        ind = random.choice([*range(len(indices)+1)])
        _indices = copy.deepcopy(indices)
        if ind < len(_indices):
            _indices[ind] = int(not _indices[ind])

        return ind, _indices

    def generate_experience_replay(self, indices:typing.List[int], 
            metrics:typing.List[int], 
            iterations:int, 
            reward_func:str, 
            reward_signal:str, 
            is_marl:bool, 
            from_buffer:typing.Union[bool, str] = False) -> None:

        print('experience replay reward func:', reward_func)
        if from_buffer:
            print('loading experience replay from buffer ', from_buffer)
            with open(from_buffer) as f:
                self.experience_replay = json.load(f)
            
            return

        self.experience_replay = [[[*indices, *(metrics if is_marl else [])], 
                None, None, None, w2_o:=getattr(self, reward_signal)()]]
        for _ in range(iterations):
            ind, _indices = self.random_action(indices)

            self.conn.apply_index_configuration(_indices)
            self.experience_replay.append([[*indices, *(metrics if is_marl else [])], ind, 
                getattr(self, reward_func)(self.experience_replay,
                        w2:=getattr(self, reward_signal)()), 
                [*_indices, *(metrics if is_marl else [])], w2])
            indices = _indices

        print('experience replay saved to:')
        print(self.save_experience_replay())


    def save_experience_replay(self, f_name = None) -> str:
        f_name = f_name if f_name is not None else f"experience_replay/dqn_index_tune/experience_replay_{self.conn.database}_{str(datetime.datetime.now()).replace(' ', '').replace('.', '')}.json"
        with open(f_name, 'a') as f:
            json.dump(self.experience_replay, f)

        return f_name

    def tune(self, iterations:int, 
            reward_func:str = None, 
            reward_signal:str = None, 
            from_buffer = None, 
            reward_buffer_size:int = 150,
            is_epoch:bool = False, 
            is_marl:bool = False) -> typing.Iterator:

        print('tuning reward function:', reward_func)
        print('tuning reward signal:', reward_signal)
        self.conn.drop_all_indices()
        metrics = db.MySQL.metrics_to_list(self.conn._metrics())
        indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())

        if not self.experience_replay:
            self.generate_experience_replay(indices, metrics, reward_buffer_size, 
                reward_func, reward_signal, is_marl, from_buffer = from_buffer)
        
        state = [*indices, *(metrics if is_marl else [])]
        print('length of state in index tune', len(state), state)
        start_state = torch.tensor([Normalize.normalize(state)], requires_grad = True)
        
        action_num, state_num = len(indices)+1, len(state)

        if not is_epoch or self.q_net is None:
            print('setting weights')
            self.init_models(state_num, action_num)

        else:
            print('skipping weight setting')

        self.conn.drop_all_indices()
        
        rewards = []
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
                reward:=getattr(self, reward_func)(self.experience_replay,
                        w2:=getattr(self, reward_signal)()), 
                [*_indices, *(metrics if is_marl else [])], w2])

            rewards.append(reward)
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
                    'rewards':rewards}, False

        yield {'experience_replay':self.experience_replay, 
                    'rewards':rewards}, True
    
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


def whittaker_smoother(vals):
    w = whittaker_eilers.WhittakerSmoother(
        lmbda=70, order=1, data_length=len(vals)
    )
    return w.smooth(vals)

def sc_savgol_filter(vals):
    return savgol_filter(vals, 5, 2)

def display_tuning_results(f_name:typing.Union[str, list], cutoff = None, smoother = None, agg_f = lambda x:sum(x)/len(x)) -> None:
    f_names = [f_name] if isinstance(f_name, str) else f_name
    full_rewards, full_params = [], collections.defaultdict(list)
    for f_name in f_names:
        with open(f_name) as f:
            tuning_data = json.load(f)

        
        rewards = [sum(i)/len(i) for i in zip(*[i['rewards'] for i in tuning_data])]
        if cutoff:
            rewards = rewards[:cutoff]

        if smoother is not None:
            rewards = smoother(rewards)

        full_rewards.append(rewards)

        params = [[j[-1]['params'] for j in i['experience_replay']] for i in tuning_data]
        param_avgs = []
        for i in zip(*params):
            p = collections.defaultdict(list)
            for j in i:
                for a, b in j.items():
                    p[a].append(b)
            
            param_avgs.append({a:sum(b)/len(b) for a, b in p.items()})

        row_params = collections.defaultdict(list)
        for i in param_avgs:
            for a, b in i.items():
                row_params[a].append(b)

        for a, b in row_params.items():
            full_params[a].append(smoother(b) if smoother is not None else b)
      
   
   
    fig, [reward_plt, *param_plt] = plt.subplots(nrows=1, ncols=len(row_params) + 1)    

    rewards = [agg_f(i) for i in zip(*full_rewards)]
    print('rewards here', rewards)

    reward_plt.plot([*range(1,len(rewards)+1)], rewards, label = 'Earned reward')

    for a, k in zip(param_plt, full_params):
        row_param_vals = [agg_f(i) for i in zip(*full_params[k])]
        

        a.plot([*range(1, len(row_param_vals)+1)], row_param_vals)
        a.title.set_text(k)
        #a.legend(loc="upper right")

        a.set_xlabel("Iteration")
        a.set_ylabel(k)
    

    reward_plt.title.set_text("Reward at each iteration")
    reward_plt.legend(loc="lower right")

    reward_plt.set_xlabel("Iteration")
    reward_plt.set_ylabel("Reward")


    plt.show()


def generate_index_tune_output_file() -> str:
    ind = max(int(re.findall('\d+(?=\.json)', i)[0]) for i in os.listdir('outputs/tuning_data') if i.endswith('.json')) + 1
    return f'outputs/tuning_data/rl_dqn{ind}.json'

def generate_knob_tune_output_file() -> str:
    ind = max(int(re.findall('\d+(?=\.json)', i)[0]) for i in os.listdir('outputs/knob_tuning_data') if i.endswith('.json')) + 1
    return f'outputs/knob_tuning_data/rl_ddpg{ind}.json'

def atlas_index_tune_dqn(config:dict) -> None:
    database = config['database']
    weight_copy_interval = config['weight_copy_interval']
    epsilon = config['epsilon']
    epsilon_decay = config['epsilon_decay']
    marl_step = config['marl_step']
    iterations = config['iterations']
    reward_func = config['reward_func']
    reward_signal = config['reward_signal']
    is_marl = config['is_marl']
    epochs = config['epochs']
    reward_buffer = config['reward_buffer']
    reward_buffer_size = config['reward_buffer_size']
    batch_sample_size = config['batch_sample_size']

    with Atlas_Index_Tune_DQN(database) as a_index:
        a_index.conn.reset_knob_configuration()
        tuning_data = []
        for i in range(epochs):
            a_index.update_config(**{'weight_copy_interval':weight_copy_interval, 
                'epsilon':epsilon, 
                'epsilon_decay':epsilon_decay, 
                'marl_step':marl_step,
                'batch_sample_size': batch_sample_size
            })
            a_index_prog = a_index.tune(iterations, 
                reward_func = reward_func, 
                reward_signal = reward_signal,
                from_buffer = reward_buffer, 
                reward_buffer_size = reward_buffer_size,
                is_epoch = epochs > 1, 
                is_marl = is_marl)
            while True:
                payload, flag = next(a_index_prog)
                if flag:
                    tuning_data.append(payload)
                    break
        
        with open(rl_output_f:=generate_index_tune_output_file(), 'a') as f:
            json.dump(tuning_data, f)
        
        print('Index tuning outputs saved to', rl_output_f)
        print('index tuning complete!!')
        display_tuning_results(rl_output_f, smoother = whittaker_smoother)


def atlas_knob_tune(config:dict) -> None:
    database = config['database']
    episodes = config['episodes']
    replay_size = config['replay_size']
    noise_scale = config['noise_scale']
    noise_decay = config['noise_decay']
    batch_size = config['batch_size']
    workload_exec_time = config['workload_exec_time']
    marl_step = config['marl_step']
    iterations = config['iterations']
    reward_func = config['reward_func']
    reward_signal = config['reward_signal']
    is_marl = config['is_marl']
    alr = config['alr']
    clr = config['clr']
    tau = config['tau']
    min_noise_scale = config['min_noise_scale']
    updates = config.get('updates', 1)
    env_reset = config.get('env_reset')
    cluster_dist = config['cluster_dist']
    noise_eliminate = config.get('noise_eliminate')
    terminate_after = config.get('terminate_after', 10)
    weight_decay = config['weight_decay']

    with Atlas_Knob_Tune(database) as a_knob:
        tuning_data = []
        for _ in range(episodes):
            a_knob.update_config(**{'replay_size':replay_size, 
                    'noise_scale':noise_scale, 
                    'noise_decay':noise_decay, 
                    'batch_size':batch_size, 
                    'workload_exec_time': workload_exec_time, 
                    'marl_step':marl_step,
                    'min_noise_scale': min_noise_scale,
                    'updates': updates,
                    'env_reset': env_reset,
                    'tau': tau,
                    'noise_eliminate': noise_eliminate,
                    'cluster_dist': cluster_dist,
                    'alr': alr,
                    'clr': clr,
                    'weight_decay': weight_decay,
                    'terminate_after': terminate_after})

            a_knob_prog = a_knob.tune(iterations, 
                reward_func = reward_func, 
                reward_signal = reward_signal,
                is_marl = is_marl,
                is_epoch = episodes > 1)

            while True:
                results, flag = next(a_knob_prog)
                if flag:
                    tuning_data.append(results)
                    break

        with open(f_name:=generate_knob_tune_output_file(), 'a') as f:
            json.dump(tuning_data, f)
        
        print('knob tuning complete!')
        print('knob tuning results saved to', f_name)
        display_tuning_results(f_name, smoother = whittaker_smoother)


def atlas_knob_tune_cdb(config:dict) -> None:
    database = config['database']
    with CDB_Wrapper(database) as a_knob:
        a_knob.update_config(**config)
        tuning_data = [a_knob.tune()]

    with open(f_name:=generate_knob_tune_output_file(), 'a') as f:
        json.dump(tuning_data, f)
    
    print('knob tuning complete!')
    print('knob tuning results saved to', f_name)
    display_tuning_results(f_name, smoother = whittaker_smoother)

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

def marl_results_v2_file_results(f_name:str) -> tuple:
    with open(f_name) as f:
        data = json.load(f)

    index_rewards = data['index_results'][0]['rewards']
    knob_rewards = [(j:=sum(i)/len(i)) for i in zip(*[i['rewards'] for i in data['knob_results']][:1])]
    knob_rewards = [max(i, -1) for i in knob_rewards]
    costs = [[statistics.geometric_mean([j['cost'] for j in i['index'].values()])for i in k] for k in data['db_stats']]
    costs= [sum(i)/len(i) for i in zip(*costs)]
    return index_rewards, knob_rewards, costs

def display_marl_results_v2(f_name:str) -> None:
    index_rewards, knob_rewards, costs = marl_results_v2_file_results(f_name)
    
    fig, [a1, a2, a3] = plt.subplots(nrows=1, ncols=3)

    print(len(costs))
    a1.plot([*range(1, len(costs)+1)], costs)
    a1.title.set_text("Total workload cost at each iteration")
    a1.legend(loc="lower right")

    a1.set_xlabel("Time increment")
    a1.set_ylabel("Workload cost")

    a2.plot([*range(1, len(index_rewards)+1)], index_rewards)
    a2.title.set_text("Index Tuner Rewards")

    a2.set_xlabel("iteration")
    a2.set_ylabel("Reward")

    a3.plot([*range(1, len(knob_rewards)+1)], knob_rewards)
    a3.title.set_text("Knob Tuner Rewards")

    a3.set_xlabel("iteration")
    a3.set_ylabel("Reward")


    plt.show()

def ___marl_agg_stats_v2(blocks:typing.List[tuple]) -> None:
    fig, [a1, a2, a3] = plt.subplots(nrows=1, ncols=3)

    for l, files in blocks:
        _index_rewards, _knob_rewards, _costs = zip(*[marl_results_v2_file_results(f_name) for f_name in files])
        index_rewards = [sum(i)/len(i) for i in zip(*_index_rewards)][:150]
        knob_rewards = [sum(i)/len(i) for i in zip(*_knob_rewards)][:150]
        costs = [sum(i)/len(i) for i in zip(*_costs)][:300]
        a1.plot([*range(1, len(costs)+1)], costs, label = l)
        a2.plot([*range(1, len(index_rewards)+1)], index_rewards, label = l)
        a3.plot([*range(1, len(knob_rewards)+1)], knob_rewards, label = l)
    
    a1.legend(loc="upper right")
    a1.title.set_text("Total Workload Cost")
    a1.set_xlabel("Iteration")
    a1.set_ylabel("Workload cost")
    
    a2.title.set_text("Index Tuner Reward")

    a2.set_xlabel("Iteration")
    a2.set_ylabel("Reward")
    a2.legend(loc="lower right")
    
    a3.title.set_text("Knob Tuner Reward")

    a3.set_xlabel("Iteration")
    a3.set_ylabel("Reward")
    a3.legend(loc="lower right")

    
    plt.show()


def marl_agg_stats_v2(blocks:typing.List[tuple]) -> None:
    fig, [a1, a2] = plt.subplots(nrows=1, ncols=2)

    for l, files in blocks:
        _index_rewards, _knob_rewards, _costs = zip(*[marl_results_v2_file_results(f_name) for f_name in files])
        index_rewards = [sum(i)/len(i) for i in zip(*_index_rewards)]
        knob_rewards = [sum(i)/len(i) for i in zip(*_knob_rewards)]
        costs = [sum(i)/len(i) for i in zip(*_costs)]
        a1.plot([*range(1, len(costs)+1)], costs, label = l)
        a2.plot([*range(1, len(index_rewards)+1)], index_rewards, label = l)
        #a3.plot([*range(1, len(knob_rewards)+1)], knob_rewards, label = l)
    
    a1.legend(loc="upper right")
    a1.title.set_text("Total Workload Cost")
    a1.set_xlabel("Iteration")
    a1.set_ylabel("Workload cost")
    
    a2.title.set_text("Index Tuner Reward")

    a2.set_xlabel("Iteration")
    a2.set_ylabel("Reward")
    a2.legend(loc="lower right")
    '''
    a3.title.set_text("Knob Tuner Reward")

    a3.set_xlabel("Iteration")
    a3.set_ylabel("Reward")
    a3.legend(loc="lower right")
    '''
    
    plt.show()

def atlas_marl_tune(config:dict) -> None:
    database = config['database']
    marl_step = config['marl_step']
    epochs = config['epochs']
    iterations = config['iterations']
    is_marl = config['is_marl']
    index_reward_func = config['index_reward_func']
    knob_reward_func = config['knob_reward_func']
    output_file = config['output_file']
    from_buffer = config.get('from_buffer')

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
                    reward_func = index_reward_func,
                    from_buffer = from_buffer,
                    is_marl = is_marl,
                    is_epoch = True)

                a_knob.update_config(**{'replay_size':50, 'noise_scale':1.5, 'noise_decay':0.006, 'batch_size':40, 'tpcc_time':4, 'marl_step':marl_step})
                a_knob_prog = a_knob.tune(iterations, 
                    reward_func = knob_reward_func, 
                    is_marl = is_marl,
                    is_epoch = True)

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

            with open(output_file, 'a') as f:
                json.dump({'index_results':index_results, 'knob_results':knob_results, 'db_stats':db_stats}, f)


            #display_marl_results(output_file)
            display_marl_results_v2(output_file)


def knob_tune_action_vis(output_file:str) -> None:
    #'outputs/knob_tuning_data/rl_ddpg28.json'
    with open(output_file) as f:
        data = json.load(f)

    f_name = re.findall("\w+(?=\.json)", output_file)[0]
    with open(f'outputs/action_vis/{f_name}.txt', 'w') as f:
        f.write('\n'.join(f'{i}: {",".join(map(str,a[1]))} | {a[-1]["throughput"]}' for i, a in enumerate(data[0]['experience_replay'][1:], 1)))


def cluster(output_file:str, dist = 'cosine') -> None:
    def normalize(v:typing.List[float]) -> typing.List[float]:
        s = sum(v)
        return [i/s for i in v]

    def clip(v:typing.List[float]) -> typing.List[float]:
        return [min(max(0, i), 1) for i in v]

    def cosine(v1:typing.List[float], v2:typing.List[float]) -> float:
        v1, v2 = normalize(clip(v1)), normalize(clip(v2))
        return 1 - sum(a*b for a, b in zip(v1, v2))/(pow(sum(a**2 for a in v1), 0.5) * pow(sum(b**2 for b in v2), 0.5))

    def euclidian(v1:typing.List[float], v2:typing.List[float]) -> float:
        v1, v2 = normalize(clip(v1)), normalize(clip(v2))
        return pow(sum((a - b)**2 for a, b in zip(v1, v2)), 0.5)

    with open(output_file) as f:
        data = json.load(f)

    actions = [i[1] for i in data[0]['experience_replay'][1:]]

    #print(cosine(actions[375], actions[373]))
    #print(euclidian(actions[375], actions[373]))
    '''
    actions[369], actions[368] #similar
    6.451746813018744e-06
    0.0018070811865447459

    actions[364], actions[373] #similar
    0.0008159711495020483
    0.02030111487558261

    actions[375], actions[373] #disimilar
    0.0022559732954188227
    0.03337650973108767
    '''
    actions = actions
    cq = ClusterQueue(dist = 0.001)
    for i in actions:
        cq.add_action(i)

    m = dict(enumerate(actions))
    clusters = []
    while m:
        v1 = m.pop(i:=random.choice([*m]))
        c1 = [(i, v1)]
        for j in [*m]:
            if all(cosine(b, m[j]) <= 0.001 for _, b in c1):
                c1.append((j, m.pop(j)))

        clusters.append(c1)
    
    print([[j for j, _ in i] for i in clusters])
    print('-'*40)
    print([[j for j, _ in i] for i in cq.clusters])
    #print(len(clusters), len(cq.clusters))
    for _ in range(5):
        print(cq.sample(50))


def test_lr_annealing() -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=100)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x:0.97 ** x)
        #scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda = lambda x: 0.999 ** x)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
        lrs = []

        for i in range(600):
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        plt.plot(lrs)
        plt.show()


if __name__ == '__main__':
    '''
    atlas_index_tune_dqn({
        'database': 'tpch1',
        'weight_copy_interval': 10,
        'epsilon': 1,
        'epsilon_decay': 0.0055,
        'marl_step': 50,
        'iterations': 200,
        'reward_func': 'compute_tpch_qph_reward',
        'reward_signal': 'tpch_queries_per_hour',
        'is_marl': True,
        'epochs': 1,
        'reward_buffer': None,
        'reward_buffer_size':60,
        'batch_sample_size':50
    })
    
    #display_tuning_results('outputs/tuning_data/rl_dqn26.json')
    '''
    def test_annealing(scale, decay, iterations):
        for _ in range(iterations):
            print(f'{_}: ',scale)
            scale -= scale * decay


    def noise_scale_anneal() -> None:
        k = []
        s = 0.5
        for i in range(600):
            if i < 200:
                s -= s*0.015
            else:
                s = 0

            k.append(s)
    
        plt.plot(k)
        plt.show()

    def test_cluster_storage(output_file) -> None:
        cq = ClusterCache(dist = 0.001)

        with open(output_file) as f:
            data = json.load(f)

        for i, a in enumerate(data[0]['experience_replay'][1:], 1):
            cq.add_entry(a[1], a[-1])


        print(cq[[1.1522041145438326,0.3880128933971036,0.35030574826270866,0.5800164983550372,0.6529931816384045,0.8166568707578936]])


    '''
    atlas_knob_tune({
        'database': 'sysbench_tune',
        'episodes': 1,
        'replay_size': 60,
        'noise_scale': 0.5,
        'noise_decay': 0.01,
        'batch_size': 100,
        'min_noise_scale': None,
        'alr': 0.0001,
        'clr': 0.0001,
        'workload_exec_time': 10,
        'marl_step': 50,
        'iterations': 600,
        'cluster_dist': 0.1,
        'noise_eliminate': 250,
        'terminate_after': 300,
        'updates': 5,
        'tau': 0.999,
        'reward_func': 'compute_sysbench_reward_throughput_scaled',
        'reward_signal': 'sysbench_latency_throughput',
        'env_reset': None,
        'is_marl': True,
        'weight_decay': 0.001
    })    
    
    '''
    
    '''
    display_tuning_results([
            'outputs/knob_tuning_data/rl_ddpg66.json',
            'outputs/knob_tuning_data/rl_ddpg67.json',
            'outputs/knob_tuning_data/rl_ddpg68.json',
            'outputs/knob_tuning_data/rl_ddpg69.json',
            'outputs/knob_tuning_data/rl_ddpg71.json',
            'outputs/knob_tuning_data/rl_ddpg72.json'
        ], 
        smoother = whittaker_smoother)
    '''
    
    
    '''
    atlas_knob_tune_cdb({
        'database': 'sysbench_tune',
        'reward_func': 'compute_sysbench_reward_throughput_qtune',
        'reward_signal': 'sysbench_latency_throughput',
        'noisy': True,
        'batch_size': 50,
        'workload_exec_time': 10,
        'iterations':200,
        'is_marl': True
    })
    '''
    #knob_tune_action_vis('outputs/knob_tuning_data/rl_ddpg49.json')
    #test_cluster_storage('outputs/knob_tuning_data/rl_ddpg49.json')
    #test_annealing(0.5, 0.01, 600)
    #display_tuning_results('outputs/knob_tuning_data/rl_ddpg46.json')

    #display_tuning_results('outputs/knob_tuning_data/rl_ddpg41.json')
    #display_tuning_results('outputs/knob_tuning_data/rl_ddpg27.json')
    #display_tuning_results('outputs/knob_tuning_data/rl_ddpg25.json', smoother = whittaker_smoother)
    #display_tuning_results('outputs/knob_tuning_data/rl_ddpg19.json', smoother = whittaker_smoother)
    #display_tuning_results('outputs/knob_tuning_data/rl_ddpg24.json', smoother = whittaker_smoother)
    #display_tuning_results('outputs/knob_tuning_data/rl_ddpg17.json')
    #cluster('outputs/knob_tuning_data/rl_ddpg35.json')
