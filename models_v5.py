'''
AtlasTune with DQN Scheduler
'''

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
import math, ddpg, tpch
import tsmoothie.smoother
import itertools

if os.environ.get('ATLASTUNE_ENVIRONMENT') == 'CC':
    #on ubunut: sudo export ATLASTUNE_ENVIRONMENT=CC
    db.MySQL = db.MySQL_CC

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

def ITER_GEN(iterations:typing.Union[int, None] = None) -> typing.Iterator:
    if iterations is not None:
        return range(iterations)

    def iter_gen_wrapper() -> typing.Iterator:
        i = 0
        while True:
            yield i
            i += 1
    
    return iter_gen_wrapper()

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

class Atlas_Scheduler_QNet(nn.Module):
    def __init__(self, state_num:int, action_num:int, hidden_layer_size:int) -> None:
        super().__init__()
        self.state_num = state_num 
        self.action_num = action_num
        self.layers = nn.Sequential(
            nn.Linear(self.state_num, hidden_layer_size),
            nn.ReLU(),
            #nn.Linear(hidden_layer_size, hidden_layer_size),
            #nn.ReLU(),
            nn.Linear(hidden_layer_size, self.action_num)
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
        _n = n
        n = abs(n)
        n_b = int((s:=str(n)[0])+'0'*(L:=len(str(n))-1))
        if n >= n_b + 5*10**(L-1):
            return [-1, 1][_n > 0]*(n_b + 10**L)

        return [-1, 1][_n > 0]*n_b

    def compute_sysbench_reward_throughput_scaled(self, experience_replay:typing.List[dict], w2:dict) -> float:
        t = self.scale_num(int(experience_replay[0][-1]['throughput']))
        return (self.scale_num(int(w2['throughput'])) - t)/t

    def compute_sysbench_reward_throughput_scaled_action_penalty(self, 
            experience_replay:typing.List[dict], w2:dict, 
            action:int, do_nothing:bool) -> float:

        if not do_nothing and experience_replay[-1][1] == action:
            return -0.5
        
        return self.compute_sysbench_reward_throughput_scaled(experience_replay, w2)

    def compute_sysbench_reward_latency_scaled(self, experience_replay:typing.List[dict], w2:dict) -> float:
        t = self.scale_num(int(experience_replay[0][-1]['latency']))
        return (t - self.scale_num(int(w2['latency'])))/t

    def compute_sysbench_reward_latency_scaled_action_penalty(self, 
            experience_replay:typing.List[dict], w2:dict, 
            action:int, do_nothing:bool) -> float:

        if not do_nothing and experience_replay[-1][1] == action:
            return -0.5
        
        return self.compute_sysbench_reward_latency_scaled(experience_replay, w2)

    def compute_sysbench_reward_latency(self, experience_replay:typing.List[dict], w2:dict) -> float:
        t = int(experience_replay[0][-1]['latency'])
        return (t - int(w2['latency']))/t

    def compute_sysbench_reward_latency_action_penalty(self, 
            experience_replay:typing.List[dict], 
            w2:dict, action:int, do_nothing:bool) -> float:
            
        if not do_nothing and experience_replay[-1][1] == action:
            return -0.5
        
        return self.compute_sysbench_reward_latency(experience_replay, w2)

    def compute_sysbench_reward_throughput_action_penalty(self, 
            experience_replay:typing.List[dict], 
            w2:dict, action:int, do_nothing:bool) -> float:

        if not do_nothing and experience_replay[-1][1] == action:
            return -0.5
        
        return self.compute_sysbench_reward_throughput(experience_replay, w2)

    def compute_sysbench_reward_marl_latency_discount(self, experience_replay:typing.List[dict], w2:dict) -> float:
        return 0.5*self.compute_sysbench_reward_latency_scaled(experience_replay, w2) + \
            self.compute_sysbench_reward_throughput_scaled(experience_replay, w2)

    def compute_sysbench_reward_marl(self, experience_replay:typing.List[dict], w2:dict) -> float:
        return self.compute_sysbench_reward_latency_scaled(experience_replay, w2) + \
            self.compute_sysbench_reward_throughput_scaled(experience_replay, w2)

    
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

    def compute_tpch_total_exec_time_scaled(self, experience_replay:typing.List[dict], w2:dict) -> float:
        lt = self.scale_num(int(experience_replay[0][-1]['total_time']))
        return (lt - self.scale_num(int(w2['total_time'])))/lt

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

    def tpch_total_exec_time(self, *args, **kwargs) -> dict:
        total_time = tpch.TPC_H.total_exec_time()
        return {
            'total_time': total_time,
            'params': {
                'total_time': total_time
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

class Atlas_States:
    def __init__(self, is_marl:bool = True) -> None:
        self.is_marl = is_marl

    def state_indices_metrics_KNOB(self, payload:dict, conn:db.MySQL) -> typing.List[float]:
        if 'metrics' not in payload:
            payload['metrics'] = db.MySQL.metrics_to_list(conn._metrics())

        if self.is_marl:
            if 'indices' not in payload:
                payload['indices'] = db.MySQL.col_indices_to_list(conn.get_columns_from_database())
            
        else:
            payload['indices'] = []

        return payload['indices'] + payload['metrics']

    def state_indices_knobs_KNOB(self, payload:dict, conn:db.MySQL) -> typing.List[float]:
        if 'knobs' not in payload:
            payload['knobs'] = conn.get_knobs()

        if self.is_marl:
            if 'indices' not in payload:
                payload['indices'] = db.MySQL.col_indices_to_list(conn.get_columns_from_database())
            
        else:
            payload['indices'] = []

        return payload['indices'] + payload['knobs']
    
    def state_indices_metrics_INDEX(self, payload:dict, conn:db.MySQL) -> typing.List[float]:
        if 'indices' not in payload:
            payload['indices'] = db.MySQL.col_indices_to_list(conn.get_columns_from_database())

        if self.is_marl:
            if 'metrics' not in payload:
                payload['metrics'] = db.MySQL.metrics_to_list(conn._metrics())

        else:
            payload['metrics'] = []

        return payload['indices'] + payload['metrics']

    def state_indices_knobs_INDEX(self, payload:dict, conn:db.MySQL) -> typing.List[float]:
        if 'indices' not in payload:
            payload['indices'] = db.MySQL.col_indices_to_list(conn.get_columns_from_database())
            
        if self.is_marl:
            if 'knobs' not in payload:
                payload['knobs'] = conn.get_knobs()
                '''
                payload['knobs'] = [random.randint(*{134217728:[134217728//2, 
                        134217728 + 134217728//2], 
                    4:[1, 32], 
                    4000:[2000, 6000]}[i]) for i in conn.get_knobs()]
                '''

        else:
            payload['knobs'] = []

        return payload['indices'] + payload['knobs']

    def state_indices_metrics(self, payload:dict, agent:str, conn:db.MySQL) -> typing.List[float]:
        return getattr(self, f'state_indices_metrics_{agent}')(payload, conn)
    
    def state_indices_knobs(self, payload:dict, agent:str, conn:db.MySQL) -> typing.List[float]:
        return getattr(self, f'state_indices_knobs_{agent}')(payload, conn)

    def state(self, state:str, payload:dict, agent:str, conn:db.MySQL) -> typing.List[float]:
        return getattr(self, state)(payload, agent, conn)


class ClusterQueue:
    def __init__(self, f:str = 'cosine', dist:float = 0.002) -> None:
        def normalize(v:typing.List[float]) -> typing.List[float]:
            s = sum(v)
            return [i/s for i in v]

        def clip(v:typing.List[float]) -> typing.List[float]:
            return [min(max(0, i), 1) for i in v]

        def cosine(v1:typing.List[float], v2:typing.List[float]) -> float:
            #v1, v2 = normalize(clip(v1)), normalize(clip(v2))
            return 1 - sum(a*b for a, b in zip(v1, v2))/(pow(sum(a**2 for a in v1), 0.5) * pow(sum(b**2 for b in v2), 0.5))

        def euclidean(v1:typing.List[float], v2:typing.List[float]) -> float:
            #v1, v2 = normalize(clip(v1)), normalize(clip(v2))
            return pow(sum((a - b)**2 for a, b in zip(v1, v2)), 0.5)

        self.f, self.dist = f, dist
        self.clusters = []
        self.f_map = {'cosine': cosine, 'euclidean': euclidean}
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

    def __getitem__(self, payload:dict) -> typing.Union[None, dict]:
        options = []
        for a, b in self.storage:
            if payload.get('direct', []) == a.get('direct', []):
                v1, v2 = payload.get('indirect', []), a.get('indirect', [])
                if not v1 and not v2:
                    options.append((a, b, 0))
                
                elif v1 and v2:
                    if (score:=self.f_map[self.f](v1, v2)) <= self.dist:
                        options.append((a, b, score))


        if options:
            return sorted(options, key=lambda x:x[-1])[0][1]

    def add_entry(self, payload:dict, data:dict) -> None:
        self.storage.append((payload, data))


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
        self.iteration = 0
        self.AGENT_STATE = {
            'noise_scale': 0,
            'noise_decay': 0,
            'cache_hit': 0,
        }

    def agent_state(self) -> typing.List:
        return [
            self.AGENT_STATE['noise_scale'],
            self.AGENT_STATE['noise_decay'],
            self.AGENT_STATE['cache_hit']/(self.iteration + 1),
        ]

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
        print('cache workload', self.config['cache_workload'])
        print('atlas state', self.config['atlas_state'])
        atlas_states = Atlas_States(is_marl)

        state = atlas_states.state(self.config['atlas_state'], {
                'knobs': knob_values
            }, 'KNOB', self.conn)

        print('starting state', state)
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
        

        if self.config.get('cluster_cache') is not None:
            c_cache = self.config['cluster_cache']
        
        else:
            c_cache = ClusterCache(f = self.config['cluster_f'], 
                dist = self.config['cluster_dist'])

        rewards = []
        skip_experience = self.config['replay_size']
        noise_scale = self.config['noise_scale']
        for i in ITER_GEN(None if iterations is None else iterations + self.config['replay_size']):
            self.iteration = i
            if self.config.get('terminate_after') is not None and self.config['noise_eliminate'] + self.config['replay_size'] + self.config['terminate_after'] <= i:
                print('terimate_after reached, training halted')
                break

            print(f'iteration {i+1} of {iterations + self.config["replay_size"] if iterations else "indefinite"}')
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

            self.AGENT_STATE['noise_scale'] = noise_scale
            clipped_noise_scale = max(noise_scale, noise_scale if (mns:=self.config.get('min_noise_scale')) is None else mns)
            self.actor.eval()
            [selected_action] = Normalize.add_noise(self.actor(start_state).tolist(), clipped_noise_scale)
            print('selected action', selected_action)
            self.actor.train()
            chosen_knobs, knob_dict = db.MySQL.activate_knob_actor_outputs(selected_action, knob_activation_payload)
            
            knob_values = self.conn.apply_knob_configuration(knob_dict)

            new_state = atlas_states.state(self.config['atlas_state'], {
                'knobs': knob_values
            }, 'KNOB', self.conn)

            if self.config['is_marl'] and (marl_state:=self.config.get('state_share')):
                marl_state['selected_action'] = selected_action

            c_c_payload = {'indirect': selected_action, 
                'direct': db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())}

            if (w2:=c_cache[c_c_payload]) is None or not self.config['cache_workload']:
                self.experience_replay.append([state, selected_action, 
                    reward:=getattr(self, reward_func)(self.experience_replay, w2:=getattr(self, reward_signal)(self.config['workload_exec_time'])),
                    new_state, w2
                ])
                c_cache.add_entry(c_c_payload, w2)

            else:
                self.AGENT_STATE['cache_hit'] += 1
                self.experience_replay.append([state, selected_action, 
                    reward:=getattr(self, reward_func)(self.experience_replay, w2),
                    new_state, w2
                ])
            
            rewards.append(reward)
            state = new_state

            self.AGENT_STATE['noise_decay'] = self.config['noise_decay']

            if (noise_eliminate:=self.config.get('noise_eliminate')) is None or i < noise_eliminate:
                noise_scale -= noise_scale*self.config['noise_decay']

            else:
                noise_scale = 0

            print('noise scale:', noise_scale)

            start_state = F.normalize(torch.tensor([[*map(float, state)]], requires_grad = True))

            if len(self.experience_replay) >= self.config['replay_size']:
                for _ in range(self.config['updates']):
                 
                    batch_size = min(self.config['batch_size'], len(self.experience_replay)-1)
                    inds = random.sample([*range(1, len(self.experience_replay))], batch_size)
                
                    _s, _a, _r, _s_prime, w2 = zip(*[self.experience_replay[i] for i in inds])
                    s = F.normalize(torch.tensor([[*map(float, i)] for i in _s]))
                    a = torch.tensor([[float(j) for j in i] for i in _a])
                    s_prime = F.normalize(torch.tensor([[*map(float, i)] for i in _s_prime]))
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
            _indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())
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
            _indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())
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


class Atlas_Index_Tune_DQN(Atlas_Index_Tune, 
        Atlas_Rewards, 
        Atlas_Reward_Signals):
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
        self.iteration = 0
        self.AGENT_STATE = {
            'epsilon': 0,
            'noise_decay': 0,
            'cache_hit': 0,
            'q_hit': 0,
        }

    def agent_state(self) -> typing.List:
        return [
            self.AGENT_STATE['epsilon'],
            self.AGENT_STATE['noise_decay'],
            self.AGENT_STATE['cache_hit']/(self.iteration + 1),
            self.AGENT_STATE['q_hit']/(self.iteration + 1)
        ]

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
            c_cache: ClusterCache,
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

        atlas_states = Atlas_States(is_marl)

        state = atlas_states.state(self.config['atlas_state'], {
                'indices': indices
            }, 'INDEX', self.conn)

        self.experience_replay = [[state, None, None, None, w2_o:=getattr(self, reward_signal)()]]
        
        c_c_payload = {'direct': indices}
        if self.config['is_marl'] and (marl_state:=self.config.get('state_share')):
            c_c_payload['indirect'] = marl_state['selected_action']

        print('index tune cc payload', c_c_payload)

        c_cache.add_entry(c_c_payload, w2_o)

        for _ in range(iterations):
            ind, _indices = self.random_action(indices)
            self.conn.apply_index_configuration(_indices)

            _indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())

            new_state = atlas_states.state(self.config['atlas_state'], {
                    'indices': _indices
                }, 'INDEX', self.conn)

            c_c_payload = {'direct': _indices}
            if self.config['is_marl'] and (marl_state:=self.config.get('state_share')):
                c_c_payload['indirect'] = marl_state['selected_action']

            print('index tune cc payload', c_c_payload)

            if (w2:=c_cache[c_c_payload]) is None or not self.config['cache_workload']:
                self.experience_replay.append([state, ind, 
                    getattr(self, reward_func)(self.experience_replay,
                            w2:=getattr(self, reward_signal)(), ind, ind == len(_indices)), new_state, w2])
                
                c_cache.add_entry(c_c_payload, w2)

            else:
                self.experience_replay.append([state, ind, 
                    getattr(self, reward_func)(self.experience_replay, w2, ind, ind == len(_indices)), new_state, w2])

            indices = _indices
            state = new_state

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
        indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())

        if self.config.get('cluster_cache') is not None:
            c_cache = self.config['cluster_cache']
        
        else:
            c_cache = ClusterCache(f = self.config['cluster_f'], 
                dist = self.config['cluster_dist'])

        if not self.experience_replay:
            self.generate_experience_replay(indices, c_cache, reward_buffer_size, 
                reward_func, reward_signal, is_marl, from_buffer = from_buffer)
        
        atlas_states = Atlas_States(is_marl)

        state = atlas_states.state(self.config['atlas_state'], {
                'indices': indices
            }, 'INDEX', self.conn)

        print('length of state in index tune', len(state), state)
        start_state = F.normalize(torch.tensor([[*map(float, state)]], requires_grad = True))
        
        action_num, state_num = len(indices)+1, len(state)

        if not is_epoch or self.q_net is None:
            print('setting weights')
            self.init_models(state_num, action_num)

        else:
            print('skipping weight setting')

        self.conn.drop_all_indices()
        
        rewards = []
        epsilon = self.config['epsilon']
        for iteration in ITER_GEN(iterations):
            self.iteration = iteration
            self.AGENT_STATE['epsilon'] = epsilon
            self.AGENT_STATE['noise_decay'] = self.config['epsilon_decay']

            print('state here', state)
            if random.random() < epsilon:
                print('random')
                ind, _indices = self.random_action(indices)

            else:
                print('q_val')
                self.AGENT_STATE['q_hit'] += 1
                with torch.no_grad():
                    ind = self.q_net(start_state).max(1)[1].item()
                    _indices = copy.deepcopy(indices)
                    if ind < len(_indices):
                        _indices[ind] = int(not _indices[ind])

            if ind != self.experience_replay[-1][1]:
                print(ind if ind < len(_indices) else 'do nothing')
                self.conn.apply_index_configuration(_indices)
                _indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())
                new_state = atlas_states.state(self.config['atlas_state'], {
                        'indices': _indices
                    }, 'INDEX', self.conn)

                c_c_payload = {'direct': _indices}
                if self.config['is_marl'] and (marl_state:=self.config.get('state_share')):
                    c_c_payload['indirect'] = marl_state['selected_action']

                print('index tune cc payload', c_c_payload)

                if (w2:=c_cache[c_c_payload]) is None or not self.config['cache_workload']:
                    self.experience_replay.append([state, ind, 
                        reward:=getattr(self, reward_func)(self.experience_replay,
                                w2:=getattr(self, reward_signal)(), ind, ind == len(_indices)), 
                        new_state, w2])
                    
                    c_cache.add_entry(c_c_payload, w2)

                else:
                    self.AGENT_STATE['cache_hit'] += 1
                    self.experience_replay.append([state, ind, 
                        reward:=getattr(self, reward_func)(self.experience_replay,
                                w2, ind, ind == len(_indices)), 
                        new_state, w2])

                rewards.append(reward)
                indices = _indices
                state = new_state
                
                start_state = F.normalize(torch.tensor([[*map(float, state)]]))
            
            else:
                print('skipping...')

            epsilon -= epsilon*self.config['epsilon_decay']

            inds = random.sample([*range(1,len(self.experience_replay))], min(self.config['batch_sample_size'], len(self.experience_replay)-1))
            _s, _a, _r, _s_prime, w2 = zip(*[self.experience_replay[i] for i in inds])
            s = F.normalize(torch.tensor([[*map(float, i)] for i in _s], requires_grad = True))
            a = torch.tensor([[i] for i in _a])
            s_prime = F.normalize(torch.tensor([[*map(float, i)] for i in _s_prime], requires_grad = True))
            r = F.normalize(torch.tensor([[float(i)] for i in _r], requires_grad = True))

            #with torch.no_grad():
            q_prime = self.q_net_target(s_prime).max(1)[0].unsqueeze(1)
            q_value = self.q_net(s).gather(1, a)
            
            target_q_value = r + self.config['gamma']*q_prime

            loss = self.loss_func(q_value, target_q_value)

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

class Atlas_Scheduler(Atlas_Rewards):
    def __init__(self, 
        history_window:int, 
        agents:list,
        conn = None,
        config = {
            'lr':0.0001,
            'gamma':0.9,
            'weight_copy_interval':10,
            'tau':0.9999,
            'epsilon':1,
            'epsilon_decay':0.001,
            'batch_sample_size':50
        }) -> None:
        self.history_window = history_window
        self.agents = agents
        self.history = []
        self.start_state = None 
        self.iteration = 0
        self.config = config
        self.q_net = None
        self.q_net_target = None
        self.loss_func = None
        self.optimizer = None
        self.experience_replay = []
        self.atlas_states = Atlas_States(True)
        self.conn = conn

    def update_config(self, **kwargs) -> None:
        self.config.update(kwargs)

    def reset_target_weights(self) -> None:
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def init_models(self, state_num:int, action_num:int) -> None:
        print('In INIT Models in Scheduler')
        self.q_net = Atlas_Scheduler_QNet(state_num, action_num, 64)
        self.q_net_target = Atlas_Scheduler_QNet(state_num, action_num, 64)
        self.reset_target_weights()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = self.config['lr'])
        self.conn.reset_knob_configuration()
        self.conn.drop_all_indices()
        self.state = self.generate_state()

    def generate_state(self) -> torch.tensor:
        window = self.history[-1*self.history_window:]
        freq = collections.Counter(self.history)
        m = [freq.get(i, 0) for i in range(1, len(self.agents)+1)]
        freq_min = min(m)
        base_state = self.atlas_states.state(self.config['atlas_state'], 
            {}, 'INDEX', self.conn)
        
        return [
            *window,
            *([0]*(self.history_window - len(window))),
            *[j for k in self.agents for j in k.agent_state()],
            #*[i - freq_min for i in m]
        ]
        
        '''
        return self.atlas_states.state(self.config['atlas_state'], 
            {}, 'INDEX', self.conn)
        '''
        #return db.MySQL.metrics_to_list(self.conn.metrics(5,1))
        
    def schedule(self) -> int:
        if self.q_net is None:
            #self.init_models(self.history_window + self.agents, self.agents)
            start_state = self.generate_state()
            print('start state in scheduler', start_state)
            self.init_models(len(start_state), len(self.agents))

        if self.iteration < self.config['replay_buffer_size'] or random.random() < self.config['epsilon']:
            chosen_agent = random.choice([*range(len(self.agents))])
            print('random chosen schedule agent', chosen_agent)

        else:
            with torch.no_grad():
                state = F.normalize(torch.tensor([[*map(float, self.state)]]))
                chosen_agent = (OUT:=self.q_net(state)).max(1)[1].item()
                print('q_val chosen schedule agent', chosen_agent)
                print('q val here', OUT)

        return chosen_agent

    def schedule_train(self, chosen_agent:int, reward:float) -> None:

        if 'schedule_decay' in self.config:
            self.config['epsilon'] = self.config['schedule_decay'](self.iteration, self.config['epsilon'])
        
        else:
            self.config['epsilon'] -= self.config['epsilon']*self.config['epsilon_decay']

        self.history.append(chosen_agent + 1)
        self.experience_replay.append([
            self.state, chosen_agent, reward, (new_state:=self.generate_state())
        ])

        self.state = new_state

        if self.iteration >= self.config['replay_buffer_size']:
            print("IN SCHEDULE TRAIN")
            for _ in range(70):
                samples = random.sample(self.experience_replay, min(self.config['batch_sample_size'], len(self.experience_replay)))
                _s, _a, _r, _s_prime = zip(*samples)
                s = F.normalize(torch.tensor([[*map(float, i)] for i in _s], requires_grad = True))
                a = torch.tensor([[i] for i in _a])
                s_prime = F.normalize(torch.tensor([[*map(float, i)] for i in _s_prime], requires_grad = True))
                r = F.normalize(torch.tensor([[float(i)] for i in _r], requires_grad = True))

                #with torch.no_grad():
                q_prime = self.q_net_target(s_prime).max(1)[0].unsqueeze(1)
                q_value = self.q_net(s).gather(1, a)
                
                target_q_value = r + self.config['gamma']*q_prime

                loss = self.loss_func(q_value, target_q_value)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.iteration and not self.iteration%self.config['weight_copy_interval']:
                self.reset_target_weights()

        self.iteration += 1

class Atlas_Scheduler_Similarity(Atlas_Scheduler):
    def _generate_state(self, history:typing.List[int]) -> typing.List:
        window = history[-1*self.history_window:]
        freq = collections.Counter(self.history)
        m = [freq.get(i, 0) for i in range(1, self.agents+1)]
        freq_min = min(m)
        return [
            *window,
            *([0]*(self.history_window - len(window))),
            #*[i - freq_min for i in m]
        ]

    def generate_state(self) -> torch.tensor:
        window = self.history[-1*self.history_window:]
        freq = collections.Counter(self.history[-50:])
        m = [freq.get(i, 0) for i in range(1, self.agents+1)]
        freq_min = min(m)
        return [
            *window,
            *([0]*(self.history_window - len(window))),
            #*[i - freq_min for i in m]
        ]
    
    def schedule(self) -> int:

        if self.iteration < self.config['replay_buffer_size'] or random.random() < self.config['epsilon']:
            chosen_agent = random.choice([*range(self.agents)])
            print('random chosen schedule agent', chosen_agent)

        else:
            
            cos = nn.CosineSimilarity(dim=0)
            agent_estimates = []
            for agent in range(1, self.agents + 1):
                s_prime = torch.tensor(self._generate_state(self.history + [agent]))
                #print(s_prime)
                targets = [(1 - cos(s_prime, torch.tensor([*map(float, s)])).item(), s, r) 
                    for s, r in self.experience_replay]
                
                if (window:=[r for score, _, r in targets if score <= self.config['cosine_score']]):
                    agent_estimates.append((agent, sum(window)/len(window)))
            
            if not agent_estimates:
                chosen_agent = random.choice([*range(self.agents)])
                print('no q estimates found, exploring', chosen_agent)

            else:
                _chosen_agent, _ = max(agent_estimates, key=lambda x:x[1])
                chosen_agent = _chosen_agent - 1
                print('q estimate agent', chosen_agent)            

        return chosen_agent

    def schedule_train(self, chosen_agent:int, reward:float) -> None:

        if 'schedule_decay' in self.config:
            self.config['epsilon'] = self.config['schedule_decay'](self.iteration, self.config['epsilon'])
        
        else:
            self.config['epsilon'] -= self.config['epsilon']*self.config['epsilon_decay']

        self.history.append(chosen_agent + 1)
        self.experience_replay.append([
            (new_state:=self.generate_state()), reward
        ])
        self.state = new_state
        self.iteration += 1

        with open('similarity_model_buffer.json', 'w') as f:
            json.dump({
                'epsilon': self.config['epsilon'],
                'iteration': self.iteration,
                'state': self.state,
                'experience_replay': self.experience_replay
            }, f)

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


def whittaker_smoother(vals, *args, **kwargs):
    w = whittaker_eilers.WhittakerSmoother(
        lmbda=70, order=1, data_length=len(vals)
    )
    return w.smooth(vals)

def sc_savgol_filter(vals, *args, **kwargs):
    return savgol_filter(vals, 15, 5)

def convolution_smoother(vals, *args, **kwargs):
    c = tsmoothie.smoother.ConvolutionSmoother(window_len=300, window_type='ones')
    c.smooth(vals)
    return c.smooth_data[0]

def display_tuning_results(f_name:typing.Union[str, list], 
    cutoff = None, 
    smoother = None, 
    agg_f = lambda x:sum(x)/len(x),
    y_axis_lim:dict = {},
    plot_titles:dict = {},
    title:typing.Union[str, None] = None,
    legend_loc = {}) -> None:

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
      
   
    with open('outputs/comparison_results/default_b_600.json') as f:
        default_benchmark = collections.defaultdict(list)
        workload_data = json.load(f)
        for i in workload_data:
            for a, b in i['params'].items():
                default_benchmark[a].append(b)

    a_r, wd = Atlas_Rewards(), [[i] for i in workload_data]
    
    baseline_rewards = [a_r.compute_sysbench_reward_throughput_scaled(wd, i) for i in workload_data]

    fig, [reward_plt, *param_plt] = plt.subplots(nrows=1, ncols=len(row_params) + 1)    

    rewards = [agg_f(i) for i in zip(*full_rewards)]

    reward_plt.plot([*range(1,len(rewards)+1)], rewards, label = 'Earned reward')
    
    b_r = baseline_rewards + [random.choice(baseline_rewards) for _ in range(max(0, len(rewards) - len(baseline_rewards)))]
    reward_plt.plot([*range(1,len(b_r)+1)], b_r if smoother is None else smoother(b_r), label = 'Baseline workload')

    if 'reward' in y_axis_lim:
        reward_plt.set_ylim(y_axis_lim['reward'])

    for a, k in zip(param_plt, full_params):
        row_param_vals = [agg_f(i) for i in zip(*full_params[k])]
        

        a.plot([*range(1, len(row_param_vals)+1)], row_param_vals, label = 'Recommended configuration')


        d_b = default_benchmark[k] + [random.choice(default_benchmark[k]) for _ in range(max(0, len(row_param_vals) - len(default_benchmark[k])))]

        a.plot([*range(1, len(d_b)+1)], d_b if smoother is None else smoother(d_b), label = f'Default configuration')

        a.legend(loc=legend_loc.get(k, 'lower right'))

        if k in y_axis_lim:
            a.set_ylim(y_axis_lim[k])

        a.title.set_text(plot_titles.get(k, k))
        #a.legend(loc="upper right")

        a.set_xlabel("Iteration")
        a.set_ylabel(k)
    

    reward_plt.title.set_text(plot_titles.get('reward', "Reward at each iteration"))
    reward_plt.legend(loc="lower right")

    reward_plt.set_xlabel("Iteration")
    reward_plt.set_ylabel("Reward")

    if title:
        plt.suptitle(title, size = 16)

    plt.show()


def generate_index_tune_output_file() -> str:
    ind = max(int(re.findall('\d+(?=\.json)', i)[0]) for i in os.listdir('outputs/tuning_data') if i.endswith('.json')) + 1
    return f'outputs/tuning_data/rl_dqn{ind}.json'

def generate_knob_tune_output_file() -> str:
    ind = max(int(re.findall('\d+(?=\.json)', i)[0]) for i in os.listdir('outputs/knob_tuning_data') if i.endswith('.json')) + 1
    return f'outputs/knob_tuning_data/rl_ddpg{ind}.json'

def generate_marl_output_file() -> str:
    ind = max(int(re.findall('\d+(?=\.json)', i)[0]) for i in os.listdir('outputs/marl_tuning_data') if i.endswith('.json')) + 1
    return f'outputs/marl_tuning_data/marl{ind}.json'

def atlas_index_tune_dqn(config:dict) -> None:
    lr = config['lr']
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
    atlas_state = config['atlas_state']
    cache_workload = config['cache_workload']
    cluster_dist = config['cluster_dist']
    cluster_f = config['cluster_f']
    cluster_queue = config.get('cluster_queue')

    with Atlas_Index_Tune_DQN(database) as a_index:
        a_index.conn.reset_knob_configuration()
        tuning_data = []
        for i in range(epochs):
            a_index.update_config(**{
                'lr': lr,
                'weight_copy_interval':weight_copy_interval, 
                'epsilon':epsilon, 
                'epsilon_decay':epsilon_decay, 
                'marl_step':marl_step,
                'batch_sample_size': batch_sample_size,
                'atlas_state': atlas_state,
                'cache_workload': cache_workload,
                'cluster_dist': cluster_dist,
                'cluster_f': cluster_f,
                'is_marl': is_marl,
                'cluster_queue': cluster_queue
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
    cluster_f = config['cluster_f']
    noise_eliminate = config.get('noise_eliminate')
    terminate_after = config.get('terminate_after', 10)
    weight_decay = config['weight_decay']
    cache_workload = config['cache_workload']
    is_cc = config['is_cc']
    atlas_state = config['atlas_state']
    cluster_cache = config.get('cluster_cache')

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
                    'cluster_f': cluster_f,
                    'alr': alr,
                    'clr': clr,
                    'weight_decay': weight_decay,
                    'cache_workload': cache_workload,
                    'atlas_state': atlas_state,
                    'cluster_cache': cluster_cache,
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
        if not is_cc:
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

def rolling_average(vals:typing.List[float], window:int = 30, f = min) -> typing.List[float]:
    #vals = whittaker_smoother(vals)
    t = [i for j in range(0, len(vals), window) for i in [f(vals[j:j+window]) for k in vals[j:j+window]]]
    '''
    r, l = [], None
    for _, b in itertools.groupby(t):
        if l is None:
            r.extend(l:=[*b])
        else:
            k = [*b]
            r.extend([(i + min(l))/2 for i in k])
            l = k
    
    return whittaker_smoother(r)
    '''
    return whittaker_smoother(t)
    

def marl_lt_th(files:typing.List[str], splice_ep:bool, lb:str, marl_step:int) -> tuple:
    d = collections.defaultdict(list)
    for f_name in files:
        with open(f_name) as f:
            if splice_ep:
                knob_tuner = (dt:=json.load(f))['knob_results'][0]['experience_replay']
                index_tuner = dt['index_results'][0]['experience_replay'][60:]
                
                lt, th = [], []
                while knob_tuner or index_tuner:
                    if lb != 'Non-MARL':
                        lt.extend([i[-1]['latency'] for i in index_tuner[:marl_step]])
                        th.extend([i[-1]['throughput'] for i in index_tuner[:marl_step]])
                        index_tuner = index_tuner[marl_step:]

                        lt.extend([i[-1]['latency'] for i in knob_tuner[:marl_step]])
                        th.extend([i[-1]['throughput'] for i in knob_tuner[:marl_step]])
                        knob_tuner = knob_tuner[marl_step:]

                    else:
                        lt.extend([i[-1]['latency'] for i in knob_tuner[:marl_step]])
                        th.extend([i[-1]['throughput'] for i in knob_tuner[:marl_step]])
                        knob_tuner = knob_tuner[marl_step:]

                        lt.extend([i[-1]['latency'] for i in index_tuner[:marl_step]])
                        th.extend([i[-1]['throughput'] for i in index_tuner[:marl_step]])
                        index_tuner = index_tuner[marl_step:]

                d['latency'].append(lt)
                d['throughput'].append(th)

            else:
                print('in here!!')
                data = json.load(f)['db_stats'][0]
                d['latency'].append([j['latency'] for j in data])
                d['throughput'].append([j['throughput'] for j in data])

    return [sum(i)/len(i) for i in zip(*d['latency'])], [sum(i)/len(i) for i in zip(*d['throughput'])]
    
    
def display_marl_results(file_payload:typing.List[tuple], 
    y_axis_lim:dict = {},
    marl_step:int = 100,
    splice_ep:bool = True,
    smoother_depth:int = 1,
    smoother = whittaker_smoother) -> None:

    def run_smoother(x, *args, **kwargs) -> typing.List[float]:
        for _ in range(smoother_depth):
            if smoother is not None:
                x = smoother(x, *args, **kwargs)
        
        return x

    
    fig, [a1, a2] = plt.subplots(nrows=1, ncols=2)


    #print(lt)
    
    if 'latency' in y_axis_lim:
        a1.set_ylim(y_axis_lim['latency'])
    

    if 'throughput' in y_axis_lim:
        a2.set_ylim(y_axis_lim['throughput'])

    with open('outputs/comparison_results/default_600_10_01012025.json') as f:
        baseline = json.load(f)

    baseline_lt = run_smoother([i['latency'] for i in baseline], f = max)
    baseline_th = run_smoother([i['throughput'] for i in baseline], f = min)

    
    for data, lb, marl_step in file_payload:
        lt, th = marl_lt_th(data, splice_ep, lb, marl_step)
        a1.plot([*range(1, len(lt)+1)], run_smoother(lt, f = max) if smoother is not None else lt, label = f'latency ({lb})')
        a2.plot([*range(1, len(th)+1)], run_smoother(th, f = min) if smoother is not None else th, label = f'throughput ({lb})')


    
    a1.plot([*range(1, len(lt)+1)], (baseline_lt+[baseline_lt[-1] for _ in range(len(lt) - len(baseline_lt))])[:len(lt)], label = 'latency (baseline)')

    a2.plot([*range(1, len(th)+1)], (baseline_th+[baseline_th[-1] for _ in range(len(th) - len(baseline_th))])[:len(th)], label = 'throughput (baseline)')

    a1.title.set_text("Latency")
    a1.legend(loc="upper right")

    

    a1.set_xlabel("iteration")
    a1.set_ylabel("latency")


    a2.title.set_text("Throughput")
    a2.legend(loc="lower right")

    a2.set_xlabel("iteration")
    a2.set_ylabel("throughput")

    plt.show()


class MARL_State_Share:
    def __init__(self, **kwargs:dict) -> None:
        self.params = kwargs

    def __getitem__(self, key:str) -> typing.Any:
        return self.params[key]
    
    def __setitem__(self, key:str, val:typing.Any) -> None:
        self.params[key] = val

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.params})'

class Agent:
    def __init__(self, name:str, agent:typing.Union[Atlas_Index_Tune_DQN, 
        Atlas_Knob_Tune], agent_stream:typing.Iterator) -> None:
        
        self.name = name
        self.agent = agent
        self.agent_stream = agent_stream

    def agent_state(self) -> list:
        return self.agent.agent_state()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name})'

def atlas_marl_tune_interleaved(config:dict) -> None:
    database = config['database']
    marl_step = config['marl_step']
    epochs = config['epochs']

    knob_tune_config = config['knob_tune_config']
    index_tune_config = config['index_tune_config']
    scheduler_config = config['scheduler_config']

    c_cache = ClusterCache(f = config['cluster_f'], 
                dist = config['cluster_dist'])

    knob_tune_config['cluster_cache'] = c_cache
    index_tune_config['cluster_cache'] = c_cache

    knob_tune_config['marl_step'] = marl_step
    index_tune_config['marl_step'] = marl_step

    state_share = MARL_State_Share()

    with Atlas_Index_Tune_DQN(database) as a_index, \
        Atlas_Knob_Tune(database, conn = a_index.conn) as a_knob:
        #a_index.conn.reset_knob_configuration()
        knob_activation_payload = {
            'memory_size':(mem_size:=a_index.conn.memory_size('b')[a_index.conn.database]*4),
            'memory_lower_bound':min(4294967168, mem_size)
        }
        state_share['selected_action'] = a_index.conn.default_selected_action(knob_activation_payload)
        
        knob_tune_config['state_share'] = state_share
        index_tune_config['state_share'] = state_share

        knob_tune_config['iterations'] = None
        index_tune_config['iterations'] = None

        AGENT_STATS = {
            'index_results': [],
            'knob_results': [],
            'db_stats': [],
            'scheduler_rewards': []
        }

        a_index.update_config(**index_tune_config)

        a_index_prog = a_index.tune(index_tune_config['iterations'], 
            reward_func = index_tune_config['reward_func'], 
            reward_signal = index_tune_config['reward_signal'],
            from_buffer = index_tune_config['reward_buffer'], 
            reward_buffer_size = index_tune_config['reward_buffer_size'],
            is_epoch = index_tune_config['epochs'] > 1, 
            is_marl = index_tune_config['is_marl'])

        a_knob.update_config(**knob_tune_config)

        a_knob_prog = a_knob.tune(knob_tune_config['iterations'], 
            reward_func = knob_tune_config['reward_func'], 
            reward_signal = knob_tune_config['reward_signal'],
            is_marl = knob_tune_config['is_marl'],
            is_epoch = knob_tune_config['episodes'] > 1)

        iteration_db_stats = []

        agents = [
            Agent('index', a_index, a_index_prog),
            Agent('knob', a_knob, a_knob_prog),
        ]


        output_file = generate_marl_output_file()
        print('marl tuning results saved to', output_file)

        for iteration in ITER_GEN(scheduler_config['iterations']):
            print('SCHEDULER ITERATION', iteration)
            chosen_agent = iteration%len(agents)
            print('selected agent', agents[chosen_agent].name)
            payload, _ = next(agents[chosen_agent].agent_stream)
            ep = payload['experience_replay'][-1*marl_step:]
            iteration_db_stats.extend([i[-1] for i in ep])
            AGENT_STATS['db_stats'] = [iteration_db_stats]

            with open(output_file, 'w') as f:
                json.dump(AGENT_STATS, f)

    print('marl tuning results saved to', output_file)

def atlas_marl_tune(config:dict) -> None:
    database = config['database']
    marl_step = config['marl_step']
    epochs = config['epochs']

    knob_tune_config = config['knob_tune_config']
    index_tune_config = config['index_tune_config']
    scheduler_config = config['scheduler_config']

    c_cache = ClusterCache(f = config['cluster_f'], 
                dist = config['cluster_dist'])

    knob_tune_config['cluster_cache'] = c_cache
    index_tune_config['cluster_cache'] = c_cache

    knob_tune_config['marl_step'] = marl_step
    index_tune_config['marl_step'] = marl_step

    state_share = MARL_State_Share()

    with Atlas_Index_Tune_DQN(database) as a_index, \
        Atlas_Knob_Tune(database, conn = a_index.conn) as a_knob:
        #a_index.conn.reset_knob_configuration()
        knob_activation_payload = {
            'memory_size':(mem_size:=a_index.conn.memory_size('b')[a_index.conn.database]*4),
            'memory_lower_bound':min(4294967168, mem_size)
        }
        state_share['selected_action'] = a_index.conn.default_selected_action(knob_activation_payload)
        
        knob_tune_config['state_share'] = state_share
        index_tune_config['state_share'] = state_share

        knob_tune_config['iterations'] = None
        index_tune_config['iterations'] = None

        AGENT_STATS = {
            'index_results': [],
            'knob_results': [],
            'db_stats': [],
            'scheduler_rewards': []
        }

        def REWARD_KNOB(th:float) -> int:
            if th > 0:
                return 1

            if th < 0:
                return 0

            return th

        def REWARD_INDEX(th:float) -> int:
            if th > 0:
                return 1
            
            if th < 0:
                return 0
            
            return th

        REWARD_SCALES = {
            0: REWARD_INDEX,
            1: REWARD_KNOB
        }

        a_index.update_config(**index_tune_config)

        a_index_prog = a_index.tune(index_tune_config['iterations'], 
            reward_func = index_tune_config['reward_func'], 
            reward_signal = index_tune_config['reward_signal'],
            from_buffer = index_tune_config['reward_buffer'], 
            reward_buffer_size = index_tune_config['reward_buffer_size'],
            is_epoch = index_tune_config['epochs'] > 1, 
            is_marl = index_tune_config['is_marl'])

        a_knob.update_config(**knob_tune_config)

        a_knob_prog = a_knob.tune(knob_tune_config['iterations'], 
            reward_func = knob_tune_config['reward_func'], 
            reward_signal = knob_tune_config['reward_signal'],
            is_marl = knob_tune_config['is_marl'],
            is_epoch = knob_tune_config['episodes'] > 1)

        iteration_db_stats = []

        agents = [
            Agent('index', a_index, a_index_prog),
            Agent('knob', a_knob, a_knob_prog),
        ]

        scheduler = Atlas_Scheduler(
            scheduler_config['history_window'], 
            agents,
            conn = a_index.conn
        )
        scheduler.update_config(**scheduler_config)

        output_file = generate_marl_output_file()
        print('marl tuning results saved to', output_file)

        for iteration in ITER_GEN(scheduler_config['iterations']):
            print('agent states', [i.agent_state() for i in agents])
            chosen_agent = scheduler.schedule()
            print('selected agent', agents[chosen_agent].name)
            payload, _ = next(agents[chosen_agent].agent_stream)
            ep = payload['experience_replay'][-1*marl_step:]
            iteration_db_stats.extend([i[-1] for i in ep])
            reward = getattr(scheduler, scheduler_config['reward_func'])(ep, ep[-1][-1])
            
            '''
            windows = [[*b] for _, b in itertools.groupby(scheduler.history + [chosen_agent + 1])]
            if windows and len(windows[-1]) >= 10:
                reward = -5
            '''
            #reward = REWARD_SCALES[int(chosen_agent)](reward)

            print('scheduler reward', reward)

            scheduler.schedule_train(chosen_agent, reward)
            AGENT_STATS['scheduler_rewards'].append(reward)

            AGENT_STATS['db_stats'] = [iteration_db_stats]

            with open(output_file, 'w') as f:
                json.dump(AGENT_STATS, f)

    print('marl tuning results saved to', output_file)

def schedule_decay(iteration:int, epsilon:float) -> float:
    if iteration < 120:
        return epsilon - epsilon * 0.006
    return epsilon - epsilon * 0.03

def schedule_decay_steep(iteration:int, epsilon:float) -> float:
    return epsilon - epsilon * 0.02

def tune_marl() -> None:

    atlas_marl_tune({
        'database': 'sysbench_tune',
        'epochs': 1,
        'marl_step': 25,
        'cluster_dist': 0.1,
        'cluster_f': 'cosine',
        'scheduler_config': {
            'epsilon': 1,
            'iterations': 200,
            'history_window': 20,
            'replay_buffer_size': 20,
            'reward_func': 'compute_sysbench_reward_throughput',
            'epsilon_decay': 0.013,
            'atlas_state': 'state_indices_knobs',
            'schedule_decay': schedule_decay_steep,
            'cosine_score': 0.2,
            'weight_copy_interval': 10,
        },
        'knob_tune_config': {
            'database': 'sysbench_tune',
            'episodes': 1,
            'replay_size': 50,
            'noise_scale': 0.5,
            'noise_decay': 0.001,
            'batch_size': 100,
            'min_noise_scale': None,
            'alr': 0.0001,
            'clr': 0.0001,
            'workload_exec_time': 10,
            'marl_step': 25,
            'iterations': None,
            'cluster_dist': 0.1,
            'cluster_f': 'cosine',
            'noise_eliminate': 800,
            'terminate_after': None,
            'updates': 5,
            'tau': 0.999,
            'reward_func': 'compute_sysbench_reward_throughput',
            'reward_signal': 'sysbench_latency_throughput',
            'env_reset': None,
            'is_marl': True,
            'cache_workload': True,
            'is_cc': True,
            'atlas_state': 'state_indices_knobs',
            'weight_decay': 0.001
        },
        'index_tune_config': {
            'database': 'sysbench_tune',
            'weight_copy_interval': 10,
            'epsilon': 1,
            'lr': 0.0001,
            'epsilon_decay': 0.00099,
            'marl_step': 25,
            'iterations': None,
            'reward_func': 'compute_sysbench_reward_throughput_action_penalty',
            'reward_signal': 'sysbench_latency_throughput',
            'atlas_state': 'state_indices_knobs',
            'cluster_dist': 0.1,
            'cluster_f': 'cosine',
            'cache_workload': True,
            'is_marl': True,
            'epochs': 1,
            'reward_buffer': 'experience_replay/dqn_index_tune/experience_replay_sysbench_tune_2025-01-0114:40:33978759.json',
            'reward_buffer_size':60,
            'batch_sample_size':200
        }
    })

def tune_index() -> None:
    atlas_index_tune_dqn({
        'database': 'sysbench_tune',
        'weight_copy_interval': 10,
        'epsilon': 1,
        'lr': 0.0001,
        'epsilon_decay': 0.005,
        'marl_step': 50,
        'iterations': 600,
        'reward_func': 'compute_sysbench_reward_latency_action_penalty',
        'reward_signal': 'sysbench_latency_throughput',
        'atlas_state': 'state_indices_knobs',
        'cluster_dist': 0.1,
        'cluster_f': 'cosine',
        'cache_workload': True,
        'is_marl': False,
        'epochs': 1,
        'reward_buffer': None,
        'reward_buffer_size':60,
        'batch_sample_size':50
    })

def tune_marl_interleaved() -> None:
    atlas_marl_tune_interleaved({
        'database': 'sysbench_tune',
        'epochs': 1,
        'marl_step': 25,
        'cluster_dist': 0.1,
        'cluster_f': 'cosine',
        'scheduler_config': {
            'epsilon': 1,
            'iterations': 100,
            'history_window': 20,
            'replay_buffer_size': 20,
            'reward_func': 'compute_sysbench_reward_throughput',
            'epsilon_decay': 0.013,
            'atlas_state': 'state_indices_knobs',
            'schedule_decay': schedule_decay_steep,
            'cosine_score': 0.2,
            'weight_copy_interval': 10,
        },
        'knob_tune_config': {
            'database': 'sysbench_tune',
            'episodes': 1,
            'replay_size': 50,
            'noise_scale': 0.5,
            'noise_decay': 0.001,
            'batch_size': 100,
            'min_noise_scale': None,
            'alr': 0.0001,
            'clr': 0.0001,
            'workload_exec_time': 10,
            'marl_step': 25,
            'iterations': None,
            'cluster_dist': 0.1,
            'cluster_f': 'cosine',
            'noise_eliminate': 800,
            'terminate_after': None,
            'updates': 5,
            'tau': 0.999,
            'reward_func': 'compute_sysbench_reward_throughput',
            'reward_signal': 'sysbench_latency_throughput',
            'env_reset': None,
            'is_marl': True,
            'cache_workload': True,
            'is_cc': True,
            'atlas_state': 'state_indices_knobs',
            'weight_decay': 0.001
        },
        'index_tune_config': {
            'database': 'sysbench_tune',
            'weight_copy_interval': 10,
            'epsilon': 1,
            'lr': 0.0001,
            'epsilon_decay': 0.00099,
            'marl_step': 25,
            'iterations': None,
            'reward_func': 'compute_sysbench_reward_throughput_action_penalty',
            'reward_signal': 'sysbench_latency_throughput',
            'atlas_state': 'state_indices_knobs',
            'cluster_dist': 0.1,
            'cluster_f': 'cosine',
            'cache_workload': True,
            'is_marl': True,
            'epochs': 1,
            'reward_buffer': 'experience_replay/dqn_index_tune/experience_replay_sysbench_tune_2025-01-0114:40:33978759.json',
            'reward_buffer_size':60,
            'batch_sample_size':200
        }
    })

if __name__ == '__main__':
    #outputs/tuning_data/rl_dqn35.json
    #tune_marl()
    tune_marl_interleaved()
    '''
    display_marl_results(
        [([
        'outputs/marl_tuning_data/marl59.json',
        'outputs/marl_tuning_data/marl60.json',
        'outputs/marl_tuning_data/marl62.json',
        'outputs/marl_tuning_data/marl63.json',
        #'outputs/marl_tuning_data/marl65.json'
        ], 'MARL', 50),
        (['outputs/marl_tuning_data/marl68.json', 'outputs/marl_tuning_data/marl69.json'], 'non-MARL', 50)
        ],
        splice_ep = False, smoother=rolling_average, smoother_depth = 15
    )
    '''
    '''
    def test():
        s = 1
        for i in range(200):
            print(i, s)
            s = schedule_decay(i, s)
    '''
    #outputs/marl_tuning_data/marl58.json hard-coded rewards

    #outputs/marl_tuning_data/marl59.json th diff rewards
    #outputs/marl_tuning_data/marl60.json th diff rewards
    '''
    scheduler = Atlas_Scheduler(10, 2)

    scheduler.update_config(**{
            'epsilon': 1,
            'iterations': 200,
            'history_window': 20,
            'replay_buffer_size': 20,
            'reward_func': 'compute_sysbench_reward_throughput',
            'epsilon_decay': 0.013,
            'atlas_state': 'state_indices_knobs',
            'schedule_decay': schedule_decay_steep,
            'batch_time_steps': 5,
            'weight_copy_interval': 10,
        })

    for _ in range(200):
        chosen_agent = scheduler.schedule()
        scheduler.schedule_train(chosen_agent, random.choice([-1,0,1]))
    '''
    '''
    scheduler = Atlas_Scheduler_Similarity(10, 2)
    scheduler.update_config(**{
            'epsilon': 1,
            'iterations': 200,
            'history_window': 20,
            'replay_buffer_size': 20,
            'reward_func': 'compute_sysbench_reward_throughput',
            'epsilon_decay': 0.013,
            'atlas_state': 'state_indices_knobs',
            'schedule_decay': schedule_decay_steep,
            'cosine_score': 0.03,
            'weight_copy_interval': 10,
        })
    
    for _ in range(200):
        agent = scheduler.schedule()
        scheduler.schedule_train(agent, random.choice([-2, -1, 0, 1, 2]))
    '''