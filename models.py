import typing, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import mysql_conn as db, random
import copy, json, datetime
import collections, time
import matplotlib.pyplot as plt

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

class Atlas_Index_Critic(nn.Module):
    def __init__(self, state_num:int, action_num:int, val_num:int) -> None:
        super().__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.val_num = val_num
        self.state_input = nn.Linear(self.state_num, 128)
        self.action_input = nn.Linear(self.action_num, 128)
        self.act = nn.Tanh()
        '''
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
        '''
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

class Atlas_Index_Tune:
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

    def workload_cost(self, w:dict) -> float:
        return sum(w[i]['cost'] for i in w)

    def compute_cost_delta(self, experience_replay:typing.List[dict], costs:typing.List[dict], w2:dict) -> float:
        j = self.workload_cost(experience_replay[0][-1]) 
        k = self.workload_cost(w2)
        if j > k:
            return 5
        
        if j < k:
            return -5

        return 1

    def compute_min_step_cost(self, experience_replay:typing.List[dict], costs:typing.List[dict], w2:dict) -> float:
        return min(map(self.workload_cost, costs[-50:])) - self.workload_cost(w2)

    def compute_increment_step_cost(self, experience_replay:typing.List[dict], costs:typing.List[dict], w2:dict) -> float:
        return (self.workload_cost(costs[-1]) - self.workload_cost(w2))/self.workload_cost(costs[0])

    def compute_cost_delta_per_query(self, experience_replay:typing.List[dict], costs:typing.List[dict], w2:dict) -> float:
        w1 = experience_replay[0][-1]
        k = [(float(w1[a]['cost']) - float(b['cost']))/w1[a]['cost'] for a, b in w2.items()]
        return max(min((sum(k)/len(k))*10, 10), -10)

    def compute_total_cost_reward(self, _, costs:typing.List[dict], w:dict) -> float:
        return -1*self.workload_cost(w)

    def compute_ranking_reward(self, experience_replay:typing.List[list], costs:typing.List[dict], w2:dict) -> float:
        c = [self.workload_cost(i[-1]) for i in experience_replay]
        w2_c = self.workload_cost(w2)
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
        k = [j for a, b in w2.items() if (j:=((float(w1[a]['cost']) - float(b['cost']))/w1[a]['cost']))]
        if not k:
            return 1

        return max(min((sum(k)/len(k))*10, 10), -10)
        '''
        return -0.5 if not (l:=(sum(k)/len(k))*10) else l
        '''

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


class Atlas_Index_Tune_DQN(Atlas_Index_Tune):
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
        ind = random.choice([*range(len(indices))])
        _indices = copy.deepcopy(indices)
        _indices[ind] = int(not _indices[ind])
        return ind, _indices

    def generate_experience_replay(self, indices:typing.List[int], metrics:typing.List[int], iterations:int, reward_func:str, from_buffer:typing.Union[bool, str] = False) -> None:
        print('experience replay reward func:', reward_func)
        if from_buffer:
            print('loading experience replay from buffer ', from_buffer)
            with open(from_buffer) as f:
                self.experience_replay = json.load(f)
            
            return

        self.experience_replay = [[[*indices, *metrics], None, None, None, w2_o:=self.conn.workload_cost()]]
        costs = [w2_o]
        for _ in range(iterations):
            ind, _indices = self.random_action(indices)

            self.conn.apply_index_configuration(_indices)
            self.experience_replay.append([[*indices, *metrics], ind, 
                getattr(self, reward_func)(self.experience_replay, costs,
                        w2:=self.conn.workload_cost()), 
                [*_indices, *metrics], w2])
            indices = _indices
            costs.append(w2)

        print('experience replay saved to:')
        print(self.save_experience_replay())


    def save_experience_replay(self, f_name = None) -> str:
        f_name = f_name if f_name is not None else f"experience_replay/dqn_index_tune/experience_replay_{self.conn.database}_{str(datetime.datetime.now()).replace(' ', '').replace('.', '')}.json"
        with open(f_name, 'a') as f:
            json.dump(self.experience_replay, f)

        return f_name

    def tune(self, iterations:int, reward_func:str = 'compute_cost_delta_per_query', is_epoch:bool = False) -> dict:
        print('tuning reward function:', reward_func)
        self.conn.drop_all_indices()
        metrics = db.MySQL.metrics_to_list(self.conn._metrics())
        indices = db.MySQL.col_indices_to_list(self.conn.get_columns_from_database())

        self.generate_experience_replay(indices, metrics, 150, reward_func)
        state = [*indices, *metrics]
        start_state = torch.tensor([Normalize.normalize(state)], requires_grad = True)
        
        action_num, state_num = len(indices), len(state)

        if not is_epoch or self.q_net is None:
            print('setting weights')
            self.init_models(state_num, action_num)

        else:
            print('skipping weight setting')

        self.conn.drop_all_indices()
        
        rewards, costs = [], [self.conn.workload_cost()]
        delta_rewards = []
        for iteration in range(iterations):
            if random.random() < self.config['epsilon']:
                print('random')
                ind, _indices = self.random_action(indices)

            else:
                print('q_val')
                with torch.no_grad():
                    ind = self.q_net(start_state).max(1)[1].item()
                    _indices = copy.deepcopy(indices)
                    _indices[ind] = int(not _indices[ind])
                
            
            self.conn.apply_index_configuration(_indices)
            self.experience_replay.append([state, ind, 
                reward:=getattr(self, reward_func)(self.experience_replay, costs,
                        w2:=self.conn.workload_cost()), 
                [*_indices, *metrics], w2])

            rewards.append(reward)
            delta_rewards.append(self.compute_cost_delta_per_query(self.experience_replay, costs, w2))
            costs.append(w2)
            indices = _indices
            state = [*indices, *metrics]
            start_state = torch.tensor([Normalize.normalize(state)])
            self.config['epsilon'] -= self.config['epsilon']*self.config['epsilon_decay']

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

        return {'rewards':rewards, 'delta_rewards':delta_rewards, 'costs':costs}
    
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
        a3.title.set_text("Delta workload cost at each iteration")
        a3.legend(loc="lower right")

        a3.set_xlabel("iteration")
        a3.set_ylabel("Workload cost")

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
    

if __name__ == '__main__':

    #display_tuning_results('outputs/tuning_data/rl_dqn3.json')
    

    with Atlas_Index_Tune_DQN('tpch1') as a:
        '''
        tuning_data = []
        for i in range(4):
            print(i + 1)
            a.update_config(**{'weight_copy_interval':10, 'epsilon':1, 'epsilon_decay':0.006})
            tuning_data.append(a.tune(500, reward_func = 'compute_total_cost_reward'))
        
        with open(f'outputs/tuning_data/rl_dqn12.json', 'a') as f:
            json.dump(tuning_data, f)
        
        '''
        display_tuning_results('outputs/tuning_data/rl_dqn12.json')
        
        
    
        '''
        
        with open(f'outputs/rl_dqn7.json') as f:
            rewards= json.load(f)
            plt.plot([*range(1,len(rewards[0])+1)], [sum(i)/len(i) for i in zip(*rewards)], label="dqn7")

        with open(f'outputs/rl_ddpg12.json') as f:
            rewards = json.load(f)
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
        
        with open(f'outputs/rl_dqn4.json') as f:
            rewards = json.load(f)
            plt.plot([*range(1,len(rewards[0])+1)], [sum(i)/len(i) for i in zip(*rewards)], label="dqn4")
        
        
        with open(f'outputs/rl_dqn5.json') as f:
            rewards = json.load(f)
            plt.plot([*range(1,len(rewards[0])+1)], [sum(i)/len(i) for i in zip(*rewards)], label="dqn5")
        
        
        plt.title("reward at each iteration (5 epochs)")
        plt.legend(loc="lower right")

        plt.xlabel("iteration")
        plt.ylabel("reward")
        plt.show()
        '''
        
        

       
