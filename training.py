import models_v4 as tuner

def tune_index() -> None:
    tuner.atlas_index_tune_dqn({
        'database': 'sysbench_tune',
        'weight_copy_interval': 10,
        'epsilon': 1,
        'lr': 0.0001,
        'epsilon_decay': 0.005,
        'marl_step': 50,
        'iterations': 600,
        'reward_func': 'compute_sysbench_reward_latency_scaled',
        'reward_signal': 'sysbench_latency_throughput',
        'atlas_state': 'state_indices_knobs',
        'cluster_dist': 0.1,
        'cluster_f': 'cosine',
        'cache_workload': True,
        'is_marl': True,
        'epochs': 1,
        'reward_buffer': None,
        'reward_buffer_size':60,
        'batch_sample_size':50
    })

    tuner.display_tuning_results('outputs/tuning_data/rl_dqn32.json', 
        smoother = whittaker_smoother, legend_loc = {'latency': 'upper left', 'throughput': 'center'})


def tune_knob() -> None:
    tuner.atlas_knob_tune({
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
        'cluster_f': 'cosine',
        'noise_eliminate': 300,
        'terminate_after': 300,
        'updates': 5,
        'tau': 0.999,
        'reward_func': 'compute_sysbench_reward_throughput_scaled',
        'reward_signal': 'sysbench_latency_throughput',
        'env_reset': None,
        'is_marl': True,
        'cache_workload': True,
        'is_cc': True,
        'atlas_state': 'state_indices_knobs',
        'weight_decay': 0.001
    })    

    tuner.display_tuning_results([
            'outputs/knob_tuning_data/rl_ddpg78.json'
        ], 
        smoother = whittaker_smoother,
        y_axis_lim = {
            'reward': [-1, 1],
            'latency': [0, 800],
            'throughput': [0, 800]
        }, 
        plot_titles = {
            'reward': 'Reward (Caching)',
            'latency': 'Latency (Caching)',
            'throughput': 'Throughput (Caching)'
        },
        title = 'Caching')

    tuner.display_tuning_results([
            'outputs/knob_tuning_data/rl_ddpg78.json'
        ], 
        smoother = whittaker_smoother,
        y_axis_lim = {
            'reward': [-1, 1],
            'latency': [0, 800],
            'throughput': [0, 500]
        }, 
        plot_titles = {
            'reward': 'Reward (Caching)',
            'latency': 'Latency (Caching)',
            'throughput': 'Throughput (Caching)'
        },
        title = 'Caching')




    tuner.knob_tune_action_vis('outputs/knob_tuning_data/rl_ddpg82.json')

def tune_knob_cdb() -> None:

    tuner.atlas_knob_tune_cdb({
        'database': 'sysbench_tune',
        'reward_func': 'compute_sysbench_reward_throughput_qtune',
        'reward_signal': 'sysbench_latency_throughput',
        'noisy': True,
        'batch_size': 50,
        'workload_exec_time': 10,
        'iterations':200,
        'is_marl': True
    })

def state_encapsulation() -> None:

    with db.MySQL(database = "sysbench_tune") as conn:
        s = Atlas_States(False)
        print(s.state_indices_knobs({
            'indices': db.MySQL.col_indices_to_list(conn.get_columns_from_database()),
            'knobs': conn.get_knobs()
        }, 'INDEX', conn))

def tune_marl() -> None:

    tuner.atlas_marl_tune({
        'database': 'sysbench_tune',
        'epochs': 1,
        'marl_step': 10,
        'cluster_dist': 0.1,
        'cluster_f': 'cosine',
        'scheduler_config': {
            'epsilon': 1,
            'iterations': 100,
            'history_window': 10
            'replay_buffer_size': 50,
            'reward_func': 'compute_sysbench_reward_throughput_scaled',
            'epsilon_decay': 0.01
        },
        'knob_tune_config': {
            'database': 'sysbench_tune',
            'episodes': 1,
            'replay_size': 10,
            'noise_scale': 0.5,
            'noise_decay': 0.02,
            'batch_size': 10,
            'min_noise_scale': None,
            'alr': 0.0001,
            'clr': 0.0001,
            'workload_exec_time': 10,
            'marl_step': 10,
            'iterations': None,
            'cluster_dist': 0.1,
            'cluster_f': 'cosine',
            'noise_eliminate': 400,
            'terminate_after': None,
            'updates': 5,
            'tau': 0.999,
            'reward_func': 'compute_sysbench_reward_throughput_scaled',
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
            'epsilon_decay': 0.02,
            'marl_step': 10,
            'iterations': None,
            'reward_func': 'compute_sysbench_reward_throughput_scaled',
            'reward_signal': 'sysbench_latency_throughput',
            'atlas_state': 'state_indices_knobs',
            'cluster_dist': 0.1,
            'cluster_f': 'cosine',
            'cache_workload': True,
            'is_marl': True,
            'epochs': 1,
            'reward_buffer': None,
            'reward_buffer_size':60,
            'batch_sample_size':200
        }
    })

    tuner.display_marl_results([([
        'outputs/marl_tuning_data/marl47.json',
        'outputs/marl_tuning_data/marl48.json',
        'outputs/marl_tuning_data/marl49.json',
        'outputs/marl_tuning_data/marl50.json'
        ], 'MARL', 100),
        ([
            'outputs/marl_tuning_data/marl51.json',
            'outputs/marl_tuning_data/marl52.json',
        ], 'Non-MARL', 1000)], smoother=rolling_average, smoother_depth = 15
    )

    tuner.display_marl_results(
        [(['outputs/marl_tuning_data/marl54.json'], 'MARL', 50)],
        splice_ep = False, smoother=rolling_average, smoother_depth = 15
    )

if __name__ == '__main__':
    pass