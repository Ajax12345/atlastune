import mysql_conn as db, random
import copy, json, datetime
import collections, time
import matplotlib.pyplot as plt
import statistics, os, re
import whittaker_eilers

if os.environ.get('ATLASTUNE_ENVIRONMENT') == 'CC':
    #on ubunut: sudo export ATLASTUNE_ENVIRONMENT=CC
    db.MySQL = db.MySQL_CC

def run_default_baseline(database:str, iterations:int, seconds:int, name:str) -> None:
    with db.MySQL(database = database) as conn:
        print('Resetting knobs')
        conn.reset_knob_configuration()
        results = []
        for _ in range(iterations):
            d = conn.sysbench_metrics(seconds)
            results.append({
                'latency': d['latency_max'],
                'throughput': d['throughput'],
                'params': {
                    'latency': d['latency_max'],
                    'throughput': d['throughput']
                }
            })
            #time.sleep(1)

        with open(f'outputs/comparison_results/{name}.json', 'a') as f:
            json.dump(results, f)

if __name__ == '__main__':
    run_default_baseline('sysbench_tune', 500, 10, 'default_600_10_01012025')
        
