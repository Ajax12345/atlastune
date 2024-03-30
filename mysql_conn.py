import mysql.connector, typing
import contextlib, csv, time
import collections

#https://dev.mysql.com/doc/connector-python/en/connector-python-example-connecting.html
#https://dev.mysql.com/doc/refman/8.0/en/innodb-parameters.html

def DB_EXISTS(requires_db = True) -> typing.Callable:
    def main_f(f:typing.Callable) -> typing.Callable:
        def wrapper(self, *args, **kwargs) -> typing.Any:
            if self.cur is None:
                raise Exception('cursor is not open')
            
            if requires_db and self.database is None:
                raise Exception('no database has been specified')
            
            return f(self, *args, **kwargs)

        return wrapper

    return main_f

class MySQL:
    VALUE_METRICS = [
        'lock_deadlocks', 'lock_timeouts', 'lock_row_lock_time_max',
        'lock_row_lock_time_avg', 'buffer_pool_size', 'buffer_pool_pages_total',
        'buffer_pool_pages_misc', 'buffer_pool_pages_data', 'buffer_pool_bytes_data',
        'buffer_pool_pages_dirty', 'buffer_pool_bytes_dirty', 'buffer_pool_pages_free',
        'trx_rseg_history_len', 'file_num_open_files', 'innodb_page_size'
    ]
    KNOBS = {
        ###'skip_name_resolve': ['enum', ['OFF', 'ON']],
        'table_open_cache': ['integer', [1, 10240, 512]],
        #'max_connections': ['integer', [1100, 100000, 80000]],
        'innodb_buffer_pool_size': ['integer', [1048576, 'memory_size', 'memory_size']],
        'innodb_buffer_pool_instances': ['integer', [1, 64, 8]],
        #1
        #'innodb_log_files_in_group': ['integer', [2, 100, 2]],
        #1
        #'innodb_log_file_size': ['integer', [134217728, 5497558138, 15569256448]],
        'innodb_purge_threads': ['integer', [1, 32, 1]],
        'innodb_read_io_threads': ['integer', [1, 64, 12]],
        'innodb_write_io_threads': ['integer', [1, 64, 12]],
        #3
        #'max_binlog_cache_size': ['integer', [4096, 4294967296, 18446744073709547520]],
        #'binlog_cache_size': ['integer', [4096, 4294967296, 18446744073709547520]],
        #'max_binlog_size': ['integer', [4096, 1073741824, 1073741824]],
        ###'innodb_adaptive_flushing_lwm': ['integer', [0, 70, 10]],
        ###'innodb_adaptive_max_sleep_delay': ['integer', [0, 1000000, 150000]],
        #4
        #'innodb_change_buffer_max_size': ['integer', [0, 50, 25]],
        #'innodb_flush_log_at_timeout': ['integer', [1, 2700, 1]],
        #'innodb_flushing_avg_loops': ['integer', [1, 1000, 30]],
        #'innodb_max_purge_lag': ['integer', [0, 4294967295, 0]],
        ###'innodb_old_blocks_pct': ['integer', [5, 95, 37]],
        'innodb_read_ahead_threshold': ['integer', [0, 64, 56]],
        #2
        #'innodb_replication_delay': ['integer', [0, 10000, 0]],
        #'innodb_rollback_segments': ['integer', [1, 128, 128]],
        'innodb_sync_array_size': ['integer', [1, 1024, 1]],
        'innodb_sync_spin_loops': ['integer', [0, 100, 30]],
        'innodb_thread_concurrency': ['integer', [0, 100, 0]],
        #1
        #'lock_wait_timeout': ['integer', [1, 31536000, 31536000]],
        ###'metadata_locks_cache_size': ['integer', [1, min(memory_size, 1048576), 1024]],
        'metadata_locks_hash_instances': ['integer', [1, 1024, 8]],
        #2
        #'binlog_order_commits': ['boolean', ['OFF', 'ON']],
        #'innodb_adaptive_flushing': [' boolean', ['OFF', 'ON']],
        'innodb_adaptive_hash_index': ['boolean', ['ON', 'OFF']],
        #1
        #'innodb_autoextend_increment': [' integer', [1, 1000, 64]],  # mysql 5.6.6: 64, mysql5.6.5: 8
        ###'innodb_buffer_pool_dump_at_shutdown': ['boolean', ['OFF', 'ON']],
        ###'innodb_buffer_pool_load_at_startup': ['boolean', ['OFF', 'ON']],
        ###'innodb_concurrency_tickets': ['integer', [1, 50000, 5000]],  # 5.6.6: 5000, 5.6.5: 500
        ###'innodb_disable_sort_file_cache': [' boolean', ['ON', 'OFF']],
        #2
        #'innodb_large_prefix': ['boolean', ['OFF', 'ON']],
        #'innodb_log_buffer_size': ['integer', [262144, min(memory_size, 4294967295), 67108864]],
        'tmp_table_size': ['integer', [1024, 1073741824, 1073741824]],
        #2
        #'innodb_max_dirty_pages_pct': ['numeric', [0, 99, 75]],
        #'innodb_max_dirty_pages_pct_lwm': ['numeric', [0, 99, 0]],
        'innodb_random_read_ahead': ['boolean', ['ON', 'OFF']],
        ###'eq_range_index_dive_limit': ['integer', [0, 2000, 200]],
        ###'max_length_for_sort_data': ['integer', [4, 10240, 1024]],
        ###'read_rnd_buffer_size': ['integer', [1, min(memory_size, 5242880), 524288]],
        'table_open_cache_instances': ['integer', [1, 64, 16]],
        'thread_cache_size': ['integer', [0, 1000, 512]],
        #1
        #'max_write_lock_count': ['integer', [1, 18446744073709551615, 18446744073709551615]],
        ###'query_alloc_block_size': ['integer', [1024, min(memory_size, 134217728), 8192]],
        ###'query_cache_limit': ['integer', [0, min(memory_size, 134217728), 1048576]],
        ###'query_cache_size': ['integer', [0, min(memory_size, int(memory_size*0.5)), 0]],
        ###'query_cache_type': ['enum', ['ON', 'DEMAND', 'OFF']],
        ###'query_prealloc_size': ['integer', [8192, min(memory_size, 134217728), 8192]],
        ###'transaction_prealloc_size': ['integer', [1024, min(memory_size, 131072), 4096]],
        ###'join_buffer_size': ['integer', [128, min(memory_size, 26214400), 262144]],
        #1
        #'max_seeks_for_key': ['integer', [1, 18446744073709551615, 18446744073709551615]],
        ###'sort_buffer_size': ['integer', [32768, min(memory_size, 134217728), 524288]],
        'innodb_io_capacity': ['integer', [100, 2000000, 20000]],
        'innodb_lru_scan_depth': ['integer', [100, 10240, 1024]],
        ###'innodb_old_blocks_time': ['integer', [0, 10000, 1000]],
        #1
        #'innodb_purge_batch_size': ['integer', [1, 5000, 300]],
        'innodb_spin_wait_delay': ['integer', [0, 60, 6]],
        'innodb_adaptive_hash_index_parts': ['integer', [1, 512, 8]],
        'innodb_page_cleaners': ['integer', [1, 64, 4]],
        'innodb_flush_neighbors': ['enum', [0, 2, 1]], 

        # two ## is not allowed, one # is allowed but not need
        ##'max_heap_table_size': ['integer', [16384, min(memory_size, 1844674407370954752), 16777216]],
        ##'transaction_alloc_block_size': ['integer', [1024, min(memory_size, 131072), 8192]],
        ##'range_alloc_block_size': ['integer', [4096, min(memory_size, 18446744073709551615), 4096]],
        ##'query_cache_min_res_unit': ['integer', [512, min(memory_size, 18446744073709551615), 4096]],
        ##'sql_buffer_result' : ['boolean', ['ON', 'OFF']],
        ##'max_prepared_stmt_count' : ['integer', [0, 1048576, 1000000]],
        ##'max_digest_length' : ['integer', [0, 1048576, 1024]],
        ##'max_binlog_stmt_cache_size': ['integer', [4096, min(memory_size, 18446744073709547520),
        ##                                            18446744073709547520]],
        ## 'innodb_numa_interleave' : ['boolean', ['ON', 'OFF']],
        ##'binlog_max_flush_queue_time' : ['integer', [0, 100000, 0]],
        #'innodb_commit_concurrency': ['integer', [0, 1000, 0]],
        ##'innodb_additional_mem_pool_size': ['integer', [2097152,min(memory_size,4294967295), 8388608]],
        #'innodb_thread_sleep_delay' : ['integer', [0, 1000000, 10000]],
        ##'thread_stack' : ['integer', [131072, memory_size, 524288]],
        #'back_log' : ['integer', [1, 65535, 900]],
    }
    def __init__(self, host:str = "localhost", user:str = "root", 
                passwd:str = "Gobronxbombers2", database:typing.Union[str, None] = None,
                create_cursor:bool = True) -> None:
        self.host, self.user = host, user
        self.passwd, self.database = passwd, database
        self.conn = mysql.connector.connect(
            host = host,
            user = user,
            passwd = passwd,
            database = database
        )
        self.cur = None
        if create_cursor:
            self.new_cur()

    def new_cur(self) -> 'cursor':
        self.cur = self.conn.cursor(dictionary = True)
        return self.cur

    @DB_EXISTS()
    def status(self) -> typing.List[dict]:
        self.cur.execute('show global status')
        return [*self.cur]

    @DB_EXISTS(requires_db = False)
    def memory_size(self) -> dict:
        self.cur.execute("""select table_schema db_name, round(sum(data_length + index_length), 1) size 
        from information_schema.tables group by table_schema;""")
        return {i['db_name']:int(i['size']) for i in self.cur}


    @DB_EXISTS(requires_db = False)
    def _metrics(self) -> dict:
        self.cur.execute('select name, count from information_schema.innodb_metrics where status="enabled" order by name;')
        return {i['name']:int(i['count']) for i in self.cur}

    @DB_EXISTS(requires_db = False)
    def metrics(self, total_time:int, interval:int = 5) -> list:
        total_metrics = collections.defaultdict(list)
        while total_time > 0:
            for a, b in self._metrics().items():
                total_metrics[a].append(b)

            time.sleep(interval)
            total_time -= interval

        return {a:sum(b)/len(b) if a in self.__class__.VALUE_METRICS else float(b[-1] - b[0])
            for a, b in total_metrics.items()}        

    @DB_EXISTS(requires_db = False)
    def use_db(self, db:str) -> None:
        self.database = db
        self.cur.execute(f'use {db}')

    @DB_EXISTS()
    def execute(self, *args, **kwargs) -> 'cursor':
        self.cur.execute(*args, **kwargs)
        return self.cur

    @DB_EXISTS()
    def get_tables(self) -> typing.Any:
        self.cur.execute("show tables")
        return [*self.cur]

    @DB_EXISTS()
    def get_columns(self, tbl:str) -> typing.List[dict]:
        self.cur.execute("""
        select t.table_schema, t.table_name, t.column_name, t.ordinal_position,
            s.index_schema, s.index_name, s.seq_in_index, s.index_type 
        from information_schema.columns t
        left join information_schema.statistics s on t.table_name = s.table_name
            and t.table_schema = s.table_schema 
            and lower(s.column_name) = lower(t.column_name)
        where t.table_schema = %s and t.table_name = %s
        order by t.ordinal_position""", [self.database, tbl])
        return [*self.cur]

    @DB_EXISTS()
    def get_indices(self, tbl:str) -> typing.List[dict]:
        self.cur.execute(f"""show index from {tbl}""")
        return [*self.cur]

    @DB_EXISTS(requires_db = False)
    def get_knobs(self) -> dict:
        self.cur.execute("show variables where variable_name in ({})".format(', '.join(f"'{i}'" for i in self.__class__.KNOBS)))
        return {i['Variable_name']:i['Value'] for i in self.cur}

    @classmethod
    def metrics_to_list(cls, metrics:dict) -> typing.List[int]:
        assert metrics, "metrics must contain data"
        return [metrics[i] for i in sorted(metrics)]

    @classmethod
    def col_indices_to_list(cls, cols:typing.List[dict]) -> typing.List:
        assert cols, "table must contain columns"
        return [int(i['INDEX_NAME'] is not None) for i in cols]


    def commit(self) -> None:
        self.conn.commit()

    def __enter__(self) -> 'MySQL':
        return self

    def __exit__(self, *_) -> None:
        if self.cur is not None:
            with contextlib.suppress():
                self.cur.close()

        self.conn.close()




if __name__ == '__main__':
    with MySQL(database = "atlas_stuff") as conn:
        '''
        conn.execute("create table test_stuff (id int, first_col int, second_col int, third_col int)")
        conn.execute("create index test_index on test_stuff (first_col)")
        conn.execute("create index another_index on test_stuff (second_col, third_col)")
        conn.commit()
        '''
        #print(conn.metrics(20))
        #print(len(MySQL.KNOBS))
        #print(conn.memory_size())
        #print(conn.get_knobs())
        #print(MySQL.metrics_to_list(conn._metrics()))
        #print(MySQL.metrics_to_list(conn.metrics(20)))
        print(MySQL.col_indices_to_list(conn.get_columns("test_stuff")))