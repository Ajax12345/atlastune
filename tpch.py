import mysql_conn as db, typing, os
import subprocess, concurrent.futures
import re, time, functools

if os.environ.get('ATLASTUNE_ENVIRONMENT') == 'CC':
    #on ubunut: sudo export ATLASTUNE_ENVIRONMENT=CC
    db.MySQL = db.MySQL_CC

class TPC_H:
    LOCATION = 'tpc/tpch/tpch_official/dbgen'
    CREATION_SCRIPTS = 'tpc/tpch/tbl_creation_scripts'
    QUERIES = [
        #(1, 20),
        (2, 0.32),
        (3, 13.31),
        (4, 12.96),
        #(5, 32.51),
        (6, 9.32),
        (7, 18.30),
        #(8, 47.44),
        (9, 38.30),
        #(10, 15.65),
        (11, 1.26),
        (12, 11.06),
        (13, 9.68),
        (14, 9.95),
        (15, None),
        (16, 0.56),
        #(17, 600),
        #(18, 600),
        (19, 16.35),
        #(20, 600),
        #(21, 39.40),
        (22, 5.21)
    ]
    def __init__(self, conn: db.MySQL) -> None:
        self.conn = conn
        assert self.conn.database.startswith('tpch')

    def create_tbls(self) -> None:
        for i in os.listdir(self.__class__.CREATION_SCRIPTS):
            if i.endswith('.sql'):
                with open(os.path.join(self.__class__.CREATION_SCRIPTS, i)) as f:
                    self.conn.execute(f.read())
        
        self.conn.commit()

    def load_tbls(self) -> None:
        #NOTE: run following line first:
        #SET GLOBAL local_infile=1;
        for i in self.__class__.tbl_files():
            p = os.path.join(os.getcwd(), self.__class__.LOCATION, i)
            tbl, _ = i.split('.')
            self.conn.execute(f'''
            LOAD DATA LOCAL INFILE '{p}' INTO TABLE {tbl} FIELDS TERMINATED BY '|';
            ''')

        self.conn.commit()
            

    @classmethod
    def tbl_files(cls) -> typing.List[str]:
        return [i for i in os.listdir(cls.LOCATION) if i.endswith('.tbl')]

    @classmethod
    def remove_tbl_files(cls) -> None:
        for i in cls.tbl_files():
            os.remove(os.path.join(cls.LOCATION, i))

    @classmethod
    def generate_tbl_files(cls, size:int) -> None:
        results = str(subprocess.run(['./dbgen', '-s', str(size)], 
            cwd = cls.LOCATION, capture_output=True).stdout)

    @classmethod
    def update_files(cls) -> typing.List[str]:
        return [i for i in os.listdir(cls.LOCATION) \
            if re.findall('delete\.\d+$', i) or re.findall('\.tbl\.u\d+$', i)]

    @classmethod
    def delete_update_files(cls) -> None:
        for i in cls.update_files():
            os.remove(os.path.join(cls.LOCATION, i))

    @classmethod
    def generate_update_files(cls, size:int, streams:int) -> None:
        results = str(subprocess.run(['./dbgen', '-s', str(size), '-U', str(streams)], 
            cwd = cls.LOCATION, capture_output=True).stdout)

    @classmethod
    def rf1_stream(cls) -> float:
        with db.MySQL(database = 'tpch_tune') as s:
            return cls.rf1(s)
    
    @classmethod
    def rf2_stream(cls) -> float:
        with db.MySQL(database = 'tpch_tune') as s:
            return cls.rf2(s)


    @classmethod
    def rf1(cls, stream, stream_id:int = 1) -> float:
        total_time = 0
        for i in cls.update_files():
            if i.endswith(f'.tbl.u{stream_id}'):
                tbl, *_ = i.split('.')
                with open(os.path.join(cls.LOCATION, i)) as f:
                    for j in f:
                        if j:
                            row = j.split('|')
                            t1 = time.time()
                            try:
                                stream.execute(f'''insert into {tbl} values ({", ".join('%s' for _ in row)})''', row)
                                total_time += time.time() - t1
                            except:
                                pass

        t1 = time.time()
        stream.commit()
        total_time += time.time() - t1
        return total_time

    @classmethod
    def rf2(cls, stream, stream_id:int = 1) -> float:
        total_time = 0
        for i in cls.update_files():
            if i == f'delete.{stream_id}':
                with open(os.path.join(cls.LOCATION, i)) as f:
                    rows = [*f.readlines()][:4]
                    for j in rows:
                        if j:
                            _id, _ = j.split('|')
                            t1 = time.time()
                            stream.execute(f'''delete from orders where o_orderkey = %s''', [int(_id)])
                            stream.execute(f'''delete from lineitem where l_orderkey = %s''', [int(_id)])
                            total_time += time.time() - t1

        t1 = time.time()
        stream.commit()
        total_time += time.time() - t1
        return total_time

    @classmethod
    def query_run(cls, stream, max_time:int = 60) -> typing.List[float]:
        with open('tpc/tpch/queries.sql') as f:
            queries = [i.strip(';\n') for i in f if i]

        timings = []
        for query, t in cls.QUERIES:
            if t is not None and t <= max_time:
                t1 = time.time()
                _ = [*stream.execute(queries[query-1])]
                timings.append(time.time() - t1)

        return timings

    @classmethod
    def query_run_with_stream(cls) -> typing.List[float]:
        with db.MySQL(database = 'tpch_tune') as s:
            return cls.query_run(s)
    
    @classmethod
    def power_test(cls) -> float:
        with db.MySQL(database = 'tpch_tune') as refresh_stream:
            with db.MySQL(database = 'tpch_tune') as query_stream:
                rf1_t = cls.rf1(refresh_stream)
                q_timings = cls.query_run(query_stream)
                rf2_t = cls.rf2(refresh_stream)
        
            return (3600*2)/pow(functools.reduce(lambda x, y:x*y, q_timings) * rf1_t * rf2_t, 1/(len(q_timings) + 2))

    @classmethod
    def throughput_test(cls) -> float:
        with concurrent.futures.ProcessPoolExecutor() as pool:
            t = time.time()
            qs1 = pool.submit(cls.query_run_with_stream)
            qs2 = pool.submit(cls.query_run_with_stream)
            rf_1 = pool.submit(cls.rf1_stream)
            rf_2 = pool.submit(cls.rf2_stream)

            q1_timings = qs1.result()
            _ = qs2.result()
            _ = rf_1.result()
            _ = rf_2.result()
            dt = time.time() - t

            return (2*len(q1_timings)*3600)/dt * 2

    @classmethod
    def qph_size(cls) -> float:
        cls.generate_update_files(2, 2)
        pt = cls.power_test()
        tt = cls.throughput_test()
        cls.delete_update_files()
        return pow(pt*tt, 0.5)

    @classmethod
    def total_exec_time(cls) -> float:
        cls.generate_update_files(2, 2)
        with db.MySQL(database = 'tpch_tune') as conn:
            q = cls.query_run(conn)
            #print('queries done')
            t1 = cls.rf1(conn)
            t2 = cls.rf2(conn)
            cls.delete_update_files()
            return pow(functools.reduce(lambda x, y: x*y, q+[t1, t2]), 1/(len(q) + 2))

if __name__ == '__main__':
    with db.MySQL(database = "tpch_tune") as conn:
        #TPC_H.generate_update_files(2, 2)
        #print(TPC_H.update_files())
        #print(sum(a for a, b in TPC_H.QUERIES if b is not None and b < 60))
        #print(TPC_H.qph_size())
        print(TPC_H.total_exec_time())
        #print(conn.memory_size('gb')['tpch_tune'])
        #TPC_H.generate_update_files(2, 2)
        #tpch = TPC_H(conn)
        #tpch.rf2(1)