import mysql_conn, typing, os
import subprocess, concurrent.futures
import re, time

class TPC_H:
    LOCATION = 'tpc/tpch/tpch_official/dbgen'
    CREATION_SCRIPTS = 'tpc/tpch/tbl_creation_scripts'

    def __init__(self, conn: mysql_conn.MySQL) -> None:
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


if __name__ == '__main__':
    with mysql_conn.MySQL(database = "tpch_tune") as conn:
        #TPC_H.generate_update_files(2, 2)
        #print(TPC_H.update_files())
        TPC_H.remove_tbl_files()
        TPC_H.delete_update_files()

        [
            1,
            
        ]