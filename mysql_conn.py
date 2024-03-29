import mysql.connector, typing
import contextlib

#https://dev.mysql.com/doc/connector-python/en/connector-python-example-connecting.html


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
    def status(self) -> typing.List[tuple]:
        self.cur.execute('show status')
        return [*self.cur]

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
    def get_columns(self, tbl:str) -> typing.Any:
        self.cur.execute("""
        select t.table_schema, t.table_name, t.column_name, 
            s.index_schema, s.index_name, s.seq_in_index, s.index_type 
        from information_schema.columns t
        left join information_schema.statistics s on t.table_name = s.table_name
            and t.table_schema = s.table_schema 
            and lower(s.column_name) = lower(t.column_name)
        where t.table_schema = %s and t.table_name = %s""", [self.database, tbl])
        return [*self.cur]

    @DB_EXISTS()
    def get_indices(self, tbl:str) -> typing.Any:
        self.cur.execute(f"""show index from {tbl}""")
        return [*self.cur]

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
        print(conn.get_columns('test_stuff'))
