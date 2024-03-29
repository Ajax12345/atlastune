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
        self.cur = self.conn.cursor()
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
    with MySQL() as conn:
        conn.use_db("grata_data")
        print(conn.status())