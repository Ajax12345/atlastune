import mysql.connector, typing

#https://dev.mysql.com/doc/connector-python/en/connector-python-example-connecting.html

conn = mysql.connector.connect(
  host ="localhost",
  user ="root",
  passwd ="Gobronxbombers2",
)

cur = conn.cursor()
cur.execute("use grata_data")
cur.execute("""select count(*) from capterra_data""")
print([*cur])


conn.close()


class MySQL:
    def __init__(self, host = "localhost", user = "root", passwd ="Gobronxbombers2", database="grata_data") -> None:
        self.host, self.user = host, user
        self.passwd, self.database = passwd, database


if __name__ == '__main__':
    pass