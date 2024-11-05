CREATE TABLE IF NOT EXISTS orders (
  `o_orderkey`       INT,
  `o_custkey`        INT,
  `o_orderstatus`    CHAR(1),
  `o_totalprice`     DECIMAL(15,2),
  `o_orderdate`      DATE,
  `o_orderpriority`  CHAR(15),
  `o_clerk`          CHAR(15),
  `o_shippriority`   INT,
  `o_comment`        VARCHAR(79),
  `o_dummy`          VARCHAR(10),
  PRIMARY KEY (`o_orderkey`));