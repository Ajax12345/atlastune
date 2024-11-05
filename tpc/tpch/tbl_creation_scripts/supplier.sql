CREATE TABLE IF NOT EXISTS supplier (
  `s_suppkey`     INT,
  `s_name`        CHAR(25),
  `s_address`     VARCHAR(40),
  `s_nationkey`   INT,
  `s_phone`       CHAR(15),
  `s_acctbal`     DECIMAL(15,2),
  `s_comment`     VARCHAR(101),
  `s_dummy` varchar(10),
  PRIMARY KEY (`s_suppkey`));