select c2.c_id, sum(case when c2.c_middle != 'Yb' then 1 else 0 end) from (
    select c.c_id, c.c_last, c.c_middle from customer1 c
    where c.c_balance >= (select avg(c1.c_balance) from customer1 c1 where c.c_first = c1.c_first)
        and c.c_credit = 'BC'
        and c.c_zip > 331277273
        and c.c_delivery_cnt > 15
) c2
where substring(c2.c_last, 1, 4) = 'ESEE'
group by c.c_id;