select d.d_name, w.w_name from district1 d
join warehouse1 w on w.w_id = d.d_w_id
where d.d_tax != (select sum(d1.d_tax)/count(*) from district1 d1)
    and w.w_city in (select c.c_city from customer1 c where c.c_balance > 100)
order by d.d_name, w.w_name