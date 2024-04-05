select d.d_name, w.w_name from district d
join warehouse w on w.w_id = d.d_w_id
where d.d_tax != (select sum(d1.d_tax)/count(*) from district d1)
    and w.w_city in (select c.c_city from customer c where c.c_balance > 100)
order by d.d_name, w.w_name