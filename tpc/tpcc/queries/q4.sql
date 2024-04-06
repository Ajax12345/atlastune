select w.w_id, count(*) from warehouse w
join orders o on w.w_id = o.o_w_id 
join (select c.c_id, c.c_w_id from customer c
    join orders o1 on o1.o_c_id = c.c_id 
    join order_line ol on ol.ol_o_id = o1.o_id and ol.ol_quantity > 10) k 
    on k.c_w_id = w.w_id
where o.O_ENTRY_D >= '2024-03-31 17:53:48'
group by w.w_id
order by w.w_id;
-- long