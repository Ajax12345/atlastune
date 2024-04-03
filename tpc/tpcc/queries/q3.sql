select sum(i.i_price) from customer c
join orders o on o.o_c_id = c.c_id and o.O_CARRIER_ID = 100
join order_line ol on ol.ol_o_id = o.o_id
join item i on i.i_id = ol.ol_i_id
where c.c_state = 'MN';