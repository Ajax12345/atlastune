select sum(i.i_price) from customer1 c
join orders1 o on o.o_c_id = c.c_id and o.O_CARRIER_ID = 100
join order_line1 ol on ol.ol_o_id = o.o_id
join item1 i on i.i_id = ol.ol_i_id
where c.c_state = 'MN';