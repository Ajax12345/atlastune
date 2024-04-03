select c.c_id from customer c
join (select w.w_id, w.w_name, w.w_state, i.i_name, w.w_zip, min(s.s_quantity) m1 from warehouse w
join stock s on s.s_w_id = w.w_id
join item i on i.i_id = s.s_i_id
where s.s_ytd != 20
group by w.w_id, w.w_name, w.w_state, i.i_name, w.w_zip) ws 
    on ws.w_state = c.c_state and ws.w_zip = c.c_zip;
