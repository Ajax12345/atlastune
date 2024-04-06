select * from (select s.s_i_id, sum(case when 
    s.s_quantity >= (select avg(s1.s_quantity) from stock s1 
    join warehouse w on w.w_id = s1.s_w_id
    join district d on d.d_w_id = w.w_id where d.d_city != '2HevADQC6IfvCXr6aOFt') 
    then 1 else 0 end) a
from stock s
group by s.s_i_id) k
join item i on i.i_id = k.s_i_id
join order_line ol on ol.ol_i_id = i.i_id
where ol.ol_delivery_d > '2024-03-31 17:54:14'
-- long