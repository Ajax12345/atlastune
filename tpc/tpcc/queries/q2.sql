select w.w_id, w.w_tax from warehouse w where w.w_tax = (select max(w1.w_tax) from warehouse w1)
order by w.w_id asc, w.w_tax desc;