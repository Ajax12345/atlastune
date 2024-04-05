SELECT c_discount, c_last, c_credit, w_tax 
FROM customer, warehouse 
WHERE w_id = 8 AND c_w_id = w_id AND c_d_id = 6 AND c_id > 20 and c_id < 400
order by c_discount, c_last, c_credit, w_tax;