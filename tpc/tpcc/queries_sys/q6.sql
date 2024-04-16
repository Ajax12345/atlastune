select s.s_i_id, count(*) from stock1 s
where s.S_DIST_02 not in ('DVbGf0IVAunoBjjtIXKiCGQO', 'gqQfyOkTn2TsXkzIV8mDMyBF')
    or s.S_DIST_08 != 'Adpf32BPZXc14M7aiZE85mxQ'
group by s.s_i_id;