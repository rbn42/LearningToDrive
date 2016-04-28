#!/usr/bin/python
import sys
import os.path
import re
root =sys.argv[1]
for n in os.listdir(root):
    if 'scale' in n:
        continue
    for scale in [2,3,4,5,6,7,8,9,10]:#,11,12,15,20]:
        p=os.path.join(root,n)
        f=open(p)
        n_out='scale%s_'%scale+n
        p_out=os.path.join(root,n_out)
        f_out=open(p_out,'w')
        while True:
            s=f.readline()
            if len(s)<1:
                break
            f_out.write(s)
            if '<Vertex>' not in s:
                continue
            s=f.readline()
            l1=[i.strip() for i in s.split()]
            l2=[str(float(i)*scale) for i in l1]
            assert len(l1)==len(l2)==3

            s='      %s %s %s\r\r\n'%tuple(l2)
            f_out.write(s)
