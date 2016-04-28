#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Greeter.

Usage:
  launcher.py <path>     

Options:
  -h --help     Show this screen.
""" 
from docopt import docopt
arguments = docopt(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.transforms as mtransforms
import matplotlib.text as mtext
import glob

l=glob.glob('%s/*'%arguments['<path>'])
l_fail=[]
l_suc=[]
for n in l:
    o=eval(open(n).read())
    h=o['history']
    t=o['target']
    r=np.asarray(h[-1])-np.asarray(t)
    r_2=np.sum(r**2)
    maxspeed=0.02*25*0.2 #*4
    r=np.asarray(h[0])-np.asarray(t)
    radius=5
    initial_distance=(np.sum(r**2)**0.5 ) -radius#/ maxspeed
    h_l=len(h)
#    h_l=2000 if h_l  > 2000 else h_l
#    initial_distance =2000 if initial_distance>2000 else initial_distance
    h_l*=maxspeed
    if r_2>25:
        l_fail.append((h_l,initial_distance))
    else:
        l_suc.append((h_l,initial_distance))
l_fail.sort()
l_suc.sort()

fig, ax = plt.subplots()
l=l_fail+l_suc
l=np.asarray(l,dtype=float)
l=np.clip(l,0,200)
#@l1=np.asarray([i[1] for i in l],dtype=float)
plt.plot(l[:,0])
plt.plot(l[:,1])
plt.show()
