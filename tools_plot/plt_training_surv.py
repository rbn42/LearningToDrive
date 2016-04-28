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

file=open(arguments['<path>'])
l=[]
for s in file:
    if 'True' in s:
        l.append(1)
    elif 'False' in s:
        l.append(0)
lt=np.asarray(l)
lf=1-lt
n=50.0
lt=np.convolve(lt,np.ones(n),'valid')/n
plt.plot(lt)
plt.show()
