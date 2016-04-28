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
"""
Show how to override basic methods so an artist can contain another
artist.  In this case, the line contains a Text instance to label it.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.transforms as mtransforms
import matplotlib.text as mtext

o=open(arguments['<path>']).read()
o=eval(o)

fig, ax = plt.subplots()
fig.set_size_inches(8, 8, forward=True)

h=o['history']
l=np.asarray(h)
plt.plot(l[:,0],l[:,1],color='b')
start_point=l[0]

wall=[[-60,-60],[-60,60],[60,60],[60,-60],[-60,-60]]
l=np.asarray(wall)
plt.plot(l[:,0],l[:,1],color='r')
plt.ylim(-60, 60)
plt.xlim(-60, 60)

obstalces=o['obstalces']
l=np.asarray(obstalces)
#plt.plot(l[:,0],l[:,1],'o',color='r')

target=o['target']
circle2=plt.Circle(target,radius=5,color='g')
ax.add_patch(circle2)

plt.plot([start_point[0],target[0]],[start_point[1],target[1]],color='g')


start=o['history'][:1]
l=np.asarray(start)
plt.plot(l[:,0],l[:,1],'o',color='y')

p=[(5,5),(0,5),(10,0)]
p=np.asarray(p)
xy = np.random.normal(size=2)
for p,r in (zip(o['obstalces'],o['obstalce_radius'])+[((-25,-25),12.5)]):
    circle2=plt.Circle(p,radius=r,color='r')
    ax.add_patch(circle2)
rect=plt.Rectangle((5,5),40,40,color='r')
ax.add_patch(rect)

plt.show()
