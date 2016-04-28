import sys
import numpy as np
import glob
histories=sys.argv[1:]
result=[]
for his in histories:
    l=glob.glob('%s/*'%his)
    r=0.0
    for n in l:
        o=eval(open(n).read())
        t=o['target']
        h=o['history']
        r2=np.sum((np.asarray(t)-np.asarray(h[-1]))**2)
        if r2<25:
            r+=1
    r=(r/len(l))
    print(r,his)
    result.append(r)
print('----------')
l=zip(result,histories)
l.sort()
for n in l:
    print(n)
