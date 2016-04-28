from pymongo import MongoClient
from bson import Binary
client = MongoClient()
db = client.test
dqndb=db.dqn
import pickle

def save(y):
    # serialize array y
    y=Binary( pickle.dumps( y, protocol=2))
    return dqndb.insert({'content':y})
def load(i):
    y=dqndb.find_one(i)['content']
    # deserialize array y
    return pickle.loads( y )
def remove(i):
    return dqndb.remove(i)


if '__main__'==__name__:
    print('test')
    import numpy as np
    a=np.zeros(9)+1
    i=save(a)
    print(i)
    a=load(i)
    print(a)
    r=remove(i)
    print(r)
    try:
        a=load(i)
    except Exception as e:
        print(e)


