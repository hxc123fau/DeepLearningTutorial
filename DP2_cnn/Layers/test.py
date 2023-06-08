import numpy as np

a= np.random.uniform(low=0, high=1, size=(3,5,6))
b=np.prod(a.shape[1:])

print(len(a))

def add(s, *arg):
    # print(s)
    if  len(arg)==3:
        res=arg[2]
        print(len(a))

# add(1)