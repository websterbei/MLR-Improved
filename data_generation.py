import random as rd

for i in range(100):
    y = rd.uniform(0,10)
    x = rd.uniform(0,10)
    if x>=y and x+y<=10:
        print(x,y)
    else:
        print(x,y)
