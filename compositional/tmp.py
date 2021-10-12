import numpy as np
import skbio.stats.composition as sk


x=np.array([0.1, 0.3, 0.4, 0.2])
sk.ilr(x,basis=np.exp(1))





p = np.array([[0,0,1],[1,0,0],[0,1,0]])
xx = np.matmul(p,x) 


mat = x
mat = sk.closure(mat)
x=mat
basis = sk.clr_inv(_gram_schmidt_basis(mat.shape[-1]))

D = len(x)

for i in range(1,D-1):
    for j in range(i+1,D):
        xj=x[i:D]
        sqrt((D-i)/(D-i+1))*np.log(x[i]/)

x= np.array([0.1, 0.3, 0.4, 0.2])

a= np.sqrt(3/4)
b= np.log(x[0]/(np.product(np.array([0.3,0.4,0.2]))**(1/3)))
a*b

a= np.sqrt(2/3)
b= np.log(x[1]/(np.product(np.array([0.4,0.2]))**(1/2)))
a*b

a= np.sqrt(1/2)
b= np.log(x[2]/(np.product(np.array([0.2]))**(1)))
a*b






np.log(x[0]/np.product(x[1:D])**(1/3))