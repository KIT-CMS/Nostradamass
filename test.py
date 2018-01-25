
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
bla = []
for a in range(1000):
	bla.append(np.random.normal(0,1))
#    bla.append(np.random.rand())
#    a = 5
#    while(abs(a) > 2.7):
#        a = np.random.normal(0,1.5)
#    bla.append(a)
    #bla.append( np.random.normal(0,2))

bla = np.array(bla)
pts = plt.figure()
n, bins, patches = plt.hist(bla, 150, normed=1, facecolor='red', alpha=0.75)
plt.savefig("exponential.pdf")
