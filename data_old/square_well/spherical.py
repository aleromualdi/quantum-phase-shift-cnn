import matplotlib.pyplot as plt
from  scipy.integrate import odeint
import scipy.integrate as integrate
import numpy as np
import scipy.special as sp
import scipy
import sys
#import importlib
import time
import math
import pylab as P
# from numpy import sqrt, sin, cos, pi
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)
plt.rcParams.update({'font.size': 18})

#importlib.import_module('scipy')
# Calogero's units 2m=h_bar= 1
print(scipy.__version__)
prova =sp.spherical_jn(0, 4., derivative=False)
print(prova)

k = 0.1   #1e-5
V0 = 0.25  #1.  #strengh of potential, V is negative square well
sizeSW = 1.
#p0 = k + 2. * V0
#p = np.sqrt(p0)
noData = 200
#######
initSeed = 21
np.random.seed(initSeed)
lMax = 3    #l=0,1,2


def checkEqual3(lst):
    return lst[1:] == lst[:-1]


print('size SW', sizeSW)

def ps0(r, k, V0):
    p0 = k**2 + V0    # 2. * V0 in atomic units
    p = np.sqrt(p0)
    x = k * r
    y = p * r
    n1 = k * np.cos(x) * np.sin(y)
    n2 = p * np.sin(x) * np.cos(y)
    nTop = n1 - n2
    n3 = k * np.sin(x) * np.sin(y)
    n4 = p * np.cos(x) * np.cos(y)
    nBot = n3 + n4
    frac = nTop/nBot
    return np.arctan(frac)


def ps0l(r, k, V0, l):
    p0 = k**2 + V0
    alfa = np.sqrt(p0)
    x = k * r
    y = alfa * r
    n1 = k * sp.spherical_jn(l, x, derivative=True) * sp.spherical_jn(l, y, derivative=False)
    n2 = alfa * sp.spherical_jn(l, x, derivative=False) * sp.spherical_jn(l, y, derivative=True)
    nTop = n1 - n2
    n3 = k * sp.spherical_yn(l, x, derivative=True) * sp.spherical_jn(l, y, derivative=False)
    n4 = alfa * sp.spherical_yn(l, x, derivative=False) * sp.spherical_jn(l, y, derivative=True)
    nBot = n3 - n4
    tanDelta = nTop/nBot
    return np.arctan(tanDelta)

sizeSW2 = 2. # np.pi/6.
V02 = 9
k2 = 0.5

print('phase shift s', ps0(sizeSW2, k2, V02))
print(' approx phase shift', np.arctan(k2))
print('phase shift s bis', ps0l(sizeSW2, k2, V02, 0))

sys.exit()

aList = []
psList = []

for counter in range(0, noData):
    aValue = np.random.uniform(0.4, 2.)
    phaseS0 = ps0(aValue, k, V0)
    ps1 = - k * aValue
    k0 = k ** 2 + V0
    k1 = np.sqrt(k0)
    ratioK = k/k1
    argKa = ratioK * aValue
    factor = np.tan(argKa)
    factor1 = ratioK * factor
    ps2 = np.arctan(factor1)
    phaseS0bis = ps1 + ps2
    tan1 = np.tan(k1 * aValue)
    tan2 = np.tan(k * aValue)
    tanNum = ratioK * tan1 - tan2
    tanDen = 1. + ratioK * tan1 * tan2
    tanFractio = tanNum/tanDen
    phaseS0ter = np.arctan(tanFractio)
    #print(k, aValue, phaseS0, phaseS0, phaseS0) # phaseS0ter
    l0 = 0
    psQuater0 = ps0l(aValue, k, V0, l0)
    l1 = 1
    psQuater1 = ps0l(aValue, k, V0, l1)
    l2 = 2
    psQuater2 = ps0l(aValue, k, V0, l2)
    aList.append(aValue)
    psList.append(psQuater0)
    print(k, aValue, psQuater0, psQuater1, psQuater2)
    f = open('sw' + 'k=0.1Bis' + '.txt', "a")
    print(k, aValue, psQuater0, psQuater1, psQuater2, file=f)
    f.close()
    boolean = checkEqual3(aList)
    if boolean == True:
        print('hay duplicates here')
        print(set([x for x in aList if aList.count(x) > 1]))

aArray = np.array(aList)
psArray = np.array(psList)

plt.xlabel('$a$')
plt.ylabel('$\delta_0$')

plt.plot(aArray, psArray,'.g', linewidth=3)

plt.show()



kAr = np.arange(0.1,10,0.1)
#print(kAr.shape)

V01 = 9.
aValue = 2. # np.pi/6.
kAr1 = aValue * kAr
listPSnew = []

for counterK in kAr1:
    l00 = 0
    psNew = ps0l(aValue, counterK, V01, l00)
    print(counterK, psNew)
    if psNew < 0:
        psNew = psNew + np.pi
        print('ops', psNew)
    listPSnew.append(psNew)


psNew = np.array(listPSnew)
print(psNew[0])

plt.xlabel('$ka$')
plt.ylabel('$\delta_0$')



plt.plot(kAr1, psNew,'.g', linewidth=3,  label='$a =2$')
plt.legend(loc='upper right', shadow=True, frameon=False, fontsize='medium')


plt.show()




#l = [1,2,3,4,4,5,5,6,1]
#print(set([x for x in l if l.count(x) > 1]))




sys.exit()


#print 'out', sp.spherical_jn(0, 4., derivative=False)

#sp.spherical_jn(l, x, derivative=False)
#sp.spherical_yn(n, z, derivative=False)

