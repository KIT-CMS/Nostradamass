import math
import numpy as np
import sys

def create_FourMomentum(in_string):
    result_vector = FourMomentum(0,0,0,0)

    vectors = in_string.split(";")
    for v in vectors:
        if len(v) ==0:
            continue
        x = v.split(",")
        result_vector += FourMomentum(float(x[0]), float(x[1]), float(x[2]), float(x[3]))

    return result_vector

METRIC = [1,-1,-1,-1]
def contract_tuples(lhs,rhs,metric = None):
    return sum(m*l*r for m,l,r in zip(metric if metric else [1]*len(lhs),lhs,rhs))

def contract(lhs, rhs):
    return contract_tuples(lhs.components(),rhs.components(),METRIC)

class FourVector(object):
    def __init__(self,x0,x1,x2,x3,cartesian=True,massless=False):
        if massless and cartesian:
            self._x0 = np.sqrt(np.square(x1) + np.square(x2) + np.square(x3))
            self._x1 = x1
            self._x2 = x2
            self._x3 = x3
        elif cartesian:
            self._x0 = x0
            self._x1 = x1
            self._x2 = x2
            self._x3 = x3
        elif not cartesian and not massless:
            m = x0
            pt = x1
            eta = x2
            phi = x3
            self._x1 = math.cos(phi) * pt
            self._x2 = math.sin(phi) * pt
            self._x3 = math.sinh(eta) * pt
            sqrt_sum = np.sum(np.array([self._x1, self._x2, self._x3, m])**2)
            self._x0 = np.sqrt(sqrt_sum)
        else:
            raise RuntimeError("Invalid configuration of cartesian and massless Fourvector")
    
    def __add__(lhs,rhs):
        return FourVector(*[sum(x) for x in zip(lhs.components(),rhs.components())])

    def __str__(self):
        return str(('x0: ', self.x0, ', x1: ', self.x1, ', x2: ', self.x2, ', x3: ', self.x3))

    @property
    def x0(self):
        return self._x0
    
    @property
    def x1(self):
        return self._x1
    
    @property
    def x2(self):
        return self._x2
    
    @property
    def x3(self):
        return self._x3
    
    @property
    def eta(self):
        if abs(self.perp()) < sys.float_info.epsilon:
            return float('inf') if self.x3 >=0 else float('-inf')
        return -math.log(math.tan(self.theta/2.))
    
    @property
    def theta(self):
        return math.atan2(self.perp(),self.x3)
    
    @property
    def phi(self):
        return math.atan2(self.x2,self.x1)
    
    def components(self):
        return (self.x0,self.x1,self.x2,self.x3)
    
    def s2(self):
        return contract(self,self)

    def s(self):
        return math.sqrt(contract(self,self))
    
    def perp2(self):
        transvers_comps = self.components()[1:-1]
        return contract_tuples(transvers_comps,transvers_comps)
        
    def perp(self):
        return math.sqrt(self.perp2())

    def as_list(self):
        return (self.x0, self.x1, self.x2, self.x3)

    def as_list_hcc(self):
        return (self.pt, self.eta, self.phi, np.sqrt(self.m2()) if self.m2() > 0 else 0)

    def as_numpy_array(self):
        return np.array(self.as_list())

    def as_numpy_array_hcc(self):
        return np.array(self.as_list_hcc())
        
class FourMomentum(FourVector):
    def __add__(lhs,rhs):
        return FourMomentum(*FourVector.__add__(lhs,rhs).components())
    
    e  = FourVector.x0
    
    px = FourVector.x1
    
    py = FourVector.x2
    
    pz = FourVector.x3
    
    @property
    def pt(self):
        return super(FourMomentum,self).perp()
    
    def m(self):
        return super(FourMomentum,self).s()
        
    def m2(self):
        return super(FourMomentum,self).s2()
    
    def pt2(self):
        return super(FourMomentum,self).perp2()
            
class FourPosition(FourVector):
    def __add__(lhs,rhs):
        return FourPosition(*FourVector.__add__(lhs,rhs).components())
    
    t = FourVector.x0
    
    x = FourVector.x1
    
    y = FourVector.x2
    
    z = FourVector.x3


