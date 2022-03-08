import sympy as sp
import numpy as np
from sympy.physics.units import ohm, volt
from sympy.physics.units.quantities import Quantity
from sympy.core.mul import Mul
from sympy.core.numbers import Zero, One, Integer, Float, Rational, Add
from matplotlib import pyplot as plt
from typing import Union
from warnings import warn


SYMPY_TYPES = (Mul, One, Zero)

def rms(x: Union[list, tuple, np.ndarray]):
        if not isinstance(x, (list, tuple, np.ndarray)):
            warn('Input x must be of array kind list, tuple or numpy.ndarray')
        return np.sqrt(
            np.mean(
                np.square(x)
            )
        )


def sympy_strip(*a: Union['SYMPY_TYPES', list, tuple]) -> float:
    """ 
    Turns sympy objects to ordinary floats so they work inside other modules (e.g. numpy
    a: Union
    """
    
    if isinstance(a[0], (list, tuple)) and len(a[0]) > 0:
        ret = list(a[0])
    else:
        ret = list(a)
    for i in range(len(ret)):
        if isinstance(a[i], Mul):
            ret[i] = float(a[i].args[0].evalf())
        elif isinstance(a[i], (Rational, One, Zero)):
            ret[i] = float(a[i].evalf())
            
        elif isinstance(a[i], (Integer, Rational)):
            
            ret[i] = float(a[i].evalf())
        elif isinstance(a[i], Quantity):
            ret[i] = 1.0
        elif isinstance(a[i], Float):
            ret[i] = float(a[i])
        elif isinstance(a[i], Add):
            r=[]
            for arg in a[i].args:
                r.append(arg.args[0])
            print('Warning: summing different units')
            ret[i] = sum(r)
        else:
            #ret[i] = float(a[i])
            #ret[i] = float(a[i].args[0].evalf()) # BUG
            ret[i] = float(a[i])
        if False: print(f'Striped {a[i]} (type {type(a[i])}) to {ret[i]} (type {type(ret[i])})')
        #ret[i] = float(ret[i])
    if len(ret) < 2:
        return ret[0]
    else:
        return tuple(ret)

class OperationModePlot:
    def __init__(self, Vo, Vi, Io, L, fs):
        self.Vo = sympy_strip(Vo)
        self.Vi = sympy_strip(Vi)
        self.L = sympy_strip(L)
        self.fs = sympy_strip(fs)
        self.Io = sympy_strip(Io)
        
        self.normV = np.linspace(0,1,101)
        self.normI = np.linspace(1e-9,0.25,101)
        self.Do = np.arange(0.1,1,0.2)# set of discrete duty cycles to plot
        #print(f'{self.Do.size=}, {self.normI.size=}')
        self.isocurves = np.zeros((self.Do.size, self.normI.size))
        
        self.D = np.linspace(0,1,101)# continuous duty cycles to plot the limit
        
    def plot_op_border(self):
        D=self.D
        plt.plot((1-D)*D/2,self.normV,'--')
        # NOTE wikipedia says (1-D)*D/2, while Ivo Barbis' 'Conversores cc-cc basico nao isolados' says only (1-D)*D
        # Ivo Barbi seems to be wrong in this one
    
    def plot_isolines(self):
        
        for n in range(self.Do.size):
            D = self.Do[n]
            for m in range(self.normI.size):
                I = self.normI[m]
                #TODO check if it is continuos or discontinuos
                if (1-D)*D/(2*I) < 1: # continuous
                    self.isocurves[n][m] = D
                    pass
                else: #discontinuos
                    self.isocurves[n][m] = D**2/(D**2+2*I)
                    pass
            plt.plot(self.normI, self.isocurves[n])
            plt.text(x=.2, y=D+.01,s=f'D={int(100*D)}%')
            
        Vo = self.Vo
        Vi = self.Vi
        L = self.L
        fs = self.fs
        Io = self.Io
        #Io = 11 # ?????
        #Io = ((Vi-Vo)*(D**2)*(1+(Vi-Vo)/Vo))/(2*L*fs)
        #Io = self.Io/(1000/Vo)
        Inorm = L*Io*fs/Vi#*2
        print(f'{Io=}')
        
        # plot operation point marker
        if Inorm > 0.25:
            # If outside of plotting range, plot it at the edge as a right facing triangle marker
            plt.plot(0.25, Vo/Vi, '>', markersize=10, c='b')
        else:
            plt.plot(Inorm, Vo/Vi, '.', markersize=20, c='b')
        print(f'{L=}'); print(f'{Io=}'); print(f'{fs=}'); print(f'{Vi=}'); print(f'{Inorm=}')
        plt.xlabel('Normalized output current |Io|')
        plt.ylabel('Normalized voltages Vout / Vin')
        plt.title('Operation mode diagram\nBuck Converter')
        pass
    
    def show(self):
        plt.show()
        
    def draw(self):
        self.plot_op_border()
        self.plot_isolines()
        self.show()
        
    
    
if __name__ == '__main__':
    if True:
        R=666*ohm
        V=220*volt
        x = [*sympy_strip(1+R/(2*R), V,V/R, R*R+V, R/R)]
        print(x)
    
    if False:
        opmode = OperationModePlot(Vo=48,Vi=480, Io=15,L=0.00108,fs=40000)
        opmode.draw()
