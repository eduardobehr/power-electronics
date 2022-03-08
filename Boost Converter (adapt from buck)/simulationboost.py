import numpy as np
from matplotlib import pyplot as plt




class SimulationBuck:
    def __init__(self, Vin, Vout, Pout, fs, dIL_max, R_inductor=0, C_out=0):
        self.Vin = Vin
        self.Vout = Vout 
        self.Pout = Pout
        self.fs = fs
        self.dIL_max = dIL_max
        self.R_inductor = R_inductor
        self.C_out = C_out
        
        self.T = 1/fs
        self.t = np.linspace(0,int(self.T*50),5001)
        self.D = Vout/Vin
        
        pass
    
    def compute(self):
        
        # loop
        for i in range(len(self.t)):
            
        pass
    
    def plot(self):
        
        pass
    pass


if __name__ == '__main__':
    
    Vin = 480           # input voltage
    Vout = 48           # output voltage
    Pout = 1000         # output power
    fs = 40e3           # switching frequency
    dIL_max = 1         # maximum Inductor current ripple
    R_inductor = 3/100  # Resistance of the inductor's coil. Only needed for analysis
    C_out = 1e-6        # Output capacitance. Needed for the analysis of filter
    simulation = SimulationBuck(Vin, Vout, Pout, fs, dIL_max, R_inductor, C_out)
    pass
