import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from background import sympy_strip
import warnings as Warn



class SimulationBuck:
    def __init__(self, Vin, D, L, R_load, fs, R_inductor=0, C_out=0, RDSon=0, Vdiode=0, Eon = 0, Eoff=0, Qr=0):
        # input parameters:
        self.Vin = sympy_strip(Vin)
        self.D = sympy_strip(D)  # set duty cycle
        self.L = sympy_strip(L)
        self.fs = sympy_strip(fs)
        self.R_inductor = sympy_strip(R_inductor)  # winding resistance
        self.C = sympy_strip(C_out)
        self.R = sympy_strip(R_load)
        self.RDSon = RDSon
        self.Vd = Vdiode
        self.Eon = Eon  # switching on energy loss per switching period
        self.Eoff = Eoff  # switching off energy loss per switching period
        self.Qr = Qr  # Diode reverse recovery charge
        
        # internal variables:
        self.T = 1/self.fs
        samples_per_period = 200  # Precision: samples per time period T
        length = 100  # total_time = length * T
        n_samples = int(samples_per_period*length)
        self.t = np.linspace(0, length*self.T, n_samples)
        
        self.fres = 1/(2*np.pi*np.sqrt(self.L*self.C))
        print(f'{self.fres=}')
        print(f'{self.L=}')
        print(f'{self.C=}')
        
        self.sawtooth = self.t/self.T-np.floor(self.t/self.T)
        
        self.iL = np.zeros(self.t.size)  # inductor current
        self.vC = np.zeros(self.t.size)  # capacitor current
        
        self.Pc = np.zeros(self.t.size)  # conduction power losses
        self.Ps = np.zeros(self.t.size)  # switching power losses
        self.id = np.zeros(self.t.size)  # diode current
        self.ifet = np.zeros(self.t.size)  # MOSFET current
        
        # Initial conditions:
        self.iL0 = 0
        self.vC0 = 0
        
        # Space State x=Ax+Bu:
        #self.A = np.array([[-self.R_inductor/self.L, -1/self.L],
                           #[1/self.C, -1/(self.R*self.C)]])
        #self.x = np.array([[self.iL],
                           #[self.vC]])
        
        #self.B = np.array([[1/self.L],
                           #[0]])
        #self.u = np.array([[self.Vin]])
        
        pass
    
    
    def rms(self, signal: np.ndarray):
        """ Returns the Root Mean Squared of a signal """
        if isinstance(signal, (list, np.ndarray)):
            return np.sqrt(np.mean(signal**2))
    
    def compute(self):
        self.iL[0] = self.iL0
        self.vC[0] = self.vC0
        dt = self.t[1]-self.t[0]
        print(f'Time step dt= {dt} seconds')
        diLdt = 0  # derivative of inductor's current
        dvCdt = 0  # derivative of capacitor's voltage
        C = self.C
        
        Vd = self.Vd
        R_inductor = self.R_inductor
        
        # loop
        for i in range(1, len(self.t)):
            
            ########### Preprocessing of non-linearities #########################
            if self.D > self.sawtooth[i]:
                # Switch conducting => Vin coupled
                Vin = self.Vin                
                
                RDSon = self.RDSon
                self.ifet[i-1] = self.iL[i-1]
                
                self.Pc[i] += (RDSon + R_inductor)*self.ifet[i-1]**2 # conduction loss
                
                if self.D < self.sawtooth[i-1]:
                    # Switch turns on this iteration
                    self.Ps[i] += self.Eon/dt
                    
                    # diode blocks on this iteration
                    self.Ps[i] += self.Qr*Vin/dt
                pass
            else:
                # Switch blocking => Vin discoupled
                if self.vC[i-1] >= Vd:
                    Vin = 0 - Vd  # forward biased
                    self.id[i-1] = self.iL[i-1]
                    
                    self.Pc[i] += self.id[i-1]*Vd  # conduction loss
                else:
                    Vin = 0
                    self.iL[i-1] = 0  # diode blocked
                    
                if self.D > self.sawtooth[i-1]:
                    # Switch turns off on this iteration
                    self.Ps[i] = self.Eoff/dt
                
            if self.iL[i-1] < 0 and self.D < self.sawtooth[i-1]:
                # Switch open (only diode closed). This prevents reverse flow
                self.iL[i-1] = 0
                    
                #self.switch[i] = 0
                RDSon = 0
                pass
            
            if False:
                ####### EXPLICIT EULER METHOD (UNSTABLE if C < 1uF) #################  
                diLdt = (Vin - self.vC[i-1] - self.iL[i-1]*(R_inductor+RDSon)) / self.L
                dvCdt = (self.iL[i-1] - self.vC[i-1]/self.R) / self.C

                #calculating the states
                self.iL[i] = np.clip(self.iL[i-1] + diLdt * dt, 0, 1e4)
                self.vC[i] = self.vC[i-1] + dvCdt * dt
                #######################
            
            else:
                ####### IMPLICIT MIDPOINT METHOD (STABLE, HARDCODED) ###############
                R = self.R
                C = self.C
                L = self.L
                vCn = self.vC[i-1]
                iLn = self.iL[i-1]
                rL = self.R_inductor
                fres = 1/(2*np.pi*np.sqrt(L*C))
                
                self.iL[i] = (iLn + dt/L * (Vin-vCn-rL*iLn) )/(1 + dt*rL/(2*L) )
    #             iLn = self.iL[i]

                if self.fres < self.fs: # Something like Nyquist condition
                    #self.vC[i] = (vCn + dt/C * (iLn - vCn/(2*R)) ) / (1 + dt/(2*R*C) ) # *C/C
                    self.vC[i] = (C*vCn + dt * (iLn - vCn/(2*R)) ) / (C*1 + dt/(2*R) )

                else:
                    if i == 1:
                        Warn.warn('Resonant frequency too high to be sampled. Capacitor disconsidered')
                        print('...')
                    self.vC[i] = iLn * R # This removes the capacitor and improves precision
                ####################################################################
            # TEST Runge-Kutta RK4
            if False:
                #print('Runge-Kutta RK4 solver.')
                def deriv_iL(iL):
                    return (Vin - vCn - rL*iL) / L
                
                
                # Compute iL:
                ik1 = deriv_iL(iLn)
                ik2 = deriv_iL(iLn + dt*ik1/2)
                ik3 = deriv_iL(iLn + dt*ik2/2)
                ik4 = deriv_iL(iLn + dt*ik3)
                self.iL[i] = iLn + dt/6*(ik1 + 2*ik2 + 2*ik3 + ik4)
                
                iLn = self.iL[i]
                def deriv_vC(vC):
                    return (iLn - vC/R) / C
                
                # COmpute vC:
                vk1 = deriv_vC(vCn)
                vk2 = deriv_vC(vCn + dt*vk1/2)
                vk3 = deriv_vC(vCn + dt*vk2/2)
                vk4 = deriv_vC(vCn + dt*vk3)
                self.vC[i] = vCn + dt/6*(vk1 + 2*vk2 + 2*vk3 + vk4)
            ####################################################################
            
            
            
            pass
        print(f'Resonant frequency: {self.fres} Hz\nSamping frequency: {1/dt}Hz')
        display(pd.DataFrame({'Parameter': ['Resonant frequency', 'Samping frequency'],
                              'Value': [self.fres, 1/dt]}, index=['','']))
        pass
    
    def plot_states(self):
        """ Plots iL and vC """
        
        ax1 = plt.subplot(2,1,1)
        plt.title('Numerical Simulation of Buck converter')
        plt.plot(self.t, self.iL)
        plt.grid(True)
        plt.ylabel('iL [A]')
        
        ax2 = plt.subplot(2,1,2, sharex=ax1)
        plt.plot(self.t, self.vC)
        plt.grid(True)
        plt.ylabel('vC [V]')
        plt.xlabel('t [s]')
        
        return ax1, ax2
    
    def plot_power(self):
        fig = plt.figure();
        ax1 = fig.add_subplot(2,1,1);
        ax1.plot(self.t, self.Pc);
        ax1.set_ylabel('Conduction \nPower losses [W]');
        
        ax2 = fig.add_subplot(2,1,2, sharex=ax1)
        ax2.plot(self.t, self.Ps)
        ax2.set_ylabel('Switching \nPower losses [W]')
        ax2.set_xlabel('time (s)');
        ax2.ticklabel_format(style='sci')
        return ax1, ax2
    
    def run(self):
        self.compute()
        self.plot()
    
    @staticmethod
    def print_table(columns: list):
        """ columns = [[par0, val0], [par1, var1], ...] """
        header = [['Parameter', 'Value']]
        df = pd.DataFrame(data=header.extend(columns))#, index=(" ,"*len(columns)).split(","))
        print(header)
        display(df)
        
    def print_results(self):
        length = len(self.iL)  # length of simulated data
        half = int(length/2+1)  # half way through the simulated data
#         print('i_diode average: ', simulation.id[half:].mean())
#         print('MosFET RMS Current (i_FET):', rms(simulation.ifet[half:]))
#         print('i_FET RMS:', rms(simulation.ifet[half:]))
#         Pc_avg = np.mean(simulation.Pc[half:])
#         Ps_avg = np.mean(simulation.Ps[half:])
#         print('Mean Conduction Power loss: ', Pc_avg, 'W')
#         print('Mean Switching Power loss: ', Ps_avg, 'W')

#         Ploss_avg = Pc_avg + Ps_avg
#         Po_avg = (simulation.vC**2/Rload)[half:].mean()
#         print("Average total losses: {:.5}".format(Ploss_avg), 'W')
#         print("Output Power: {:.5}".format( Po_avg), 'W')
#         print("Efficiency: {:.2%}".format((Po_avg-Ploss_avg)/Po_avg))
        self.print_table([
            ['i_diode average', self.id[half:].mean()]
        ])
        
    pass


if __name__ == '__main__':
    k=1
    Vin = 480       # input voltage
    D = 0.1       # duty cycle
    fs = 40e3*k           # switching frequency
    R_inductor = 3/1000  # Resistance of the inductor's coil. Only needed for analysis
    C_out = (10e-6)/k        # Output capacitance. Needed for the analysis of filter
    R_load = 2.3
    L = 0.00108/k
    Tres=2*np.pi*np.sqrt(C_out*L)
    
    simulation = SimulationBuck(Vin, D, L, R_load, fs, R_inductor, C_out, Vdiode = 0.7, RDSon=1e-2)
    # simulation = SimulationBuck(480, 0.1, .01, 100, 40e3,1/100,1e-6)
    simulation.compute()
    simulation.plot()
    
    plt.plot(simulation.sawtooth)
    plt.show()
    print(Tres, simulation.T,simulation.T/Tres)
    pass
