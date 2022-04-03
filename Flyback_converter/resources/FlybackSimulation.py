from fileinput import filename
import subprocess as sp
import pandas as pd
from sys import argv
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class Snubber:
    """ Parallel RC with Diode """
    def __init__(self, C=1e-9, R=100) -> None:
        self.C = C
        self.R = R

class FlybackSimulation:

    def __init__(self, filename_circuit="flyback.cir", filename_csv = "output.csv") -> None:
        self.filename_circuit = filename_circuit
        self.filename_csv = filename_csv
        self.netlist_generated = False
        self.df: pd.DataFrame = None
        self.waveforms = ["t", "Vds", "Vout", "Ip", "Is", "gate", "Im"]
        pass

    def _fix_ngspice_csv(self, filename):
        """
        Ngspice generates a weird waveforms output
        Line are formatted as 
            ' 0.000000000e+0  1.000000000e-1 '
            ' 0.000000000e+0 -1.000000000e-1 '

        The problem is that the leading and trailing spaces forbid pandas to read the csv with sep=" "
        Also, the vectors separation is either "  " or " -", which makes parsing even harder
        This methods attempts to get rid of these issues and reformats the lines as:
            '0.000000000e+0;1.000000000e-1'
            '0.000000000e+0;-1.000000000e-1'
        """
        lines=[]
        corrected = []

        # get messy output
        with open(filename, "r") as file:
            lines = file.readlines()
            for line in lines:
                corrected.append(
                    line.replace(" -", ";-")
                    .replace("  ", ";")
                    .replace(" ", "")
                )
        
        # overwrite as a proper ";" separated file
        with open(filename, "w") as new_file:
            new_file.writelines(corrected)
        
        return corrected


    def generate_flyback_netlist(self, 
    _Vin=311,
    _Vout=12,
    _D=0.1,
    _N=10,
    _Lm=4e-6, 
    _Rl=1, 
    _fs=160e3, 
    _Co=100e-6,
    _RDSon=1e-3,
    _transformer_coupling=0.995,
    _endtime=300e-6,
    _steps_per_switch_period=500,
    _snubber: Snubber = Snubber()
    ):
        """
        Generates a Spice netlist from a template with a simple string substitution
        """
        

        _Ts = 1/_fs
        netlist=f"""
        .title Flyback converter
        *.param _N                           = 10; N1/N2 winding ratio
        *.param _D                           = {_Vout}/({_Vin}/{_N})
        .param _Lp                          = {_Lm}*{_N}
        .param _Ls                          = {_Lm}/{_N}

        * Primary side
        Vin         nVin        gnd         {_Vin}
        .ic v(nVin) = {_Vin}
        Rin         nVin        nP          1m
        Cin         nP          gnd         1nF         ic={_Vin}
        Lstray      nP          nStray      {0};{_Lm*_N/_transformer_coupling*(1-_transformer_coupling/_N)}
        bLp         nStray      drain       v={_N}*v(sgnd, ndiod)
        Lm          nStray      drain       {_Lm}
        S1          drain       gnd         gate            gnd         switchModel  OFF
        .model      switchModel sw          vt=0.5          ron={_RDSon}      roff=100Meg
        Cds         drain       gnd         700pF; Drain to source capacitance

        *K1          Lp          Ls          1

        * Secondary side
        bLs          sgnd        ndiod      v=v(drain, nP)/{_N}
        D1          ndiod       nOut        DMOD
        .model DMOD D ( bv=500 is=1e-13 n=1.05)
        Co          nOut        sgnd        {_Co}
        Rl          nOut        sgnd        {_Rl}

        * PWM
        vModulant   signal      gnd         {_D}
        BsawtoothP  sawtoothP   gnd         v=0.5+(time*{_fs}-floor(0.5+time*{_fs}))
        Bpwm        gate        gnd         v=v(signal)>=v(sawtoothP) && v(signal)>=0 ? 1 : 0;v(signal)<=v(sawtoothN) && v(signal)<=0 ? -1 : 0


        * Snubber
        Dsnub       drain       snub        DMOD
        Csnb        nP          snub        {_snubber.C}
        Rsnb        nP          snub        {_snubber.R}

        * Misc
        Rgnd        gnd         sgnd        1Meg


        .tran {(1/_fs)/_steps_per_switch_period} {_endtime} uic

        .control
            run
            *plot v(nOut, sgnd)
            *plot Lp#branch Ls#branch    xlimit {_endtime-4*_Ts} {_endtime} 
            *plot drain                  xlimit {_endtime-4*_Ts} {_endtime} 
            set wr_singlescale
            wrdata {self.filename_csv} drain v(nOut, sgnd) bLp#branch bLs#branch gate lm#branch
            echo Simulation done
            *iplot drain
            display
            exit 
        .endc
        """

        with open(self.filename_circuit, "w") as file:
            file.write(netlist)
        self.netlist_generated = True


    def begin(self):
        """
        Invokes Ngspice from the system's shell while passing the generated netlist as first argument
        """
        if not self.netlist_generated:
            raise warnings.WarningMessage("The netlist was not generated by this instance \
            (if the circuit file is found, it might be not updated)")

        print("Calling Ngspice for generated netlist...\n")
        print(sp.getoutput("ngspice --version"))
        sp.getoutput(f"ngspice {self.filename_circuit}")
        print(f'Available waveforms: {self.waveforms}')

    def read_outputs(self):
        """
        Reads the generated csv outputs (";" separated)
        """
        self._fix_ngspice_csv(self.filename_csv)
        self.df = pd.read_csv(self.filename_csv, sep=";", names=self.waveforms)


    def plot(self, x="t", y=None, *args, **kwargs):
        if y is None:
            y=self.waveforms[1]
        if self.df is None:
            raise Exception("DataFrame not loaded. Call 'read_outputs' to load it.")
        return plt.plot(self.df[x], self.df[y], *args, **kwargs)
        
  
    

if __name__ == "__main__":
    simulation = FlybackSimulation()
    simulation.generate_flyback_netlist()
    simulation.begin()
    simulation.read_outputs()

    simulation.plot()
    plt.show()
