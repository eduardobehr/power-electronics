from fileinput import filename
import subprocess as sp
import pandas as pd
from sys import argv
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def fix_ngspice_csv(filename):
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


def generate_flyback_netlist( _Vin=311,
                    _Vout=12,
                    _Lp=40e-6, 
                    _Rl=1, 
                    _fs=160e3, 
                    _Co=100e-6,
                    _RDSon=1e-3,
                    _endtime=300e-6,
                    _steps_per_switch_period=500,
                    filename_circuit="flyback.cir",
                    filename_csv = "output.csv"
            ):



    
    filename_csv = "output.csv"

    _Ts = 1/_fs
    netlist=f"""
    .title Flyback converter
    .param _N                           = 10; Np/Ns winding ratio
    .param _D                           = {_Vout}/({_Vin}/_N)
    .param _Ls                          = {_Lp}/(_N**2)

    * Primary side
    Vin         nVin        gnd         {_Vin}
    .ic v(nVin) = {_Vin}
    Rin         nVin        nP          1m
    Cin         nP          gnd         1nF         ic={_Vin}
    Lp          nP          drain       {_Lp}
    S1          drain       gnd         gate            gnd         switchModel  OFF
    .model      switchModel sw          vt=0.5          ron={_RDSon}      roff=100Meg
    Cds         drain       gnd         700pF; Drain to source capacitance

    K1          Lp          Ls          0.995

    * Secondary side
    Ls          sgnd        ndiod       {{_Ls}}
    D1          ndiod       nOut        DMOD
    .model DMOD D ( bv=500 is=1e-13 n=1.05)
    Co          nOut        sgnd        {_Co}
    Rl          nOut        sgnd        {_Rl}

    * PWM
    vModulant   signal      gnd         {{_D}}
    BsawtoothP  sawtoothP   gnd         v=0.5+(time*{_fs}-floor(0.5+time*{_fs}))
    Bpwm        gate        gnd         v=v(signal)>=v(sawtoothP) && v(signal)>=0 ? 1 : 0;v(signal)<=v(sawtoothN) && v(signal)<=0 ? -1 : 0


    * Snubber
    Dsnub       drain       snub        DMOD
    Csnb        nP          snub        1nF
    Rsnb        nP          snub        100

    * Misc
    Rgnd        gnd         sgnd        1Meg


    .tran {(1/_fs)/_steps_per_switch_period} {_endtime} uic

    .control
        run
        *plot v(nOut, sgnd)
        *plot Lp#branch Ls#branch    xlimit {_endtime-4*_Ts} {_endtime} 
        *plot drain                  xlimit {_endtime-4*_Ts} {_endtime} 
        wrdata {filename_csv} drain                  xlimit {_endtime-4*_Ts} {_endtime}
        echo Simulation done
        #iplot drain
        exit 
    .endc
    """

    

    with open(filename_circuit, "w") as file:
        file.write(netlist)

    print("Calling Ngspice for generated netlist...")
    print(sp.getoutput("ngspice --version"))
    sp.getoutput(f"ngspice {filename_circuit}")

    fix_ngspice_csv(filename_csv)

    # sp.call(["./plot_csv.py", "Vds.csv"])
    df = pd.read_csv(filename_csv, sep=";", names=["t", "y"])
    # print(df.head())
    # fig=plt.figure(dpi=150)
    return df.plot(x="t", y="y", figsize=(10,8))#, xlim=(df.iloc[int(len(df)/2)]["t"], df.iloc[-1]["t"]))
     
    # plt.show()
    

if __name__ == "__main__":
    if True:
        generate_flyback_netlist()

        df = pd.read_csv("output.csv", sep=";", names=["t", "y"])
        df.plot(x="t", y="y")
        plt.show()
        print(df)