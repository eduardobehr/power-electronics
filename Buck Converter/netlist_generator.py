import subprocess as sp

def BuckNetlist(Vin,
            D, 
            L, 
            Rload, 
            fs, 
            R_inductor, 
            C_out,
            RDSon=20e-3,
            Vdiode=0.7):

    Ts = 1/fs
    DTs = D*Ts

    template=f"""
    .title DC-DC Buck Converter
    SW1 /Vin /Vsw /pwm GND switchModel ON
    Vpwm /pwm GND pulse(0 10 1u 1u 1u {DTs} {Ts})
    R1 /Vout GND {Rload}
    C1 /Vout GND {C_out}
    D1 GND /Vsw DIODE
    L1 /Vsw /Vout {L}
    V1 /Vin GND dc({Vin})
    .save @vpwm[i]
    .save @r1[i]
    .save @l1[i]
    .save @v1[i]
    .save V(/Vin)
    .save V(/Vout)
    .save V(/Vsw)
    .save V(/pwm)

    .model switchModel sw vt=1 vh=0.2 ron={RDSon} roff=1Meg
    .tran 1u 20m

    .control
        run
        * plot V("/Vout")
        *hardcopy Vout.eps "/Vout"
        wrdata Vout.csv @l1[i] ;"/Vout"
        * shell './plot_csv.py Vout.csv'
        exit
    .endc

    .end

    """

    filename="netlist.txt"

    with open(filename, "w") as file:
        file.write(template)

    print("Calling Ngspice for generated netlist...")
    print(sp.getoutput("ngspice --version"))
    sp.getoutput(f"ngspice {filename}")
    sp.call(["./plot_csv.py", "Vout.csv"])
    

if __name__ == "__main__":
    BuckNetlist(
        Vin=400,
        D=0.9,
        L=7.2e-3,
        Rload=324,
        fs=5000,
        R_inductor=0,
        C_out=14e-6
    )