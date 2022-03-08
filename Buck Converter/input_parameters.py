from sympy.physics.units import *
from datetime import datetime
# this module was meant to be used as an input sheet so the parameters could be displayed in a nice DataFrame. But if they are class attributes, it takes much effort to call them as 'InputParam.Vin'
print(f'Importing module {__name__} at {datetime.now().__str__()[:19]}')


class InputParam:
    Vin = 480*volts    # input voltage
    Vout = 48*volts    # output voltage
    Pout = 1000*watts  # output power
    fs = 40e3*hertz    # switching frequency


