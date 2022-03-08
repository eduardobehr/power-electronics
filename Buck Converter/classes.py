from typing import Union, Tuple, List

POW_DICT = {
    0:'⁰',
    1:'¹',
    2:'²',
    3:'³',
    4:'⁴',
    5:'⁵',
    6:'⁶',
    7:'⁷',
    8:'⁸',
    9:'⁹',
    '.': '⋅'
}

class Unit:
    """ Used for handling physical units beside their numerical values """
    def __init__(self, num: str='', den: str=''):
        self.num: List[str] = list()
        self.den: List[str] = list()
        assert isinstance(num, (str, list)), f'num must be str, not {type(num)}.'
        assert isinstance(den, (str, list)), f'den must be str, not {type(den)}.'
        if isinstance(num, str):
            self.num.append(num)  # numerator
        elif isinstance(num, list):
            self.num.extend(num)  # numerator
            
        if isinstance(den, str):
            self.den.append(den)  # denominator
        elif isinstance(den, list):
            self.num.extend(den)  # denominator
        
        if True:
            self.num = num
            self.den = den
        
    @staticmethod
    def count_duplicates(li: list) -> dict:
        """ Takes a string and returns a dictionary counting the occurences of each character """
        occurences = dict()  # {unit: n⁰ occurences}
        for unit in li:
            occurences.update({str(unit): li.count(unit)})  # not ideal for huge strings, but simple
        return occurences
                
    @staticmethod
    def colapse_duplicates(count_dict: dict, string: str) -> str:
        """ 
        Takes a str and a dict counting its chars, returns a str with exponentiation
        E.g.: '(m)/(ss)' -> '(m)/(s²)'
        """
        ret = string
        duplicates = count_dict
        for key in duplicates:
            if key.isalpha() and duplicates[key] > 1:
                ret = ret.replace(key, '', duplicates[key]-1)
                ret = ret.replace(key, key + POW_DICT[duplicates[key]], 1)
        return ret
    
    @staticmethod
    def cancel_out():
        print('Not cancelling out')
    
    def aggregate_units(self) -> Tuple[str, str]:
        # Cancel out units in both numerator and denominator
        #count each occurence and store counts in dictionary. Then, if >1, use POW_DICT
        #DO IT ONLY WHEN SHOWING THE UNIT (repr, str). DO NOT CHANGE THE ATTRIBUTES num AND den!
        self.cancel_out()
        # count chars in num:
        duplicates_num = self.count_duplicates(self.num)
        # aggregate chars in num:
        num = self.colapse_duplicates(duplicates_num, self.num)
        # count chars in den:
        duplicates_den = self.count_duplicates(self.den)
        # aggregate chars in den:
        den = self.colapse_duplicates(duplicates_den, self.den)
        return num, den
    
    @staticmethod
    def list2str(li: list) -> str:
        print(f'input list: {li}')
        string = ''
        for element in li:
            string += element
        print(f'output string: {string}')
        return string
    
    def __str__(self) -> str:
        num, den = self.aggregate_units()
        n, d = self.list2str(num), self.list2str(den)
        if d and n:
            return f'({n})/({d})'
        elif d and not n:
            return f'1/({d})'
        elif n and not d:
            return n
    
    def __repr__(self) -> str:
        return self.__str__()
    
#     def include_num(self, arg: 'Unit'):
#         assert isinstance(arg, Unit)
#         self.num = 
        
    def __mul__(self, other: 'Unit'):
        assert isinstance(other, Unit), f'Multiplication only valid among instances of {type(self)}'
        return Unit(
            num=self.num + other.num,
            den=self.den + other.den
        )
    
    def __truediv__(self, other: 'Unit'):
        assert isinstance(other, Unit), f'Multiplication only valid among instances of {type(self)}'
        return Unit(
            num=self.num + other.den,
            den=self.den + other.num
        )
    def __pow__(self, value):
        if isinstance(value, (float, int)):
            if self.num and self.den:
                return Unit(num=self.num+POW_DICT[value], den=self.den+POW_DICT[value])
            elif self.num:
                return Unit(num=self.num+POW_DICT[value]) 
    
    pass

class var(float):
    def __init__(self, value, unit:Unit = None):
        float.__init__(value)
        self.value = value
        self.unit = unit
        
    def aggregate_units(self):
        pass
        
        
    def __str__(self):
        return float(self.value).__str__() + ' ' + str(self.unit)
    
    def __add__(self, other: 'var'):
        # TODO
        if not isinstance(other, type(self)):
            ans = self.value + other
            print(f'Warning: adding type {type(self)} with type {type(other)}. Returning type {type(ans)}.')
            return ans
        elif self.unit != other.unit:
            print('Warning: adding variables of different units!')
            return self.value + other
        elif self.unit == other.unit:
            return var(self.value + other.value, unit = self.unit)
            
    
    def __mul__(self, other):
        if not isinstance(other, type(self)):
            return var(self.value * other, unit = self.unit)
        elif isinstance(other, type(self)):
            return var(self.value * other.value, unit = self.unit*self.unit)
    
    def __truediv__(self, other):
        if not isinstance(other, type(self)):
            return var(self.value / other, unit = self.unit)
        elif isinstance(other, type(self)):
            return var(self.value / other.value, unit = self.unit/self.unit)
    
    def __pow__(self, other):
        if not isinstance(other, type(self)):
            return var(self.value ** other, unit = self.unit**other)
        elif isinstance(other, type(self)):
#             return var(self.value * other.value, unit = self.unit*self.unit)
            pass
        
    
if __name__ == '__main__':
    v1 = var(220,unit='V')
    v2 = var(110, unit='V')
    i1 = var(5, unit='A')
    # v1+10
    print(v1 + v2)
    print(v1 + i1)
    print(v1 + 5.)
    print(v1*v2)
    print(v1*i1)