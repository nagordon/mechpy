# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 13:03:59 2014
author: Neal Gordon

Typical use of unitconvert are to find the equivalent of a unit at the command line
without the need to be online


"""

  
''' Pint Uses
    # General Use and syntax
    from pint import UnitRegistry
    ureg = UnitRegistry()
    home = 25.4 * ureg.degC
    print(home.to('degF'))
    ft = 1 * ureg.feet
    lb = 10 * ureg.pound_force
    T = ft*lb  
    print(T)
    print(T.magnitude)
    print(T.units)
    print(T.dimensionality)
    print(T.to_base_units())
    ureg.newton * ureg.meter
    print(repr(T))
    T.to(ureg.inch * ureg.pound_force)
    T.to(ureg.inch * ureg.pound_force)
    T.to(ureg.meter * ureg.newton)
    num = 1.0
    x=num*ureg.lbf
    print( x.to(ureg.newton) )
    ureg.lbf.to(ureg.newton)        
'''

''' lb to kg example
# equivalent of kg to lbf
print('\nthe units of mass is ',units.mass)
print('the units of force is ',units.mass*units.acceleration)
m = 1 # kg
a = 9.81 # acceleration at sea level on earth , kg/m*s^2
F = m*a # newton , 1 kg = 9.81 newtons = 2.205 lbf
numout = uc(F,'newton','pound')
print('\n',m,'kilogram =', F, 'newtons =',numout , 'lbf')
'''

''' pint and units used together
print( (units.psi*ureg.psi) / (units.pascal*ureg.pascal) )
print(1.0 *  (units.inch*ureg.inch) / (units.miles*ureg.miles) )
print(1.0* (units.inch*ureg.inch) / (units.millimeter*ureg.millimeter) )
'''
    
''' sympy
from sympy.physics import units
# ksi/MPa
1.0 * (units.psi*1e3) / (units.pascal*1e6)
# lb/N
1.0 * units.pound / units.newton
# km/mile
1.0 * units.kilometer  / units.mile
# mile/km
1.0 * units.mile / ( 1.0* units.kilometer )
# inch/mm
print(1.0 * units.inch / units.millimeter)
# lbs / kg
1.0 *  units.pound / units.kilogram 
# cc/cuin
1.0 *  units.inch**3 / units.centimeter**3
'''

def uc1(numin,frm,to):
    # uses sympy module for unit conversion(uc)
    # converts number 'num' from 'frm' units to 'to' units
    ''' examples
             from    to
    uc(1.0,'pascal','psi')
    uc(1.0,'psi','kPa')
    uc(1.0,'atm','psi')
    uc(1.0,'inch','mm')
    uc(1.0,'kilometer','mile')
    uc(1.0,'mile','kilometer')
    uc(1.0,'newton','pound')
    uc(1.0,'pound','newton')
    '''
    from sympy.physics import units
    strin = 'numin * units.'+frm+'/units.'+to
    numout = float(eval(strin))
    #print(numin , frm , '=' , numout , to)
    print('%.2f %s = %.2f  %s '%(numin, frm, numout, to  ))
    #return numout
    

def uc2(num,frm,to):
    # uses pint module for unit conversion
    # converts number 'num' from 'frm' units to 'to' units
    ''' examples
    uc(17.5,'lbf','newton')
    uc(1,'lbf','newton')
    uc(300,'pascal','psi')
    uc(1,'inch','mm')
    uc(1,'kilometer','mile')
    uc(1,'mile','kilometer')  
    '''
    from pint import UnitRegistry
    ureg = UnitRegistry()
    strin = 'num * ureg.'+frm+'.to(ureg.'+to+')'
    numout = eval(strin)
    print(num , frm , '=' ,numout )
    return numout


def in_mm(n=16):
    # %pylab inline # command in ipython that imports sci modulues
    import sympy as sym
    sym.init_printing()
    for k in range(n+1):
        n = float(n)
        print(' %5s in - %.6f in - %.6f mm ' %  (sym.Rational(k,n) , k/n, 25.4*k/n ) )  
