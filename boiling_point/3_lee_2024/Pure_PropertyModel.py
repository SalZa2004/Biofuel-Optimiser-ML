from math import log
from math import exp
from math import log10

from rdkit import Chem
from rdkit.Chem import Descriptors

from numpy import array
from numpy import zeros
from numpy import dot
from numpy.linalg import inv

import Pure_PropertyModel



## Property model parameters estimation for pure component
## Jun-Woo Kim
## 2023-10-13 (version 1.0)
##
## The reference has been recorded below each model function.
## The function order follows the sequence in the manuscript, which is different from the below Python code order.
## (to prevent errors in the solution sequence)
##
##
##
## 0. Nomenclature for process variables
## The unit of process variable has been recorded below each model function.
##
## M : Molecular weight
## T : Temperature
## P : Pressure
## V : Volume
## Tb : Normal boiling temperature
## Tc : critical temperature
## Pc : critical pressure
## Vc : critical volume
## Hform : Heat of formation (ideal gas, 298 K)
## Gform : Gibbs energy of formation (ideal gas, 298 K)
## MUP : Dipole moment
## PL : Vapor pressure
## VL : Liquid molar volume
## omega: Pitzer acentric factor
## CPG : Ideal gas heat capacity
## DHVL : Enthalpy of vaporization
## MUL : Liquid viscosity
## MUG : Gas viscosity
## KL : Liquid thermal viscosity
## KG : Gas thermal viscosity
## SIGMA : Surface tension
##
##
##
## 1. Reference propertymodel function list
##
## 1.1 Joback group contribution method for scalar properties
## (It also serves as an applied model.)
## [M, Tb, Tc, Pc, Vc, Hform, Gform, CPIG] = JOBACK(smiles)
## 1.2 Riedel model for vapor pressure
## PL = RIEDEL(T, Tb, Tc, Pc)
## 1.3 Gunn-Yamada model for liquid molar volume
## VL = GUNN(T, Tc, Pc, omega)
## 1.4 Redlich–Kwong equation of state (RKEOS) objective function form for gas molar volume
## y = OBJRK(T, P, V, Tc, Pc)
## 1.5 Clausius-Clapeyron equation for Enthalpy of vaporization
## DHVL = DHVLCC(T, Tc, Pc, RKTZRA, C)
## 1.6 Letsou-Stiel model for liquid viscosity
## MUL = LETSOU(T, Tc, Pc, M, omega)
## 1.7 Chapman-Enskog-Brokaw model for gas viscosity
## (It also serves as an applied model.)
## MUG = CHAPMAN(T, Tb, Vb, MUP, M)
## 1.8 Sato-Riedel model for liquid thermal conductivity
## KL = SATO(T, Tb, Tc, M)
## 1.9 Stiel-Thodos model for gas thermal conductivity
## (It also serves as an applied model.)
## KG = STIEL(T, Tb, Vb, MUP, M, C_CPIG)
## 1.10 Block-Bird model for surface tension
## SIGMA = BROCK(T, Tb, Tc, Pc)
##
##
##
## 2. Applied propertymodel function list
##
## 2.1 Extended Antoine equation parameters for vapor pressure
## PL = PLXANT(T, C)
## 2.2 Rackett model parameter for liquid molar volume
## VL = RACKETT(T, Tc, Pc, RKTZRA)
## 2.3 Watson model parameters for enthalpy of vaporization
## DHVL = DHVLWT(T, Tc, C)
## 2.4 Aspen polynomial model for ideal gas heat capacity
## CPG = CPIG(T, C)
## 2.5 Andrade model parameters for liquid viscosity
## MUL = MULAND(T, C)
## 2.6 DIPPR equation 100 model for liquid thermal conductivity
## KL = KLDIP(T, Tc, C)
## 2.7 DIPPR equation 106 model for liquid surface tension
## SIGMA = SIGDIP(T, Tc, C)
##
##
##
## 3. Parameter estimation function list
##
## 3.1 PLXANT = PLXANT_PCES(Tb, Tc, Pc)
## 3.2 OMEGA = OMEGA_PCES(Tc, Pc, PLXANT)
## 3.2 RKTZRA = RKTZRA_PCES(omega)
## 3.3.1 DHVLB = DHVLB_PCES(Tb, Tc, Pc, RKTZRA, PLXANT)
## 3.3.2 DHVLWT = DHVLWT_PCES(Tb, Tc, Pc, DHVLB, RKTZRA, PLXANT)
## 3.4 MULAND = MULAND_PCES(Tb, Tc, Pc, M, omega)
## 3.5 KLDIP = KLDIP_PCES(Tb, Tc, M)
## 3.6 SIGDIP = SIGDIP_PCES(Tb, Tc, Pc)
##
##
##
## 4. Helper function list for numerial analysis
##
## 4.1 Newton-Raphson method, newton()
## 4.2 Nelder-Mead method, nelder()
## 4.3 Polynomial curve fitting, polyfit()
## 4.4 MULAND curve fitting, MULANDfit()





# Newton-Raphson method
def newton(func, xi):
    
    def diff(func, x):
        dx = 1E-5
        y1 = func(x - dx)
        y2 = func(x + dx)
        slope = (y2 - y1) / (2 * dx)
        return slope

    tol = 1E-9
    maxiter = 1000

    x = xi
    xn = xi + tol * 2
    n = 0

    while abs(xn - x) > tol and n < maxiter:
        n = n +1
        x = xn
        y = func(x)
        slope = diff(func, x)
        xn = x - y / slope

    return x


# Nelder-Mead method
# https://doi.org/10.1007/s10589-010-9329-3
def nelder(func, xi):
    # Parameters
    alpha = 1         # Scalar parameter (reflection)
    beta = 2          # Scalar parameter (expension)
    gamma = 0.5       # Scalar parameter (contraction)
    delta = 0.5       # Scalar parameter (shrink)
    tau = 0.05        # Scalar parameter (initial simplex for non-zero)
    tau0 = 0.00025    # Scalar parameter (initial simplex for zero)
    Maxiter = 1000    # maximum iteration
    Tol = 1e-9        # Tolerance

    Ns = len(xi)      # Number of simplex points
    n = 0             # Number of iteration

    # Blank matrices
    x = zeros((Ns + 1, Ns))
    f = zeros(Ns + 1)
    x0 = zeros(Ns)
    xr = zeros(Ns)
    xe = zeros(Ns)
    xc = zeros(Ns)
    Sol = zeros(Ns)

    # Initial simplex
    x[0] = xi
    for i in range(0, Ns + 1):
        for j in range(0, Ns):
            if i == j + 1:
                if x[i,j]==0:
                    x[i,j] = x[0,j] + tau0
                else:
                    x[i,j] = x[0,j] + tau
            else: x[i,j] = x[0,j]

    for i in range(0, Ns + 1):
        f[i] = func(x[i])

    # Iterative calculation
    while abs(max(f) - min(f)) > Tol and n < Maxiter:
        n = n + 1

        for i in range(0, Ns + 1):
            f[i] = func(x[i])

        # Step 1. Order
        Index = f.argsort()
        x = x[Index]
        f = f[Index]

        # Step 2. Centroid
        for i in range(0, Ns):
            x0[i] = sum(x[:Ns][:,i]) / Ns

        # Step 3. Reflection
        xr = x0 + alpha * (x0 - x[Ns])
        fr = func(xr)

        if fr >= f[0] and fr < f[Ns-1]:
            x[Ns] = xr
            f[Ns] = fr

        else:
            # Step 4. Expansion
            if fr < f[0]:
                xe = x0 + beta * (xr - x0)
                fe = func(xe)
                if fe < fr:
                    x[Ns] = xe
                    f[Ns] = fe
                else:
                    x[Ns] = xr
                    f[Ns] = fr

            if fr > f[Ns-1]:
                # Step 5. Contraction (Case 1)
                if fr < f[Ns]:
                    xc = x0 + gamma * (xr - x0)
                    fc = func(xc)
                    if fc < fr:
                        x[Ns] = xc
                        f[Ns] = fc
                    # Step 6. Shrink
                    else:
                        xi = x[0] + delta * (xi - x[0])
                        x[0] = xi
                        for i in range(0, Ns + 1):
                            for j in range(0, Ns):
                                if i == j + 1:
                                    if x[i,j]==0:
                                        x[i,j] = x[0,j] + tau0
                                    else:
                                        x[i,j] = x[0,j] + tau
                                else: x[i,j] = x[0,j]
                                
                # Step 5. Contraction (Case 2)
                elif fr > f[Ns]:
                    xc = x0 + gamma * (x[Ns] - x0)
                    fc = func(xc)
                    if fc < f[Ns]:
                        x[Ns] = xc
                        f[Ns] = fc
                    # Step 6. Shrink
                    else:
                        xi = x[0] + delta * (xi - x[0])
                        x[0] = xi
                        for i in range(0, Ns + 1):
                            for j in range(0, Ns):
                                if i == j + 1:
                                    if x[i,j]==0:
                                        x[i,j] = x[0,j] + tau0
                                    else:
                                        x[i,j] = x[0,j] + tau
                                else: x[i,j] = x[0,j]

    # Results
    for i in range(0, Ns):
        Sol[i] = sum(x[:,i]) / (Ns+1)

    fval = sum(f) / (Ns+1)

    return Sol, fval, n


# Polynomial curve fitting
# https://en.wikipedia.org/wiki/Polynomial_regression
def polyfit(x, y, m):
    
    # Vandermonde matrix
    V = array([x**i for i in range(m+1)]).T

    # Coefficients
    Sol = dot(dot(inv(dot(V.T, V)), V.T), y)
    
    return Sol


# MULAND curve fitting
# https://en.wikipedia.org/wiki/Ordinary_least_squares
def MULANDfit(x, y):

    X = zeros((len(x), 3))

    for i in range(len(x)):
        X[i] = array([
            1,
            1 / x[i],
            log(x[i])
            ])
        y[i] = log(y[i])

    # Coefficients
    Sol = dot(dot(inv(dot(X.T, X)), X.T), y)
    
    return Sol


# Aspen PLXANT model
def PLXANT(T, C):
    if T < C[7]:
        dT = 1E-5
        PLmin1 = PLXANT(C[7], C)
        PLmin2 = PLXANT(C[7] + dT, C)
        slope = (log(PLmin1) - log(PLmin2)) / ((1 / ((C[7] + dT) + 273.15)) - (1 / (C[7] + 273.15)))
        lnPL = log(PLmin1) + slope * ((1 / (C[7] + 273.15)) - (1 / (T + 273.15)))
        
    if T > C[8]:
        dT = 1E-5
        PLmax1 = PLXANT(C[8] - dT, C)
        PLmax2 = PLXANT(C[8], C)
        slope = (log(PLmax1) - log(PLmax2)) / ((1 / (C[8] + 273.15)) - (1 / ((C[8] - dT) +273.15)))
        lnPL = log(PLmax2) + slope * ((1 / (C[8] + 273.15)) - (1 / (T + 273.15)))
        lnPLmax = 7 + log(PLXANT(C[8], C))
        lnPL = min(lnPL,lnPLmax)
        
    if T >= C[7] and T <= C[8]:
        lnPL = C[0] + C[1] / ((T + 273.15) + C[2]) + C[3] * (T + 273.15) + C[4] * log((T + 273.15)) + C[5] * (T + 273.15)**C[6]
        
    PL = exp(lnPL)
    return PL


# Redlich–Kwong equation of state (RKEOS) objective function form
# https://doi.org/10.1021/cr60137a013
def OBJRK(T, P, V, Tc, Pc):
    # T : temperature (C)
    # P : pressure (bar)
    # V : volume (cum/kmol)
    # Tc: critical temperature (C)
    # Pc: critical pressure (bar)    

    # C to K    
    T = T + 273.15
    Tc = Tc + 273.15
    
    R = 8.31446E-2
    a = 0.4278 * R**2 * Tc**2.5 / Pc
    b = 0.08664 * R * Tc / Pc

    # Objective function
    y = (R * T * V * (V + b) * T**0.5 - a * (V - b)) - (P * (V - b) * V * (V + b) * T**0.5)

    return y


# Riedel model
# https://doi.org/10.1016/j.fluid.2005.12.018
def RIEDEL(T, Tb, Tc, Pc):
    # T : temperature (C)
    # Tb: boiling temperature (C)
    # Tc: critical temperature (C)
    # Pc: critical pressure (bar)

    # C to K
    T = T + 273.15    
    Tb = Tb + 273.15
    Tc = Tc + 273.15
    
    Tbr = Tb / Tc
    Tr = T / Tc

    K = 0.0838
    PSIb = -35 + 36 / Tbr + 42 * log(Tbr) - Tbr**6
    alphac = (3.758 * K * PSIb + log(Pc / 1.01325)) / (K * PSIb - log(Tbr))
    
    Q = K * (3.758 - alphac)
    A = -35 * Q
    B = -36 * Q
    C = 42 * Q + alphac
    D = -Q

    lnPLr = A - B / Tr + C * log(Tr) + D * Tr**6
    PLr = exp(lnPLr)
    PL = PLr * Pc
    
    return PL


# Gunn-Yamada Model
# https://doi.org/10.1002/aic.690170613
def GUNN(T, Tc, Pc, omega):
    # T : temperature (C)
    # Tc: critical temperature (C)
    # Pc: critical pressure (bar)
    # omega: Pitzer acentric factor

    # C to K
    T = T + 273.15
    Tc = Tc + 273.15

    R = 0.08314463
    Zsc = 0.2920 - 0.0967 * omega
    Vsc = Zsc / Pc * R * Tc
    Tr = T / Tc
    
    if Tr < 0.2: print("Warning: temperature is out of ragne. Tr is must be over 0.2")
    
    if Tr >= 0.2:
        delta = 0.29607 - 0.09045 * Tr - 0.04842 * Tr**2
        
        if Tr < 0.8:
            Vr0 = 0.33593 - 0.33953 * Tr + 1.51941 * Tr**2 - 2.02512 * Tr**3 + 1.11422 * Tr**4
            
        if Tr >= 0.8 and Tr<1:
            Vr0=1.0+1.3*(1-Tr)**0.5*log10(1-Tr)-0.50879*(1-Tr)-0.91534*(1-Tr)**2
            
        if Tr == 1:
            Vr0 = 1
            
    VL = Vr0 * (1 - omega * delta) * Vsc
    return VL


# Rackett Model
# https://pubs.acs.org/doi/10.1021/je60047a012
def RACKETT(T, Tc, Pc, RKTZRA):
    # T : temperature (C)
    # Tc: critical temperature (C)
    # Pc: critical pressure (bar)
    # RKTZRA: Rackett parameter

    # C to K    
    T = T + 273.15
    Tc = Tc + 273.15
    
    R = 0.08314463
    Tr = T / Tc
    VRA = R * Tc / Pc * RKTZRA

    if Tr <= 0.99:
        VL = 10**((1 + (1 - Tr)**(2 / 7)) * log10(RKTZRA) - log10(Pc / R / Tc))

    if Tr > 0.99:
        dT = 1E-5
        V0 = RACKETT((Tc * 0.99) - 273.15, Tc - 273.15, Pc, RKTZRA)
        V1 = RACKETT((Tc * 0.99) - 273.15 - dT, Tc - 273.15, Pc, RKTZRA)
        VDUM1 = VRA - V0
        slope = (V0 - V1) / dT
        TDUM2 = -VDUM1 / slope - ((VDUM1 / slope)**2 + VDUM1**2)**0.5
        T00 = 0.99 * Tc + abs(TDUM2)
        if T > T00:
            VL = VRA
        else:
            Vr = TDUM2 / slope
            V00 = V0 + Vr
            a = 1
            b = -2 * V00
            c = -(VRA**2 - 2 * VRA * V00 + V00**2 - V00**2 - T**2 + 2 * T * T00 - T00**2)
            VL = (-b + (b**2 - 4 * a * c)**0.5) / (2 * a)
        
    return VL


# Clausius–Clapeyron equation
def DHVLCC(T, Tc, Pc, RKTZRA, C):
    # T : temperature (C)
    # Tc: critical temperature (C)
    # Pc: critical pressure (bar)
    # RKTZRA: Rackett parameter
    # C : PLXANT coefficients

    # C to K
    T = T + 273.15
    Tc = Tc + 273.15

    R = 8.31446E-2    # cum-bar/K-kmol
    
    if T >= Tc:
        PL = PLXANT(T - 273.15, C)
        slope = 0
    else:
        dT = 1E-5
        PL = PLXANT(T - 273.15, C)
        PL2 = PLXANT(T - 273.15 - dT, C)
        slope = (PL - PL2) / dT
        
    def OBJ(V):
        y = OBJRK(T - 273.15, PL, V, Tc - 273.15, Pc)
        return y

    xi = R * T / PL
    VG = newton(OBJ, xi)
    VL = RACKETT(T - 273.15, Tc - 273.15, Pc, RKTZRA)
    DHVL = slope * T * (VG - VL) / 10 / 4.1868

    return DHVL


# Watson model
# https://doi.org/10.1021/ie50256a006
def DHVLWT(T, Tc, C):
    # T : temperature (C)
    # Tc: critical temperature (C)
    # C : coefficients of DHVLWT
    
    if T >= C[4]:
        # C to K
        T = T + 273.15
        Tc = Tc + 273.15
        Tb = C[1] + 273.15
        DHVL = C[0] * ((1 - T / Tc) / (1 - Tb / Tc))**(C[2] + C[3] * (1 - T / Tc))
    else:
        dT = 1E-5
        DHVL1 = DHVLWT(C[4], Tc, C)
        DHVL2 = DHVLWT(C[4] + dT, Tc, C)
        slope = (DHVL2 - DHVL1) / dT
        DHVL = DHVL1 + slope * (T - C[4])
    return DHVL


# Aspen CPIG Model
def CPIG(T, C):
    # T : temperature (C)
    
    if T < C[6]:
        # C to T
        CPG = C[8] + C[9] * (T + 273.15)**C[10]
        
    if T > C[7]:
        dT = 1E-5
        CPGmax = CPIG(C[7], C)
        CPGmax1 = CPIG(C[7] - dT, C)
        slope = (CPGmax - CPGmax1) / dT
        CPG = CPGmax + slope * (T - C[7])
        
    if T >= C[6] and T <= C[7]:
        CPG = C[0] + C[1] * T + C[2] * T**2 + C[3] * T**3 + C[4] * T**4 + C[5] * T**5

    return CPG


# MULAND model
def MULAND(T, C):
    # T : temperature (C)

    # C to K
    T = T + 273.15
    C4 = C[3] + 273.15
    C5 = C[4] + 273.15

    if T < C4:
        dT = 1E-5
        MULmin1 = MULAND(C4 - 273.15, C)
        MULmin2 = MULAND(C4 - 273.15 + dT, C)
        slope = (log(MULmin1) - log(MULmin2)) / ((1 / (C4 + dT)) - (1 / C4))
        lnMUL = log(MULmin1) + slope * ((1 / C4) - (1 / T))

    if T > C5:
        dT = 1E-5
        MULmax1 = MULAND(C5 - 273.15 - dT, C)
        MULmax2 = MULAND(C5 - 273.15, C)
        slope = (log(MULmax1) - log(MULmax2)) / ((1 / C5) - (1 / (C5 -dT)))
        lnMUL = log(MULmax2) + slope * ((1 / C5) - (1 / T))

    if T >= C4 and T <= C5:
        lnMUL = C[0] + C[1] / T + C[2] * log(T)

    MUL = exp(lnMUL)
        
    return MUL


# Letsou-Stiel model
# https://doi.org/10.1002/aic.690190241
def LETSOU(T, Tc, Pc, M, omega):
    # T : temperature (C)
    # Tc: critical temperature (C)
    # Pc: critical pressure (bar)
    # M : molecular weight (g/mol)
    # omega: Pitzer acentric factor

    # C to K
    T = T + 273.15
    Tc = Tc + 273.15

    Tr = T / Tc
    xi = Tc**(1 / 6) / M**(1 / 2) / Pc**(2 / 3)
    
    etaxi0 = 0.015174 - 0.02135 * Tr + 0.0075 * Tr**2
    etaxi1 = 0.042552 - 0.07674 * Tr + 0.0340 * Tr**2
    
    etaxi = etaxi0 + omega * etaxi1
    MUL = etaxi / xi
    return MUL


# Chpaman-Enskog-Brokaw model
# https://doi.org/10.1021/i260030a015
# https://doi.org/10.1063/1.1678363
def CHAPMAN(T, Tb, Vb, MUP, M):
    # T : temperature (C)
    # Tb: normal boiling temperature (C)
    # Vb: volume at TB (cum/kmol)
    # MUP: dipole moment (debye)
    # M: molecular weight (g/mol)
    
    # C to K
    T = T + 273.15
    Tb = Tb + 273.15
    
    delta = 1.94 * 10**3 * MUP**2 / (Vb * 1000) / Tb
    sigma = (1.585 * (Vb * 1000) / (1 + 1.3 * delta**2))**(1 / 3)
    LJ = 1.18 * (1 + 1.3 * delta**2) * Tb
    Tstar = 1 / LJ * T    
    O22n = 1.16145 * Tstar**-0.14874 + 0.52487 * exp(-0.7732 * Tstar) + 2.16178 * exp(-2.43787 * Tstar)
    O22p = O22n + 0.2 * delta**2 / Tstar
    MUG = 26.693 * (M * T)**0.5 / (sigma**2 * O22p) * 10**-4

    return MUG


# DIPPR model
def KLDIP(T, Tc, C):
    # T : temperature (C)

    if T < C[5]:
        dT = 1E-5
        Tmin = C[5]
        KLmin1 = C[0] + C[1] * Tmin + C[2] * Tmin**2 + C[3] * Tmin**3 + C[4] * Tmin**4
        KLmin2 = C[0] + C[1] * (Tmin + dT) + C[2] * (Tmin + dT)**2 + C[3] * (Tmin + dT)**3 + C[4] * (Tmin + dT)**4
        slope = (KLmin2 - KLmin1) / dT
        KL = KLmin1 - slope * (Tmin - T)

    if T > C[6] and T < Tc:
        dT = 1E-5
        Tmax = C[6]
        KLmax1 = C[0] + C[1] * Tmax + C[2] * Tmax**2 + C[3] * Tmax**3 + C[4] * Tmax**4
        KLmax2 = C[0] + C[1] * (Tmax - dT) + C[2] * (Tmax - dT)**2 + C[3] * (Tmax - dT)**3 + C[4] * (Tmax - dT)**4
        slope = (KLmax1 - KLmax2) / dT
        KL = KLmax1 + slope * (T - Tmax)

    if T >= Tc:
        KL = C[0] + C[1] * Tc + C[2] * Tc**2 + C[3] * Tc**3 + C[4] * Tc**4        

    if T >= C[5] and T <= C[6]:
        KL = C[0] + C[1] * T + C[2] * T**2 + C[3] * T**3 + C[4] * T**4
    
    return KL


# Sato-Riedel model
# https://doi.org/10.1016/j.egypro.2014.01.066
def SATO(T, Tb, Tc, M):
    # T : temperature (C)
    # Tb: normal boiling point (C)
    # Tc: critical temperature (C)
    # M : molecular weight (g/mol)

    # C to K
    T = T + 273.15
    Tb = Tb + 273.15
    Tc = Tc + 273.15

    Tr = T / Tc
    Tbr = Tb / Tc
    
    KL = 0.9510 / M**0.5 * (3 + 20 * (1 - Tr)**(2/3)) / (3 + 20 * (1 - Tbr)**(2/3))

    return KL


# Stiel-Thodos model
# https://doi.org/10.1002/aic.690100114
# R. C. Reid, J. M. Prausnitz, B. E. Poling, The Properties of Gases and Liquids, 4th Ed. 
def STIEL(T, Tb, Vb, MUP, M, C_CPIG):
    R = 8.314    # J/mol-K
    CPG = CPIG(T, C_CPIG) * 4.1868    # cal/mol-K to J/mol-K
    MUG = CHAPMAN(T, Tb, Vb, MUP, M)    # cP
    
    KG = (1.15 + 2.03 / (CPG / R - 1)) * MUG * (CPG - R) / M    # W/m-K
    KG = KG * 3.6 / 4.1868    # kcal-m/hr-sqm-K

    return KG


# Brock-Bird model
# https://doi.org/10.1002/aic.690010208
# https://doi.org/10.1021/i160005a015
def BROCK(T, Tb, Tc, Pc):
    # T : temperature (C)
    # Tb: normal boilng temperature (C)
    # Tc: critical tempearture (C)
    # Pc: critical pressure (bar)

    # C to K
    T = T + 273.15
    Tb = Tb + 273.15
    Tc = Tc + 273.15

    Tr = T / Tc
    Tbr = Tb / Tc

    #ac = 0.9076 * (1 + (Tbr * log(Pc)) / (1 - Tbr))
    ac = 0.9076 * (1 + (Tbr * log(Pc / 1.01325)) / (1 - Tbr))
    SIGMA = Pc**(2 / 3) * Tc**(1 / 3) * (-0.281 + 0.133 * ac) * (1 - Tr)**(11 / 9)

    return SIGMA


# SIGDIP(106) model 
def SIGDIP(T, Tc, C):
    # T : temperature (C)
    # Tc : critical tempearture (C)

    # C to K
    T = T + 273.15
    Tc = Tc + 273.15
    Tmin = C[5] + 273.15
    Tmax = C[6] + 273.15

    Tr =T / Tc

    if T < Tmin:
        dT = 1E-5
        Trmin1 = Tmin / Tc
        Trmin2 = (Tmin + dT) / Tc
        
        SIGMAmin1 = C[0] * (1 - Trmin1)**(C[1] + C[2] * Trmin1 + C[3] * Trmin1**2 + C[4] * Trmin1**3)
        SIGMAmin2 = C[0] * (1 - Trmin2)**(C[1] + C[2] * Trmin2 + C[3] * Trmin2**2 + C[4] * Trmin2**3)
        slope = (SIGMAmin2 - SIGMAmin1) / dT
        SIGMA = SIGMAmin1 - slope * (Tmin - T)

    if T > Tmax:
        dT = 1E-5
        Trmax1 = Tmax / Tc
        Trmax2 = (Tmax - dT) / Tc
        
        SIGMAmax1 = C[0] * (1 - Trmax1)**(C[1] + C[2] * Trmax1 + C[3] * Trmax1**2 + C[4] * Trmax1**3)
        SIGMAmax2 = C[0] * (1 - Trmax2)**(C[1] + C[2] * Trmax2 + C[3] * Trmax2**2 + C[4] * Trmax2**3)
        slope = (SIGMAmax1 - SIGMAmax2) / dT
        SIGMA = SIGMAmax1 - slope * (Tmax - T)
        SIGMA = max(SIGMA, 0)

    if T >= Tmin and T<= Tmax:
        SIGMA = C[0] * (1 - Tr)**(C[1] + C[2] * Tr + C[3] * Tr**2 + C[4] * Tr**3)

    return SIGMA
    

# PLXANT parameter estimation
# https://doi.org/10.1016/j.fluid.2005.12.018
def PLXANT_PCES(Tb, Tc, Pc):
    # Tb: boiling temperature (C)
    # Tc: critical temperature (C)
    # Pc: critical pressure (bar)
    
    Tb = Tb + 273.15    # C to K
    Tc = Tc + 273.15    # C to K
    Tbr = Tb / Tc

    K = 0.0838
    PSI = -35 + 36 / Tbr + 42 * log(Tbr) - Tbr**6
    alpha = (3.758 * K * PSI + log(Pc / 1.01325)) / (K * PSI - log(Tbr))
    
    Q = K * (3.758 - alpha)
    A = -35 * Q
    B = -36 * Q
    C = 42 * Q + alpha
    D = -Q
    
    C1 = A + log(Pc) - C * log(Tc)
    C2 = -B * Tc
    C3 = 0
    C4 = 0
    C5 = C
    C6 = D / Tc**6
    C7 = 6
    C8 = Tb - 273.15
    C9 = Tc - 273.15
    C = [C1, C2, C3, C4, C5, C6, C7, C8, C9]
    
    return C


# Pitzer acentric factor estimation
# https://doi.org/10.1021/ja01618a001
def OMEGA_PCES(Tc, Pc, PLXANTC):
    # Tc: critical temperature (C)
    # Pc: critical pressure (bar)
    # PLXANTC: Parameters of PLXANT

    Tc = Tc + 273.15    # C to K
    
    Tr = 0.7    
    T = Tc * Tr
    PLd7 = PLXANT(T - 273.15, PLXANTC)
    omega = -log10(PLd7 / Pc) - 1

    return omega


# RKTZRA parameter estimation
# https://doi.org/10.1021/ja01618a001
def RKTZRA_PCES(omega):

    RKTZRA = 0.2918 - 0.0928 * omega

    return RKTZRA


# DHVLB parameter estimation
def DHVLB_PCES(Tb, Tc, Pc, RKTZRA, PLXANT):

    DHVLB = DHVLCC(Tb, Tc, Pc, RKTZRA, PLXANT)

    return DHVLB


# DHVLWT parameter estimation
def DHVLWT_PCES(Tb, Tc, Pc, DHVLB, RKTZRA, PLXANT):
    # DHVLCC data from Tb to Tc
    DATAref = [ ]

    Ti = Tb
    Tf = Tc
    n = 10

    for i in range(0, n):
        T = Ti + i * (Tf - Ti) / (n - 1)
        DATAref.append(DHVLCC(T, Tc, Pc, RKTZRA, PLXANT))

    def func(x):    

        C = array([
            DHVLB,
            Tb,
            x[0],
            x[1],
            (Tb + 273.15) * - 273.15
            ])

        # Simulation
        f = 0
        Ti = Tb
        Tf = Tc
        n = len(DATAref)
        DATAcalc = zeros(n)
        
        for i in range(0, n):
            T = Ti + i * (Tf - Ti) / (n - 1)
            DATAcalc[i] = Pure_PropertyModel.DHVLWT(T, Tc, C)

        f = sum((DATAref - DATAcalc)**2)
        
        return f


    # Nelder-Mead method
    # Initial value calculation
    
    DATAi = array([
        [Tb + (Tc - Tb) / 3, DHVLCC(Tb + (Tc - Tb) / 3, Tc, Pc, RKTZRA, PLXANT)],
        [Tb + (Tc - Tb) * 2 / 3, DHVLCC(Tb + (Tc - Tb) * 2/ 3, Tc, Pc, RKTZRA, PLXANT)],
        ])

    # C to K
    DATAi[:, 0] = DATAi[:, 0] + 273.15
    Tb = Tb + 273.15
    Tc = Tc + 273.15

    X = array([
        log(DATAi[0, 1] / DHVLB),
        log(DATAi[1, 1] / DHVLB)
        ])
    Y = array([
        log((1 - DATAi[0, 0] / Tc) / (1 - Tb / Tc)),
        log((1 - DATAi[1, 0] / Tc) / (1 - Tb / Tc))
        ])

    E = X / Y

    Z = array([
        1 - DATAi[0, 0] / Tc,
        1 - DATAi[1, 0] / Tc
        ])

    xi = zeros(2)
    xi[1] = (E[0] - E[1]) / (Z[0] - Z[1])
    xi[0] = E[0] - Z[0] * xi[1]

    [Sol, fval, n] = nelder(func, xi)

    DHVLWT = [
        DHVLB,
        Tb - 273.15,
        Sol[0],
        Sol[1],
        (Tb * 0.4 - 273.15)
        ]
        
    return DHVLWT


# MULAND parameter estimation
def MULAND_PCES(Tb, Tc, Pc, M, omega): 
    C4 = Tb
    C5 = (Tc + 273.15) * 0.99 - 273.15

    Ti = C4
    Tf = C5
    n = 10

    DataT = zeros(n)
    DataMUL = zeros(n)

    for i in range(0, n):
        T = Ti + i * (Tf - Ti) / (n - 1)
        MUL = LETSOU(T, Tc, Pc, M, omega)
        DataT[i] = (T + 273.15)
        DataMUL[i] = MUL

    Sol = MULANDfit(DataT, DataMUL)

    MULAND = [
        Sol[0],
        Sol[1],
        Sol[2],
        Tb,
        (Tc + 273.15) * 0.99 - 273.15
        ]
    
    return MULAND


# KLDIP parameter estimation
def KLDIP_PCES(Tb, Tc, M): 
    Ti = Tb
    Tf = (Tc + 273.15) * 0.99 - 273.15
    n = 10

    DataT = zeros(n)
    DataKL = zeros(n)

    for i in range(0, n):
        T = Ti + i * (Tf - Ti) / (n - 1)
        KL = SATO(T, Tb, Tc, M)
        DataT[i] = T
        DataKL[i] = KL

    Sol = polyfit(DataT, DataKL, 4)

    KLDIP = [
        Sol[0],
        Sol[1],
        Sol[2],
        Sol[3],
        Sol[4],
        Tb,
        (Tc + 273.15) * 0.99 - 273.15
        ]
    
    return KLDIP
    

# SIGDIP parameter estimation
def SIGDIP_PCES(Tb, Tc, Pc):
    Tc = Tc + 273.15
    Tb = Tb + 273.15
    Tbr = Tb / Tc
    #ac = 0.8942 * (1 + (Tbr * log(Pc / 1.01325)) / (1 - Tbr))
    ac = 0.9076 * (1 + (Tbr * log(Pc)) / (1 - Tbr))

    SIGDIP = [
        Pc**(2 / 3) * Tc**(1 / 3) * (-0.281 + 0.133 * ac),
        11 / 9,
        0,
        0,
        0,
        Tb -273.15,
        Tc * 0.98 - 273.15
        ]

    return SIGDIP    
    

# Joback group contribution theory
# https://doi.org/10.1021/acsomega.7b01464
def JOBACK(SMILES):
    # Future work: rdkit으로 구한 총 원자의 수와 JOBACK method로 구한 원자의 수가 일치하지 안을 때 경고 메세지 출력 필요
    molecule = Chem.MolFromSmiles(SMILES)
    
    

    # Blank
    group = [0 for i in range(41)]
    SMARTS = [0 for i in range(41)]
    n = [0 for i in range(41)]

    # SMARTS Codes
    group[0] = '-CH3 (non-ring)'; SMARTS[0] = Chem.MolFromSmarts('[CX4H3]')
    group[1] = '-CH2- (non-ring)'; SMARTS[1] = Chem.MolFromSmarts('[!R;CX4H2]')
    group[2] = '>CH- (non-ring)'; SMARTS[2] = Chem.MolFromSmarts('[!R;CX4H]')
    group[3] = '>C< (non-ring)'; SMARTS[3] = Chem.MolFromSmarts('[!R;CX4H0]')
    group[4] = '=CH2 (non-ring)'; SMARTS[4] = Chem.MolFromSmarts('[CX3H2]')
    group[5] = '=CH- (non-ring)'; SMARTS[5] = Chem.MolFromSmarts('[!R;CX3H1;!$([CX3H1](=O))]')
    group[6] = '=C< (non-ring)'; SMARTS[6] = Chem.MolFromSmarts('[$([!R;#6X3H0]);!$([!R;#6X3H0]=[#8])]')
    group[7] = '=C= (non-ring)'; SMARTS[7] = Chem.MolFromSmarts('[$([CX2H0](=*)=*)]')
    group[8] = '≡CH (non-ring)'; SMARTS[8] = Chem.MolFromSmarts('[$([CX2H1]#[!#7])]') 
    group[9] = '≡C− (non-ring)'; SMARTS[9] = Chem.MolFromSmarts('[$([CX2H0]#[!#7])]') 
    group[10] = '−CH2− (ring)'; SMARTS[10] = Chem.MolFromSmarts('[R;CX4H2]')
    group[11] = '>CH- (ring)'; SMARTS[11] = Chem.MolFromSmarts('[R;CX4H]')
    group[12] = '>C< (ring)'; SMARTS[12] = Chem.MolFromSmarts('[R;CX4H0]')
    group[13] = '=CH- (ring)'; SMARTS[13] = Chem.MolFromSmarts('[R;CX3H1,cX3H1]')
    group[14] = '=C< (ring)'; SMARTS[14] = Chem.MolFromSmarts('[$([R;#6X3H0]);!$([R;#6X3H0]=[#8])]')
    group[15] = '-F'; SMARTS[15] = Chem.MolFromSmarts('[F]')
    group[16] = '-Cl'; SMARTS[16] = Chem.MolFromSmarts('[Cl]')
    group[17] = '-Br'; SMARTS[17] = Chem.MolFromSmarts('[Br]') 
    group[18] = '-I'; SMARTS[18] = Chem.MolFromSmarts('[I]')
    group[19] = '-OH (alcohol)'; SMARTS[19] = Chem.MolFromSmarts('[OX2H;!$([OX2H]-[#6]=[O]);!$([OX2H]-a)]')
    group[20] = '-OH (phenol)'; SMARTS[20] = Chem.MolFromSmarts('[O;H1;$(O-!@c)]')
    group[21] = '-O- (non-ring)'; SMARTS[21] = Chem.MolFromSmarts('[OX2H0;!R;!$([OX2H0]-[#6]=[#8])]')
    group[22] = '-O- (ring)'; SMARTS[22] = Chem.MolFromSmarts('[#8X2H0;R;!$([#8X2H0]~[#6]=[#8])]') 
    group[23] = '>C=O (non-ring)'; SMARTS[23] = Chem.MolFromSmarts('[$([CX3H0](=[OX1]));!$([CX3](=[OX1])-[OX2]);!R]=O')
    group[24] = '>C=O (ring)'; SMARTS[24] = Chem.MolFromSmarts('[$([#6X3H0](=[OX1]));!$([#6X3](=[#8X1])~[#8X2]);R]=O')
    group[25] = 'O=CH- (aldehyde)'; SMARTS[25] = Chem.MolFromSmarts('[CH;D2;$(C-!@C)](=O)')
    group[26] = '-COOH (acid)'; SMARTS[26] = Chem.MolFromSmarts('[OX2H]-[C]=O')
    group[27] = '-COO- (ester)'; SMARTS[27] = Chem.MolFromSmarts('[#6X3H0;!$([#6X3H0](~O)(~O)(~O))](=[#8X1])[#8X2H0]')
    group[28] = '=O (other than above)'; SMARTS[28] = Chem.MolFromSmarts('[OX1H0;!$([OX1H0]~[#6X3]);!$([OX1H0]~[#7X3]~[#8])]')
    group[29] = '-NH2'; SMARTS[29] = Chem.MolFromSmarts('[NX3H2]')
    group[30] = '>NH (non-ring)'; SMARTS[30] = Chem.MolFromSmarts('[NX3H1;!R]')
    group[31] = '>NH (ring)'; SMARTS[31] = Chem.MolFromSmarts('[#7X3H1;R]') 
    group[32] = '>N- (non-ring)'; SMARTS[32] = Chem.MolFromSmarts('[#7X3H0;!$([#7](~O)~O)]')
    group[33] = '-N= (non-ring)'; SMARTS[33] = Chem.MolFromSmarts('[#7X2H0;!R]')
    group[34] = '-N= (ring)'; SMARTS[34] = Chem.MolFromSmarts('[#7X2H0;R]')
    group[35] = '=NH'; SMARTS[35] = Chem.MolFromSmarts('[#7X2H1]')
    group[36] = '-CN'; SMARTS[36] = Chem.MolFromSmarts('[#6X2]#[#7X1H0]')
    group[37] = '-NO2'; SMARTS[37] = Chem.MolFromSmarts('[$([#7X3,#7X3+][!#8])](=[O])∼[O-]')
    group[38] = '-SH'; SMARTS[38] = Chem.MolFromSmarts('[SX2H]')
    group[39] = '-S- (non-ring)'; SMARTS[39] = Chem.MolFromSmarts('[#16X2H0;!R]')
    group[40] = '-S- (ring)'; SMARTS[40] = Chem.MolFromSmarts('[#16X2H0;R]')

    # Parameters: https://en.wikipedia.org/wiki/Joback_method
    # [Tc,Pc,Vc,Tb,Tm,Hform,Gform,a,b,c,d,Hfusion,Hvap,ηa,ηb]
    p = [
    [0.0141,    -0.0012,    65,     23.58,      -5.1,       -76.45,     -43.96,     1.95E+01,       -8.08E-03,      1.53E-04,       -9.67E-08,      0.908,      2.373,      548.29,     -1.719],
    [0.0189,    0,          56,     22.88,      11.27,      -20.64,     8.42,       -9.09E-01,      9.50E-02,       -5.44E-05,      1.19E-08,       2.59,       2.226,      94.16,      -0.199],
    [0.0164,    0.002,      41,     21.74,      12.64,      29.89,      58.36,      -2.30E+01,      2.04E-01,       -2.65E-04,      1.20E-07,       0.749,      1.691,      -322.15,    1.187],
    [0.0067,    0.0043,     27,     18.25,      46.43,      82.23,      116.02,     -6.62E+01,      4.27E-01,       -6.41E-04,      3.01E-07,       -1.46,      0.636,      -573.56,    2.307],
    [0.0113,    -0.0028,    56,     18.18,      -4.32,      -9.63,      3.77,       2.36E+01,       -3.81E-02,      1.72E-04,       -1.03E-07,      -0.473,     1.724,      495.01,     -1.539],
    [0.0129,    -0.0006,    46,     24.96,      8.73,       37.97,      48.53,      -8,             1.05E-01,       -9.63E-05,      3.56E-08,       2.691,      2.205,      82.28,      -0.242],
    [0.0117,    0.0011,     38,     24.14,      11.14,      83.99,      92.36,      -2.81E+01,      2.08E-01,       -3.06E-04,      1.46E-07,       3.063,      2.138,      'NA',       'NA'],
    [0.0026,    0.0028,     36,     26.15,      17.78,      142.14,     136.7,      2.74E+01,       -5.57E-02,      1.01E-04,       -5.02E-08,      4.72,       2.661,      'NA',       'NA'],
    [0.0027,    -0.0008,    46,     9.2,        -11.18,     79.3,       77.71,      2.45E+01,       -2.71E-02,      1.11E-04,       -6.78E-08,      2.322,      1.155,      'NA',       'NA'],
    [0.002,     0.0016,     37,     27.38,      64.32,      115.51,     109.82,     7.87,           2.01E-02,       -8.33E-06,      1.39E-09,       4.151,      3.302,      'NA',       'NA'],
    [0.01,      0.0025,     48,     27.15,      7.75,       -26.8,      -3.68,      -6.03,          8.54E-02,       -8.00E-06,      -1.80E-08,      0.49,       2.398,      307.53,     -0.798],
    [0.0122,    0.0004,     38,     21.78,      19.88,      8.67,       40.99,      -2.05E+01,      1.62E-01,       -1.60E-04,      6.24E-08,       3.243,      1.942,      -394.29,    1.251],
    [0.0042,    0.0061,     27,     21.32,      60.15,      79.72,      87.88,      -9.09E+01,      5.57E-01,       -9.00E-04,      4.69E-07,       -1.373,     0.644,      'NA',       'NA'],
    [0.0082,    0.0011,     41,     26.73,      8.13,       2.09,       11.3,       -2.14,          5.74E-02,       -1.64E-06,      -1.59E-08,      1.101,      2.544,      259.65,     -0.702],
    [0.0143,    0.0008,     32,     31.01,      37.02,      46.43,      54.05,      -8.25,          1.01E-01,       -1.42E-04,      6.78E-08,       2.394,      3.059,      -245.74,    0.912],
    [0.0111,    -0.0057,    27,     -0.03,      -15.78,     -251.92,    -247.19,    2.65E+01,       -9.13E-02,      1.91E-04,       -1.03E-07,      1.398,      -0.67,      'NA',       'NA'],
    [0.0105,    -0.0049,    58,     38.13,      13.55,      -71.55,     -64.31,     3.33E+01,       -9.63E-02,      1.87E-04,       -9.96E-08,      2.515,      4.532,      625.45,     -1.814],
    [0.0133,    0.0057,     71,     66.86,      43.43,      -29.48,     -38.06,     2.86E+01,       -6.49E-02,      1.36E-04,       -7.45E-08,      3.603,      6.582,      738.91,     -2.038],
    [0.0068,    -0.0034,    97,     93.84,      41.69,      21.06,      5.74,       3.21E+01,       -6.41E-02,      1.26E-04,       -6.87E-08,      2.724,      9.52,       809.55,     -2.224],
    [0.0741,    0.0112,     28,     92.88,      44.45,      -208.04,    -189.2,     2.57E+01,       -6.91E-02,      1.77E-04,       -9.88E-08,      2.406,      16.826,     2173.72,    -5.057],
    [0.024,     0.0184,     -25,    76.34,      82.83,      -221.65,    -197.37,    -2.81,          1.11E-01,       -1.16E-04,      4.94E-08,       4.49,       12.499,     3018.17,    -7.314],
    [0.0168,    0.0015,     18,     22.42,      22.23,      -132.22,    -105,       2.55E+01,       -6.32E-02,      1.11E-04,       -5.48E-08,      1.188,      2.41,       122.09,     -0.386],
    [0.0098,    0.0048,     13,     31.22,      23.05,      -138.16,    -98.22,     1.22E+01,       -1.26E-02,      6.03E-05,       -3.86E-08,      5.879,      4.682,      440.24,     -0.953],
    [0.038,     0.0031,     62,     76.75,      61.2,       -133.22,    -120.5,     6.45,           6.70E-02,       -3.57E-05,      2.86E-09,       4.189,      8.972,      340.35,     -0.35],
    [0.0284,    0.0028,     55,     94.97,      75.97,      -164.5,     -126.27,    3.04E+01,       -8.29E-02,      2.36E-04,       -1.31E-07,      'NA',       6.645,      'NA',       'NA'],
    [0.0379,    0.003,      82,     72.24,      36.9,       -162.03,    -143.48,    3.09E+01,       -3.36E-02,      1.60E-04,       -9.88E-08,      3.197,      9.093,      740.92,     -1.713],
    [0.0791,    0.0077,     89,     169.09,     155.5,      -426.72,    -387.87,    2.41E+01,       4.27E-02,       8.04E-05,       -6.87E-08,      11.051,     19.537,     1317.23,    -2.578],
    [0.0481,    0.0005,     82,     81.1,       53.6,       -337.92,    -301.95,    2.45E+01,       4.02E-02,       4.02E-05,       -4.52E-08,      6.959,      9.633,      483.88,     -0.966],
    [0.0143,    0.0101,     36,     -10.5,      2.08,       -247.61,    -250.83,    6.82,           1.96E-02,       1.27E-05,       -1.78E-08,      3.624,      5.909,      675.24,     -1.34],
    [0.0243,    0.0109,     38,     73.23,      66.89,      -22.02,     14.07,      2.69E+01,       -4.12E-02,      1.64E-04,       -9.76E-08,      3.515,      10.788,     'NA',       'NA'],
    [0.0295,    0.0077,     35,     50.17,      52.66,      53.47,      89.39,      -1.21,          7.62E-02,       -4.86E-05,      1.05E-08,       5.099,      6.436,      'NA',       'NA'],
    [0.013,     0.0114,     29,     52.82,      101.51,     31.65,      75.61,      1.18E+01,       -2.30E-02,      1.07E-04,       -6.28E-08,      7.49,       6.93,       'NA',       'NA'],
    [0.0169,    0.0074,     9,      11.74,      48.84,      123.34,     163.16,     -3.11E+01,      2.27E-01,       -3.20E-04,      1.46E-07,       4.703,      1.896,      'NA',       'NA'],
    [0.0255,    -0.0099,    'NA',   74.6,       'NA',       23.61,      'NA',       'NA',           'NA',           'NA',           'NA',           'NA',       3.335,      'NA',       'NA'],
    [0.0085,    0.0076,     34,     57.55,      68.4,       55.52,      79.93,      8.83,           -3.84E-03,      4.35E-05,       -2.60E-08,      3.649,      6.528,      'NA',       'NA'],
    ['NA',      'NA',       'NA',   83.08,      68.91,      93.7,       119.66,     5.69,           -4.12E-03,      1.28E-04,       -8.88E-08,      'NA',       12.169,     'NA',       'NA'],
    [0.0496,    -0.0101,    91,     125.66,     59.89,      88.43,      89.22,      3.65E+01,       -7.33E-02,      1.84E-04,       -1.03E-07,      2.414,      12.851,     'NA',       'NA'],
    [0.0437,    0.0064,     91,     152.54,     127.24,     -66.57,     -16.83,     2.59E+01,       -3.74E-03,      1.29E-04,       -8.88E-08,      9.679,      16.738,     'NA',       'NA'],
    [0.0031,    0.0084,     63,     63.56,      20.09,      -17.33,     -22.99,     3.53E+01,       -7.58E-02,      1.85E-04,       -1.03E-07,      2.36,       6.884,      'NA',       'NA'],
    [0.0119,    0.0049,     54,     68.78,      34.4,       41.87,      33.12,      1.96E+01,       -5.61E-03,      4.02E-05,       -2.76E-08,      4.13,       6.817,      'NA',       'NA'],
    [0.0019,    0.0051,     38,     52.1,       79.93,      39.1,       27.76,      1.67E+01,       4.81E-03,       2.77E-05,       -2.11E-08,      1.557,      5.984,      'NA',       'NA']
    ]

    # Number of functional groups
    for i in range(0, 41):
        n[i] = len(molecule.GetSubstructMatches(SMARTS[i]))

    # Calculations
    # Number of atoms
    molecule_with_Hs = Chem.AddHs(molecule)
    na = molecule_with_Hs.GetNumAtoms()

    # Molecular weight
    MW = Descriptors.MolWt(molecule)

    # Boiling point
    TB = 198.2
    for i in range(0, 41):
        if n[i] != 0:
            if p[i][3] == 'NA':
                print('Warning: There is not available parameter in boiling point calculation (Group %0.0f,'%i, group[i], ')')
            else:
                TB = TB + n[i] * p[i][3]

    # Critical temperature
    TC1 = 0; TC2 = 0
    for i in range(0, 41):
        if n[i] != 0:
            if p[i][0] == 'NA':
                print('Warning: There is not available parameter in critical temperature calculation (Group %0.0f,'%i, group[i], ')')
            else:
                TC1 = TC1 + n[i] * p[i][0]
                TC2 = TC2 + n[i] * p[i][0]
        TC = TB * (0.584 + 0.965 * TC1 - TC2**2)**-1

    # Critical pressure
    PC1 = 0
    for i in range(0, 41):
        if n[i] != 0:
            if p[i][1] == 'NA':
                print('Warning: There is not available parameter in critical pressure calculation (Group %0.0f,'%i, group[i], ')')
            else:
                PC1 = PC1 + n[i] * p[i][1]
        PC = (0.113 + 0.0032 * na - PC1)**-2
      
    # Critical volume
    VC = 17.5
    for i in range(0, 41):
        if n[i] != 0:
            if p[i][2] == 'NA':
                print('Warning: There is not available parameter in critical volume calculation (Group %0.0f,'%i, group[i], ')')
            else:
                VC = VC + n[i] * p[i][2]

    # Heat of formation (ideal gas, 298 K)
    Hform = 68.29
    for i in range(0, 41):
        if n[i] != 0:
            if p[i][5] == 'NA':
                print('Warning: There is not available parameter in heat of formation calculation (Group %0.0f,'%i, group[i], ')')
            else:
                Hform = Hform + n[i] * p[i][5]

    # Gibbs energy of formation (ideal gas, 298 K)
    Gform = 53.88
    for i in range(0, 41):
        if n[i] != 0:
            if p[i][6] == 'NA':
                print('Warning: There is not available parameter in Gibbs energy of formation calculation (Group %0.0f,'%i, group[i], ')')
            else:
                Gform = Gform + n[i] * p[i][6]

    # Ideal gas heat capacity
    a = 0; b = 0; c = 0; d = 0
    for i in range(0, 41):
        if n[i] != 0:
            if p[i][7] == 'NA':
                print('Warning: There is not available parameter in ideal gas heat capacity calculation (Group %0.0f,'%i, group[i], ')')
            else:
                a = a + n[i] * p[i][7]
                b = b + n[i] * p[i][8]
                c = c + n[i] * p[i][9]
                d = d + n[i] * p[i][10]
                
                
    
    
    # Unit conversion
    TB = TB - 273.15        # K to C
    TC = TC - 273.15        # K to C
    VC = VC / 1000          # cc/mol to cum/kmol
    Hform = Hform / 4.1868   # kJ/mol to kcal/mol
    Gform = Gform / 4.1868   # kJ/mol to kcal/mol

    # CPIG parameter estimation (
    c1 = (a - 37.93) / 4.1868          # J/mol-K to cal/mol-K
    c2 = (b + 0.21) / 4.1868           # J/mol-K to cal/mol-K
    c3 = (c - 0.000391) / 4.1868       # J/mol-K to cal/mol-K
    c4 = (d + 0.000000206) / 4.1868    # J/mol-K to cal/mol-K

    C1 = c1 + 273.15 * c2 + 273.15**2 * c3 + 273.15**3 * c4
    C2 = c2 + 2 * 273.15 * c3 + 3 * 273.15**2 * c4
    C3 = c3 + 3 * 273.15 * c4
    C4 = c4
    C5 = 0
    C6 = 0
    C7 = 6.85
    C8 = 826.85
    C9 = 8.60543
    C10 = ((C1 + C2 * C7 + C3 * C7**2 + C4 * C7**3) - C9) / 280**1.5
    C11 = 1.5

    CPIG = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11]

    return [MW, TB, TC, PC, VC, Hform, Gform, CPIG]