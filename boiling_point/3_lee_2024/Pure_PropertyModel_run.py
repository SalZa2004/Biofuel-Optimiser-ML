import Pure_PropertyModel

## Inputs
## The following example is an input for GSH.

comp = 'GSH'
smiles = 'C(CC(=O)N[C@@H](CS)C(=O)NCC(=O)O)[C@@H](C(=O)O)N'
MUP = 4.767

## The inputs for other substance cases presented in the manuscript are as follows.

## Case 1 : DHMF(Dihydro-2-methyl-3-furanone)
## comp = 'DHMF'
## smiles = 'CC1C(=O)CCO1'
## MUP = 1.500


## Case 2 : FDA(2-Furaldehyde diethyl acetal)
## comp = 'FDA'
## smiles = 'CCOC(C1=CC=CO1)OCC'
## MUP = 2.313


## Case 3 : DEMB(1,1-diethoxy-3-methylbutane)
## comp = 'DEMB'
## smiles = 'CCC(C)C(OCC)OCC'
## MUP = 2.859


## Case 4 : GSH(Glutathione)
## comp = 'GSH'
## smiles = 'C(CC(=O)N[C@@H](CS)C(=O)NCC(=O)O)[C@@H](C(=O)O)N'
## MUP = 4.767


## Case 5 : VITB5(vitamin B5)
## comp = 'VITB5'
## smiles = 'CC(C)(CO)[C@H](C(=O)NCCC(=O)O)O'
## MUP = 3.425


## Case 6 : HCYS(Homocysteine)
## comp = 'HCYS'
## smiles = 'C(CS)[C@@H](C(=O)O)N'
## MUP = 3.330


## Case 7 : AH(O-Acetyl-L-homoserine)
## comp = 'AH'
## smiles = 'C(CS)[C@@H](C(=O)O)N'
## MUP = 3.790



print("Component name: " + comp)
print("SMILES: " + smiles)
print("Dipole moment, μ (debye): %0.3f" % (MUP))
print()



## Scalar properties
R = 8.314463
[MW, TB, TC, PC, VC, DHFORM, DGFORM, CPIG] = Pure_PropertyModel.JOBACK(smiles)
ZC = PC * VC / R / (TC + 273.15) * 100

print("Normal boiling temperature, TB (℃): %0.3f" % (TB))
print("Critical temperature, TC (℃): %0.3f" % (TC))
print("Critical pressure, PC (bar): %0.3f" % (PC))
print("Critical volume, VC (m3 kmol–1): %0.3f" % (VC))
print("Enthalpy of formation, DHFORM (kcal mol–1): %0.3f" % (DHFORM))
print()



## Pure component propertymodel parameters

## Extended Antoine equation parameters for vapor pressure
PLXANT = Pure_PropertyModel.PLXANT_PCES(TB, TC, PC)
print("Extended Antoine equation parameters, C_(p,n):")
for number in PLXANT:
    print(f'{number:.3f}')
print()


## Pitzer acentric factor, ω
OMEGA = Pure_PropertyModel.OMEGA_PCES(TC, PC, PLXANT)
print("Pitzer acentric factor, ω: %0.3f" % (OMEGA))
print()


## Rackett model parameter for liquid molar volume
RKTZRA = Pure_PropertyModel.RKTZRA_PCES(OMEGA)
print("Rackett model parameter , Z_RA: %0.3f" % (RKTZRA))
print()


## Watson model parameters for enthalpy of vaporization
DHVLB = Pure_PropertyModel.DHVLB_PCES(TB, TC, PC, RKTZRA, PLXANT)
DHVLWT = Pure_PropertyModel.DHVLWT_PCES(TB, TC, PC, DHVLB, RKTZRA, PLXANT)
print("Watson model parameters, C_(WT,n):")
for number in DHVLWT:
    print(f'{number:.3f}')
print()


## Polynomial model parameters for ideal gas heat capacity
## Calculated by Joback group contribution method as above
print("Ideal gas heat capacity polynomial model parameters, C_(cp,n):")
for number in CPIG:
    print(f'{number:.3f}')
print()


## Andrade model parameters for liquid viscosity
MULAND = Pure_PropertyModel.MULAND_PCES(TB, TC, PC, MW, OMEGA)
print("Andrade model parameters, C_(η,n):")
for number in MULAND:
    print(f'{number:.3f}')
print()


## DIPPR equation 100 model for liquid thermal conductivity
KLDIP = Pure_PropertyModel.KLDIP_PCES(TB, TC, MW)
print("DIPPR equation 100 model parameters, C_(λ,n):")
for number in KLDIP:
    print(f'{number:.3f}')
print()


## DIPPR equation 106 model for liquid surface tension
SIGDIP = Pure_PropertyModel.SIGDIP_PCES(TB, TC, PC)
print("DIPPR equation 106 model parameters, C_(σ,1):")
for number in SIGDIP:
    print(f'{number:.3f}')
print()


## THRSWT, TRNSWT
print("Thermodynamic property switch, THRSWT: [100, 0, 101, 0, 100, 100, 0, 104]")
print("Transport phenomenon property switch, TRNSWT: [0, 0, 100, 0, 106]")