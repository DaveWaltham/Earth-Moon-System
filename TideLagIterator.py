##############################################################
# Read a file containing k(t) (precession vs age)            #
# and calculate tidal-lag history required to explain it.    #
# Program also calculates evolution of Earth-Moon parameters #
# and associated tidal dissipation power.                    #                                                           #
# Dave Waltham, Royal Holloway, March 2024                   #
#                                                            #
# Minor tidying up                                           #
# Dave Waltham, Royal Holloway, September 2024               #
#                                                            #
# Code released under creative commons licence               #
# https://creativecommons.org/licenses/by/4.0/               #
##############################################################

# Program reads a simple, 2-column csv file containing a 
# list of ages in the 1st column and corresponding k in 2nd.
# Ages in millions of years, precession in "/y.
# First row contains text as column headings.
# Ages must be uniformly sampled in time.
# Recommended time step is 1My or smaller.
# E.g.
#
#   Age (Ma),k ("/y)
#   0,50.47387512
#   1,50.49356131
#   2,50.5132475
#   3,50.53293369
#   ...
#   2498	105.4354982
#   2499	105.4546375
#   2500	105.4737769

#Hard-coded input filename
fileIn = "kHistory.csv"

#program writes a csv file containing all results
#Hard-coded output filename
fileOut = "LagEvolution.csv"

#Remaining code is changed at your own risk :-)

#libraries
import numpy as np
import matplotlib.pyplot as plt
import math

#assumed values of key constants
ne = 1.99099e-7                 #Earth mean motion (radians/s)
omega0 = 7.29246e-5             #Earth modern siderial rotation rate (radians/s)
mm = 7.3477e22                  #Moon mass (kg)
ms = 1.9889e30                  #Sun mass (kg)
me = 5.9722e24                  #Earth mass (kg)
ae = 1.496e11                   #Earth Sun distance (m)
C = 8.0345e37                   #Earth moment of Inertia (SI)
G = 6.6743e-11                  #Gravitational constant (SI)
ee = 0.021566                   #Earth orbit mean eccentricity
tideLagM = 7055.0               #tidal lag for Moon (s)
Re = 6.3781366e6                #Earth equatorial radius (m)
Rm = 1.738e6                    #Moon equatorial radius (m)
k2 = 0.305                      #Love number for Earth
k2m = 0.0302                    #Love number for Moon
am0 = 3.844e8                   #Modern Earth Moon distance (m)
o0 = 23.270773                  #average modern obliquity (degrees)
em0 = 0.0549                    #Modern Moon orbit eccentricity
inc0 = 5.145                    #modern inclination of lunar orbit (degrees)
A = 6.1566e5                    #dynamic ellipticity constant (s^2)
yearSid = 365.256               #siderial year (days)
tideLag0 = 600.0                #Modern Earth tidal lag (s)

#consequences
mu = me*mm/(me+mm)              #reduced lunar mass
GR5k2 = G*(Re**5)*k2            #frequently used constant
y2s = yearSid*24*3600           #seconds per sidereal year

################### NdSL97 Functions #################################
#Functions to implement Earth-Moon system modelling equations
#Taken from Nero de Sugry and Laskar (1997)

#solar tide effect on Earth spin
def dLdtSolar(X,L,tidelag):
    multiplier = -3*GR5k2*tidelag*(ms**2)/(2*(ae**6))
    term1 = ( 1 + 7.5*(ee**2) )*( 1 + (X/L)**2  )*(L/C)
    term2 = -2*( 1 + 13.5*(ee**2) )*(X/L)*ne
    dLdt = multiplier*( term1 + term2 )
    return dLdt

#solar tide effect on Earth obliquity
def dXdtSolar(X,L,tidelag):
    multiplier = -3*GR5k2*tidelag*(ms**2)/(2*(ae**6))
    term1 = 2*( 1 + 7.5*(ee**2) )*(X/C)
    term2 = -2*( 1 + 13.5*(ee**2) )*ne
    dXdt = multiplier*( term1 + term2 )
    return dXdt

#lunar tide effect on Earth spin
def dLdtLunar(X,L,am,em,cosim,tidelag):
    cos2 = cosim**2
    nm = math.sqrt(G*(me+mm)/(am**3))
    multiplier = -3*GR5k2*tidelag*(mm**2)/(2*(am**6))
    term1 = (0.5*( 1 + 7.5*(em**2) )*(3-cos2+(3*cos2-1)*((X/L)**2)))*(L/C)
    term2 = -2*( 1 + 13.5*(em**2) )*(X/L)*nm*cosim
    dLdt = multiplier*( term1 + term2 )
    return dLdt

#lunar tide effect on Earth obliquity
def dXdtLunar(X,L,am,em,cosim,tidelag):
    cos2 = cosim**2
    nm = math.sqrt(G*(me+mm)/(am**3))
    multiplier = -3*GR5k2*tidelag*(mm**2)/(2*(am**6))
    term1 = ( 1 + 7.5*(em**2) )*(1+cos2)*(X/C)
    term2 = -2*( 1 + 13.5*(em**2) )*nm*cosim
    dXdt = multiplier*( term1 + term2 )
    return dXdt

#cross tide effect on Earth spin
def dLdtCross(X,L,am,em,cosim,tidelag):
    cos2 = cosim**2
    multiplier = -3*GR5k2*tidelag*(mm*ms)/(4*(am**3)*(ae**3))
    term1 = (1+1.5*(em**2))*(1+1.5*(ee**2))*(3*cos2-1)*(1-(X/L)**2)*(L/C)
    dLdt = multiplier*term1
    return dLdt

#lunar tide effect on Earth-Moon distance
def damdtLunar(X,L,am,em,cosim,tidelag):
    nm = math.sqrt(G*(me+mm)/(am**3))
    multiplier = 6*GR5k2*tidelag*(mm**2)/(mu*(am**7))
    term1 = (1+13.5*(em**2))*(X/(C*nm))*cosim
    term2 = -( 1 + 23*(em**2) )
    damdt = multiplier*( term1 + term2 )
    return damdt

#lunar tide effect on Moon eccentricity
def demdtLunar(X,L,am,em,cosim,tidelag):
    nm = math.sqrt(G*(me+mm)/(am**3))
    multiplier = 3*GR5k2*tidelag*(mm**2)*em/(mu*(am**8))
    term1 = 5.5*(X/(C*nm))*cosim
    demdt = multiplier*( term1 - 9 )
    return demdt

#lunar tide effect on Lunar orbit inclination
def dcosimdtLunar(X,L,am,em,cosim,tidelag):
    nm = math.sqrt(G*(me+mm)/(am**3))
    sin2 = 1.0 - cosim**2
    multiplier = 3*GR5k2*tidelag*(mm**2)/(2*mu*(am**8))
    term1 = (1+8*(em**2))*(X/(C*nm))*sin2
    dcosimdt = multiplier*term1
    return dcosimdt

#Earth tide (on Moon) effect on Earth-Moon distance
def damdtEarth(am,em):
    GR5k2DtM = G*(Rm**5)*k2m*tideLagM
    damdt = -57*GR5k2DtM*(me**2)*(em**2)/(mu*(am**7))
    return damdt

#Earth tide (on Moon) effect on lunar ecccentricity
def demdtEarth(am,em):
    GR5k2DtM = G*(Rm**5)*k2m*tideLagM
    demdt = -21*GR5k2DtM*(me**2)*em/(2*mu*(am**8))
    return demdt

#Solar tide (on Earth) effect on Earth-Sun distance
def daedtSolar(X,tidelag):
    msp = ms*me/(ms+me)
    multiplier = 6.0*GR5k2*(ms**2)*tidelag/(msp*ae**7)
    term1 = ( 1 + 13.5*ee**2 )*X/(C*ne)
    term2 =1 + 23*ee**2
    daedt = multiplier*( term1 - term2 )
    return daedt

#################     OTHER FUNCTIONS    #######################
#calculate tidal lag that forces calculated k to match observed dk/dt
def newDt(X,L,am,em,cosim,dkdt):

#useful constants
    B = 3.0*A*ne**2/(2.0*C)
    E = (1.0 - ee**2)**(-3.0/2.0)
    emterm = (1.0 - em**2)**(-3.0/2.0)
    D = (mm/(2*ms))*(ae**3)
    iterm = (3*cosim**2-1)
    am3 = am**3
    
#gradients if tidelag = 1.0
    dXdt1 = dXdtSolar(X,L,1.0)
    dXdt1 += dXdtLunar(X,L,am,em,cosim,1.0)
    dcosimdt1 = dcosimdtLunar(X,L,am,em,cosim,1.0)
    damdtL1 = damdtLunar(X,L,am,em,cosim,1.0)
    damdtE = damdtEarth(am,em)
    demdtL1 = demdtLunar(X,L,am,em,cosim,1.0)
    demdtE = demdtEarth(am,em)
    
#gradient pre-multipliers
    mX = B*( E + (D/am3)*emterm*iterm)
    mAm = -3*B*X*D*emterm*iterm/(am*am3)
    mEm = 3*B*X*D*iterm*em*((1-em**2)**(-5/2))/am3
    mC = 6*B*X*D*emterm*cosim/am3
    
#required tide lag
    alpha = mX*dXdt1 + mAm*damdtL1 + mEm*demdtL1 + mC*dcosimdt1
    beta = mAm*damdtE + mEm*demdtE
    DelT = -(dkdt+beta)/alpha
    return DelT

#calculate precession rate
def kCalc(A,am,Wcoso,em,cosi):
    sini2 = 1.0 - (cosi**2)
    earthTerm = ( 1.0 - ee**2 )**-1.5
    moon2sun = (mm/ms)*((ae/am)**3)*((1-em**2)**-1.5)*(1.0-1.5*sini2)
    k = 1.5*(ne**2)*A*Wcoso*(earthTerm+moon2sun)              #radians/sec
    k = k*180.0*3600*y2s/math.pi                            #"/y
    return k

#updates for all modelled parameters
def change(X,L,am,em,cosim,dAge,tidelag):
    
#gradients
    dLdt = dLdtSolar(X,L,tidelag)
    dLdt += dLdtLunar(X,L,am,em,cosim,tidelag) 
    dLdt += dLdtCross(X,L,am,em,cosim,tidelag)
    dXdt = dXdtSolar(X,L,tidelag)
    dXdt += dXdtLunar(X,L,am,em,cosim,tidelag)
    dcosimdt = dcosimdtLunar(X,L,am,em,cosim,tidelag)
    damdt = damdtLunar(X,L,am,em,cosim,tidelag)
    damdt += damdtEarth(am,em)
    demdt = demdtLunar(X,L,am,em,cosim,tidelag)
    demdt += demdtEarth(am,em)
    
#time step in sec
    seconds = -dAge*1e6*y2s             #-ve to go backwards in time
    
#hence change over time step
    DL = seconds*dLdt
    DX = seconds*dXdt
    Dcosim = seconds*dcosimdt
    Dam = seconds*damdt
    Dem = seconds*demdt
                                    
    return DL,DX,Dcosim,Dam,Dem

#Energy dissipation rate in Earth's tides
# 1. Due to lunar tides
def dissLunar(X,L,am,em,cosim,tidelag):
    damdt = damdtLunar(X,L,am,em,cosim,tidelag) #lunar recession due to tides on Earth
    nm = math.sqrt(G*(me+mm)/(am**3))
    mprime = mm*me/(mm+me)
    Omega = L/C
    D = 0.5*mprime*nm*am*(Omega-nm)*damdt
    return D

# 2. Due to Solar tides
def dissSolar(X,L,tidelag):
    daedt = daedtSolar(X,tidelag)
    msp = ms*me/(ms+me)
    Omega = L/C
    D = 0.5*msp*ne*ae*(Omega-ne)*daedt
    return D

#Simple plot of results
def plot(FigNum,Ytitle,x,y):
    plt.figure(FigNum, figsize=(10,8))
    plt.plot(x,y)
    plt.xlim(age[nAge-1],0.0)
    plt.xlabel("Age (Ma)")
    plt.ylabel(Ytitle)
    plt.grid()
    return

#################          MAIN  PROGRAM       ################################

#read k history to be reproduced
age,kIn = np.loadtxt(fileIn,skiprows=1,unpack=True,delimiter=',')
dAge = age[1]-age[0]

#initial (i.e. present day) values of everything
L = C*omega0                                    #Ang mom in Earth's spin
X = L*math.cos(math.radians(o0))                #X=Lcos(obliquity) by definition
cosim = math.cos(math.radians(inc0))            #cos(inclination)
am = am0                                        #Earth-Moon distance (m)
em = em0                                        #Lunar eccentricity
tidelag = tideLag0                              #Tide lag (s)
Wcoso = X/C                                     #Omega.cos(obliquity)
k = kCalc(A,am,Wcoso,em,cosim)                  #k recalculated from other parameters

#arrays for saving results
nAge = age.size                                 #length of all arrays
Am = np.zeros_like(age)                         #Earth-Moon distance (m)
Em = np.zeros_like(age)                         #Lunar eccentricty
Dt = np.zeros_like(age)                         #tidal lag (s)
Ci = np.zeros_like(age)                         #cosine lunar inclination
o = np.zeros_like(age)                          #obliquity (degrees)
LOD = np.zeros_like(age)                        #Length of day (hours)
kOut = np.zeros_like(age)                       #output precession rate ("/y)
tPowL = np.zeros_like(age)                      #lunar tidal dissipation (W)
tPowS = np.zeros_like(age)                      #solar tidal dissipation (W)
    
#save initial values for later plotting
Am[0] = am
Em[0] = em
Dt[0] = tidelag
Ci[0] = cosim 
o[0] = math.degrees(math.acos(X/L))
omega = L/C                             #Earth sidereal rotation rate (rad/s)
omegaSol = omega - ne                   #correction for solar rather than siderial day
LODs = 2*math.pi/omegaSol               #Length of day in seconds
LOD[0] = LODs / 3600.0                  #convert to hours
kOut[0] = k
tPowL[0] = dissLunar(X,L,am,em,cosim,tidelag)
tPowS[0] = dissSolar(X,L,tidelag)

#loop backwards in time from present day
for iage in range(nAge-1):

#dk/dt, from input file, in SI units
    dkdt = ( kIn[iage+1] - kIn[iage] ) / dAge
    dkdt *= math.pi / ( 3600.0*180 )                #rad/y My
    dkdt /= y2s*1e6*y2s                             #rad/s2

#quick dirty first guess at updated parameters 
#(centrally differenced in space but backward differenced in time)
    DelT = newDt(X,L,am,em,cosim,dkdt)
    DL,DX,Dcosim,Dam,Dem = change(X,L,am,em,cosim,dAge,tidelag)

#iterate to refine these using mid-point values, i.e. central differencing in both time and space
    for iter in range(10):              #10 iterations generally gives good convergence
        
#latest mid-point estimates
        lagM = 0.5*(tidelag + DelT)
        XM = X + 0.5*DX
        LM = L + 0.5*DL
        amM = am + 0.5*Dam
        emM = em + 0.5*Dem
        cosiM = cosim + 0.5*Dcosim
        
#use mid-point estimates to calculate changes
        tidelagNew = newDt(XM,LM,amM,emM,cosiM,dkdt)
        DL,DX,Dcosim,Dam,Dem = change(XM,LM,amM,emM,cosiM,dAge,lagM)
    
#update values for next time round time loop
    tidelag = tidelagNew
    L += DL
    X += DX
    am += Dam
    em += Dem
    cosim += Dcosim
    Wcoso = X/C
    k = kCalc(A,am,Wcoso,em,cosim)
    
#add to arrays for later plotting and output
    Am[iage+1] = am
    Em[iage+1] = em
    Dt[iage+1] = tidelag
    Ci[iage+1] = cosim
    o[iage+1] = math.degrees(math.acos(X/L))
    omega = L/C        
    omegaSol = omega - ne        
    LODs = 2*math.pi/omegaSol           
    LOD[iage+1] = LODs / 3600.0
    kOut[iage+1] = k              
    tPowL[iage+1] = dissLunar(X,L,am,em,cosim,tidelag)
    tPowS[iage+1] = dissSolar(X,L,tidelag)
 
#Once finished, do lots of plots
plot(0,"Input Precession Rate (sec/y)",age,kIn)
plot(1,"Tidal Lag (s)",age[1:]-0.5*dAge,Dt[1:])
plot(2,"Earth-Moon Distance (1000 km)",age,Am/1e6)
plot(3,"Moon Orbit Eccentricity",age,Em)
plot(4,"Moon Orbit Inclintion (degrees)",age,np.degrees(np.arccos(Ci)))
plot(5,"Earth Obliquity (degrees)",age,o)
plot(6,"Day Length (hours)",age,LOD)
plot(7,"Lunar Tidal Dissipation (TW)",age,tPowL/1e12)
plot(8,"Solar Tidal Dissipation (TW)",age,tPowS/1e12)
plot(9,"Model Error (%)",age,100*(kOut-kIn)/kIn)

#also output to a csv file
midAge = np.zeros_like(age)
midAge[1:] = 0.5*( age[0:nAge-1] + age[1:])
output = np.zeros([nAge,12])
output[:,0] = age
output[:,1] = kIn
output[:,2] = Am/1e6
output[:,3] = Em
output[:,4] = np.degrees(np.arccos(Ci))
output[:,5] = o
output[:,6] = LOD
output[:,7] = tPowL/1e12
output[:,8] = tPowS/1e12
output[:,9] = kOut
output[:,10] = midAge
output[:,11] = Dt
header = "Age (Ma),Input k (sec/y),Earth-Moon Distance (1000km)"
header += ",Lunar eccentricity,Lunar inclination (degrees),obliquity (degrees)"
header += ",LOD (hours), Lunar Tidal Power (TW), Solar Tidal Power (TW)"
header += ",Output k (sec/y),Midpoint (Ma),Tide Lag (s)"
np.savetxt(fileOut,output,delimiter=",",header=header)