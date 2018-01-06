# scattering coeffitient

# calculation of kinetic energy flux with the mehtod of madea, 2008
# 1. calculate the mean of all traces for all models and all phis to get only the coherent phase
# 2. calculate the FFT
# 3. calculate the energy density flux of Rayleigh waves using formula 31

# calculate the integrate over the depth with the eigenmodes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from _utils import *




# functions #


def integraion(x,y,rho):
    inte=0
    x_d=np.diff(x)
    for ii in range(0,len(x_d)):
        inte=inte+x_d[ii]*y[ii]**2*rho[ii]
    return inte



def int_EM(disp_s_z):

    # disp_s: displacement at the surface

    # this function integrate the value over the depth

    # allocate

    inte_z = np.zeros((len(disp_s_z)))
    inte_r = np.zeros((len(disp_s_z)))

    #### read depth values and calculate the depth vector

    temp=pd.read_csv("./SLDER_3_Hz.TXT", skiprows=range(0, 4), nrows=153, delim_whitespace=True)
    d=np.array(temp)
    dist=np.cumsum(d[:,1])*1000     # da wir nur distance haen macht diese fkt. eine plot array [m] (darum mal 1000)
    rho=d[:,4]*1000                 # kg/m^3


    temp=pd.read_csv("./SRDER_3_Hz.TXT", skiprows=range(0, 164), delim_whitespace=True)
    prov=np.array(temp)
    UR=prov[:,1]
    print(UR[110])
    UZ=prov[:,3]


    for ii in range(0,len(disp_s_z)):
        temp=UZ*disp_s_z[ii]
        inte_z[ii] = integraion(dist, temp, rho)
        temp=UR*disp_s_z[ii]*el_10       ###############
        inte_r[ii] = integraion(dist, temp, rho)

    return inte_z,inte_r










ff=3                       # frequency ##############
c=U_r_3                    # phase velo #####################
ff=ff*10



# gepmetrical spreading
d_r=400
r=np.arange(1,25)*d_r      # distance vector


#output von sofi3D sind glaub m/s --->>> achtung mit phase velocity von oben!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# ALLOCATION #############



# nean acceleration and displacement
disp_m_r=np.zeros((10,24))






for kk in xrange(1, 11):
    vy = np.empty([14, 24 * 4, 1000])
    stri=stri='/home/djamel/PHD_projects/scatering_Paper/seismogram/model_3/seismogram_3_'+str(kk)+'_exp'+'/Model_3_'+str(kk)+'_exp'+'_vy.bin'
    dumb=np.fromfile(stri, dtype='<f4')
    dumb=np.reshape(dumb,(14,24*4,1000))
    vy=vy+dumb

vy=vy[:,::4,:]/10.0      # durch 10 um mean zu erhalten
n=vy[0, 0, :].size
dt=1.000000e-02
freq=np.fft.fftfreq(n, d=dt)[0:n/2]
om=2*np.pi*freq  # omega
VY=abs(np.fft.fft(vy, axis=2)[:,:,0:n/2])
disp_y=VY[:,:,ff]/om[ff]








# calculata the transversal (Love wave) kinetic energy
# calculate acceleration from rotation rate a=2*C*rot where C is phase velocity

# print some important values
print(c/float(ff/10)/200.0)
print(c/float(ff/10))
############


d_r=np.zeros((24))        # mean displacement of radial and over all seismogram
for ii in range(0,24):
    d_r[ii] = np.mean(disp_y[:, ii])

#
inte_z,inte_r=int_EM(d_r)  # parameters
E_z=inte_z*om[ff]**2*c      # calculate the velocity from the displacement
E_r=inte_r*om[ff]**2*c      # calculate the velocity from the displacement
E_ray=E_r+E_z

for ii in range(1,25):      # geometrical spreading
    E_ray[ii-1]=E_ray[ii-1]*np.sqrt(ii*400)

#
# plt.plot(E_ray)
# plt.show()



slope, intercept, r_value, p_value, std_err = stats.linregress(r,np.log(E_ray))
print(-1/slope)
