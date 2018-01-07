# calculation of kinetic energy with the mehtod of tanimoto, 2015
# 1. calculate the rotattion rate
#2 calculate the acceleration
# 3. calculate the displacement
# calculate the integrate over the depth with the eigenmodes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from _utils import *
from scipy import stats
from scipy.fftpack import fft



# global variables

U=0                                 # group velocity



# functions #


def integraion(x,y,rho):
    inte=0
    x_d=np.diff(x)
    for ii in range(0,len(x_d)):
        inte=inte+x_d[ii]*y[ii]**2*rho[ii]
    return inte



def int_EM(disp_s_l,disp_s_z,EM_freq):

    # disp_s: displacement at the surface

    # this function integrate the value over the depth

    # allocate

    inte_l=np.zeros((len(disp_s_l)))
    inte_z = np.zeros((len(disp_s_z)))
    inte_r = np.zeros((len(disp_s_z)))

    #### read depth values and calculate the depth vector

    # read depth vector
    temp=pd.read_csv("/home/djamel/PHD_projects/scatering_Paper/theo_EM_scattering/SLDER_"+str(EM_freq)+"_Hz.TXT", skiprows=range(0, 4), nrows=153, delim_whitespace=True)
    d=np.array(temp)
    dist=np.cumsum(d[:,1])*1000     # da wir nur distance haen macht diese fkt. eine plot array [m] (darum mal 1000)
    rho=d[:,4]*1000                 # kg/m^3

    # read love wave EM
    temp=pd.read_csv("/home/djamel/PHD_projects/scatering_Paper/theo_EM_scattering/SLDER_"+str(EM_freq)+"_Hz.TXT", skiprows=range(0, 164), delim_whitespace=True)
    prov=np.array(temp)
    UT=prov[:,1]

    # read rayleigh wave EM
    temp=pd.read_csv("/home/djamel/PHD_projects/scatering_Paper/theo_EM_scattering/SRDER_"+str(EM_freq)+"_Hz.TXT", skiprows=range(0, 164), delim_whitespace=True)
    prov=np.array(temp)
    UR=prov[:,1]
    UZ=prov[:,3]

    # read elipicity data
    temp=pd.read_csv("/home/djamel/PHD_projects/scatering_Paper/theo_EM_scattering/SREGN_"+str(EM_freq)+"_Hz.TXT", skiprows=range(0, 3), delim_whitespace=True)
    el=np.array(temp)[0,7]
    global U
    U=np.array(temp)[0,4]*1000                               # read group velocity of Rayleigh waves



    for ii in range(0,len(disp_s_l)):
        temp=UT*disp_s_l[ii]
        inte_l[ii] = integraion(dist, temp, rho)
        temp=UZ*disp_s_z[ii]
        inte_z[ii] = integraion(dist, temp, rho)
        temp=UR*disp_s_z[ii]*el       ###############
        inte_r[ii] = integraion(dist, temp, rho)

    return inte_l,inte_z,inte_r





EM_freq=5.1         # ACHTUNG!!!!!!!! 2.0 nicht 2 !!!!!!!!!!!!!!!!!!!!



### smfp ############

vy_smfp = np.empty([14, 24 * 4, 1000])
for kk in xrange(1, 11):

    strii='/home/djamel/PHD_projects/scatering_Paper/seismogram/model_3/seismogram_3_'+str(kk)+'_exp'+'/Model_3_'+str(kk)+'_exp'+'_vy.bin'
    dumbb=np.fromfile(strii, dtype='<f4')
    dumbb=np.reshape(dumbb,(14,24*4,1000))
    vy_smfp=vy_smfp+dumbb


vy_smfp=vy_smfp[:,::4,:]/10.0      # durch 10 um mean zu erhalten

n=vy_smfp[0, 0, :].size
dt=1.000000e-02
freq=np.fft.fftfreq(n, d=dt)[0:n/2]
om=2*np.pi*freq  # omega

VY_smfp=abs(np.fft.fft(vy_smfp, axis=2))[:,:,0:n/2]
disp_y_smfp=VY_smfp[:,:,int(EM_freq*10)]/om[int(EM_freq*10)]

d_r_smfp=np.empty((24))        # mean displacement of radial and over all seismogram
for ii in range(0,24):
    d_r_smfp[ii] = np.mean(disp_y_smfp[:, ii])

inte_smfp, inte_z_smfp, inte_r_smfp = int_EM(d_r_smfp,d_r_smfp,EM_freq)  # parameters
E_z_smfp=inte_z_smfp*om[int(EM_freq*10)]**2*U  # z energy
E_r_smfp=inte_r_smfp*om[int(EM_freq*10)]**2*U  # r energy
E_ray_smfp=E_r_smfp + E_z_smfp



for ii in range(1,25):      # geometrical spreading
    E_ray_smfp[ii-1]=E_ray_smfp[ii-1]*np.sqrt(ii*400)

# gepmetrical spreading
dr=400
r=np.arange(1,25)*dr      # distance vector

slope,intercept,r_value,p_value,std_err = stats.linregress(r, np.log(E_ray_smfp))
smfp=-1/slope

#############################



# read love wave phase velocity
temp = pd.read_csv("/home/djamel/PHD_projects/scatering_Paper/theo_EM_scattering/SLDER_" + str(EM_freq) + "_Hz.TXT",
                   skiprows=range(0, 160), nrows=1, delim_whitespace=True)
c_l = np.array(temp[[1]]*1000)





d=20.0                    # d: distance between microarray-receivers



# gepmetrical spreading
d_r=400
r=np.arange(1,25)*d_r      # distance vector


#output von sofi3D sind glaub m/s --->>> achtung mit phase velocity von oben!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# ALLOCATION #############

rot_x = np.zeros((14, 24,1000))
rot_y = np.zeros((14, 24,1000))
rot_z = np.zeros((14, 24,1000))

# nean acceleration and displacement
a_m=np.zeros((10,24))
disp_m_l=np.zeros((10,24))
disp_m_r=np.zeros((10,24))




for kk in xrange(1,11):

    stri='/home/djamel/PHD_projects/scatering_Paper/seismogram/model_3/seismogram_3_'+str(kk)+'_exp'+'/Model_3_'+str(kk)+'_exp'+'_vx.bin'
    vx=np.fromfile(stri, dtype='<f4')
    vx=np.reshape(vx, (14,24*4, 1000))

    stri='/home/djamel/PHD_projects/scatering_Paper/seismogram/model_3/seismogram_3_'+str(kk)+'_exp'+'/Model_3_'+str(kk)+'_exp'+'_vy.bin'
    vy=np.fromfile(stri, dtype='<f4')
    vy=np.reshape(vy, (14,24*4, 1000))

    stri='/home/djamel/PHD_projects/scatering_Paper/seismogram/model_3/seismogram_3_'+str(kk)+'_exp'+'/Model_3_'+str(kk)+'_exp'+'_vz.bin'
    vz=np.fromfile(stri, dtype='<f4')
    vz=np.reshape(vz, (14,24*4, 1000))




    # 1. rotation rate
    #atention!!! in sofi3D y axis is downpointing axis!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



    for ii in range(0, 14):
        ll = 0
        for jj in range(0, 24):
            rot_y[ii,jj,:]=1/2.0*((vx[ii,ll+3,:]-vx[ii,ll,:])/d-(vz[ii,ll+1,:]-vz[ii,ll,:])/d)
            ll=ll+4



# fft of the rotations

    n=vz[0,0,:].size
    dt = 1.000000e-02
    freq = np.fft.fftfreq(n, d=dt)[0:n/2]
    om=2*np.pi*freq                 # omega
    ROT_Y=abs(np.fft.fft(rot_y, axis=2)[:,:,0:n/2])





# calculata the transversal (Love wave) kinetic energy
# calculate acceleration from rotation rate a=2*C*rot where C is phase velocity

    # print some important values
    # print(c/float(ff/10)/200.0)
    # print(c/float(ff/10))
    ############

    at=2*c_l*ROT_Y[:,:,int(EM_freq*10)]
    disp=at/(om[int(EM_freq*10)]**2)






    # calculate the Rayleigh wave energy
    # fft of the velocity

    VY = abs(np.fft.fft(vy, axis=2)[:, :, 0:n / 2])
    disp_y = VY[:,::4,int(EM_freq*10)]/om[int(EM_freq*10)]



    for ii in range(0,24):
        a_m[kk-1,ii]=np.mean(at[:,ii])
        disp_m_l[kk-1,ii]=np.mean(disp[:,ii])
        disp_m_r[kk-1,ii]=np.mean(disp_y[:,ii])




a=np.zeros((24))        # mean acceleration of radial and over all seismogram
d_l=np.zeros((24))        # mean displacement of radial and over all seismogram
d_r=np.zeros((24))        # mean displacement of radial and over all seismogram
for ii in range(0,24):
    a[ii]=np.mean(a_m[:,ii])
    d_l[ii] = np.mean(disp_m_l[:, ii])
    d_r[ii] = np.mean(disp_m_r[:, ii])

inte_l,inte_z,inte_r=int_EM(d_l,d_r,EM_freq)  # parameters

E_l=inte_l*om[int(EM_freq*10)]**2      # calculate the velocity from the displacement
E_z=inte_z*om[int(EM_freq*10)]**2      # calculate the velocity from the displacement
E_r=inte_r*om[int(EM_freq*10)]**2      # calculate the velocity from the displacement
E_ray=E_r+E_z



fig, ax = plt.subplots(figsize=(20, 12))
plt.plot(r/float(smfp),E_l/E_ray)
plt.title('love: '+str(c_l_10/float(int(EM_freq*10)/10)/200.0)+'    rayleigh: '+str(c_r_10/float(int(EM_freq*10)/10)/200.0))
#plt.show()

fig.savefig('model_f_'+str(EM_freq)+'.png',format='png')      # save figure

# 1. Problem: ab wann nehmen wir seismogram -> first peak weg lassen???
# wir nehmen mal ganze energy -> kann aber spater geandert werden



#vergleiche exp mit von karman bei hoheren frequenzen ---> es sollte eigentlich eine starken unterschied geben zwischen exp und von karman weil von
#karman starkerer scatterer hat bei hoheren freqs....

#berechne nur displacenemt ratios ....


#vllt zwischenfreqs berechnen