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
# sql store libs
import sqlite3
import json

# local libs
import adapt_conv_sql as acsql

# global variables

U=0                                 # group velocity



# functions #

# sql libs

sqlite3.register_adapter(np.ndarray, acsql.adapt_array)
sqlite3.register_converter("MATRIX", acsql.converter_array)

#
#
# #
# # ##### sql part #############
# #
connection = sqlite3.connect("scattering.db",detect_types=sqlite3.PARSE_DECLTYPES)

cursor = connection.cursor()
#E_l_ray: E_l/E_ray
# l_lam_cl : love wavelength/correlation lengh
# r_lam_cl: rayleigh wavelength/correlation lengh

cursor.execute("CREATE TABLE if not exists model_3_exp (freq FLOAT, E_l_ray MATRIX, smfp FLOAT, l_lam_cl, r_lam_cl FLOAT"
               ", cp4_lam_cl FLOAT, cr4_lam_cl FLOAT, cp3_lam_cl FLOAT, cr3_lam_cl FLOAT, cp2_lam_cl FLOAT, cr2_lam_cl FLOAT, cp1_lam_cl FLOAT, cr1_lam_cl FLOAT)")
#####################






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
    global c_r
    c_r=np.array(temp)[0,3]*1000                               # read group velocity of Rayleigh waves


    for ii in range(0,len(disp_s_l)):
        temp=UT*disp_s_l[ii]
        inte_l[ii] = integraion(dist, temp, rho)
        temp=UZ*disp_s_z[ii]
        inte_z[ii] = integraion(dist, temp, rho)
        temp=UR*disp_s_z[ii]*el       ###############
        inte_r[ii] = integraion(dist, temp, rho)

    return inte_l,inte_z,inte_r





EM_freq=8.1        # ACHTUNG!!!!!!!! 2.0 nicht 2 !!!!!!!!!!!!!!!!!!!!



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

    at=2*c_l[0][0]*ROT_Y[:,:,int(EM_freq*10)]
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


print(str(c_l[0][0]/EM_freq/200.0))




# fig, ax = plt.subplots(figsize=(20, 12))
# plt.plot(r/float(smfp),E_l/E_ray)
# plt.title('love: '+str(c_l[0][0]/float(int(EM_freq*10)/10)/200.0)+'    rayleigh: '+str(c_r/float(int(EM_freq*10)/10)/200.0)+
#           ' s-velocity: '+str(2130/float(int(EM_freq*10)/10)/200.0)+
#           ' p-velocity: '+str(3700/float(int(EM_freq*10)/10)/200.0))
# #plt.show()
#
# #fig.savefig('model_f_'+str(EM_freq)+'.png',format='png')      # save figure
#
#
# # save data file
# print(smfp)



# body wave velocities
c_p_4=4500/EM_freq/200.0                          # lam at layer i over correlation length
c_s_4=2600/EM_freq/200.0

c_p_3=4300/EM_freq/200.0
c_s_3=2500/EM_freq/200.0

c_p_2=4000/EM_freq/200.0
c_s_2=2310/EM_freq/200.0

c_p_1=3700/EM_freq/200.0
c_s_1=2130/EM_freq/200.0



cursor.execute("INSERT INTO model_3_exp VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (EM_freq,E_l/E_ray,smfp,c_l[0][0]/EM_freq/200.0,c_r/EM_freq/200.0,
                                                                              c_p_4,c_s_4,c_p_3,c_s_3,c_p_2,c_s_2,c_p_1,c_s_1))
connection.commit()
connection.close()


#model 2 cp etc nicht speichern und von model 3 nehmen