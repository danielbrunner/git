# calculation of kinetic energy with the mehtod of tanimoto, 2015
# 1. calculate the rotattion rate
#2 calculate the acceleration
# 3. calculate the displacement
# calculate the integrate over the depth with the eigenmodes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




# functions #


def integraion(x,y,rho):
    inte=0
    x_d=np.diff(x)
    for ii in range(0,len(x_d)):
        inte=inte+x_d[ii]*y[ii]**2*rho[ii]
    return inte



def int_EM(disp_s_l,disp_s_z):

    # disp_s: displacement at the surface

    # this function integrate the value over the depth

    # allocate

    inte_l=np.zeros((len(disp_s_l)))
    inte_z = np.zeros((len(disp_s_z)))
    inte_r = np.zeros((len(disp_s_z)))

    #### read depth values and calculate the depth vector

    temp=pd.read_csv("./SLDER_1_Hz.TXT", skiprows=range(0, 4), nrows=183, delim_whitespace=True)
    d=np.array(temp)
    dist=np.cumsum(d[:,1])*1000     # da wir nur distance haen macht diese fkt. eine plot array [m] (darum mal 1000)
    rho=d[:,4]*1000                 # kg/m^3

    temp=pd.read_csv("./SLDER_1_Hz.TXT", skiprows=range(0, 194), delim_whitespace=True)
    prov=np.array(temp)
    UT=prov[:,1]

    temp=pd.read_csv("./SRDER_1_Hz.TXT", skiprows=range(0, 194), delim_whitespace=True)
    prov=np.array(temp)
    UR=prov[:,1]
    UZ=prov[:,3]


    for ii in range(0,len(disp_s_l)):
        temp=UT*disp_s_l[ii]
        inte_l[ii] = integraion(dist, temp, rho)
        temp=UZ*disp_s_z[ii]
        inte_z[ii] = integraion(dist, temp, rho)
        temp=UR*disp_s_z[ii]
        inte_r[ii] = integraion(dist, temp, rho)

    return inte_l,inte_z,inte_r








c_l_1=2579              #  [m/s]Love phase velocity at 5 Hz; is writen on file SLDER.TXT (atention!!!! vaules are given in km/s)
c_r_1=2349              # [m/s] Rayleigh phase velocity at 5 Hz; is written on the SRDER.TXT (atention!!!! vaules are given in km/s)
c_l_2=2525              #  [m/s]Love phase velocity at 5 Hz; is writen on file SLDER.TXT (atention!!!! vaules are given in km/s)
c_r_2=2321              # [m/s] Rayleigh phase velocity at 5 Hz; is written on the SRDER.TXT (atention!!!! vaules are given in km/s)
c_l_5=2350              #  [m/s]Love phase velocity at 5 Hz; is writen on file SLDER.TXT (atention!!!! vaules are given in km/s)
c_r_5=2162              # [m/s] Rayleigh phase velocity at 5 Hz; is written on the SRDER.TXT (atention!!!! vaules are given in km/s)
c_l_10=2233             #  [m/s]Love phase velocity at 10 Hz; is writen on file SLDER.TXT (atention!!!! vaules are given in km/s)
c_r_10=2017             # [m/s] Rayleigh phase velocity at 10 Hz; is written on the SRDER.TXT (atention!!!! vaules are given in km/s)


d=20.0                # d: distance between microarray-receivers

# gepmetrical spreading
d_r=400
r=np.sqrt(np.arange(0,24)*d_r)      # geometrical spreading
r[0]=1

#output von sofi3D sind glaub m/s --->>> achtung mit phase velocity von oben!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# ALLOCATION #############

rot_x = np.zeros((14, 24,1000))
rot_y = np.zeros((14, 24,1000))
rot_z = np.zeros((14, 24,1000))

# nean acceleration and displacement
a_m=np.zeros((10,24))
disp_m_l=np.zeros((10,24))
disp_m_r=np.zeros((10,24))



#vz=np.empty([14,24*4,1000])
for kk in xrange(4,5):

    stri='/home/djamel/PHD_projects/scatering_Paper/seismogram/model_2/seismogram_2_'+str(kk)+'_exp'+'/Model_2_'+str(kk)+'_exp'+'_vx.bin'
    vx=np.fromfile(stri, dtype='<f4')
    vx=np.reshape(vx, (14,24*4, 1000))

    stri='/home/djamel/PHD_projects/scatering_Paper/seismogram/model_2/seismogram_2_'+str(kk)+'_exp'+'/Model_2_'+str(kk)+'_exp'+'_vy.bin'
    vy=np.fromfile(stri, dtype='<f4')
    vy=np.reshape(vy, (14,24*4, 1000))

    stri='/home/djamel/PHD_projects/scatering_Paper/seismogram/model_2/seismogram_2_'+str(kk)+'_exp'+'/Model_2_'+str(kk)+'_exp'+'_vz.bin'
    vz=np.fromfile(stri, dtype='<f4')
    vz=np.reshape(vz, (14,24*4, 1000))




    # 1. rotation rate
    #atention!!! in sofi3D y axis is downpointing axis!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



    for ii in range(0, 14):
        ll = 0
        for jj in range(0, 24):
            rot_x[ii,jj,:]=1/2.0*((vz[ii,ll+2,:]-vz[ii,ll,:])/d-(vy[ii,ll+3,:]-vy[ii,ll,:])/d)
            rot_y[ii,jj,:]=1/2.0*((vx[ii,ll+3,:]-vx[ii,ll,:])/d-(vz[ii,ll+1,:]-vz[ii,ll,:])/d)
            rot_z[ii,jj,:]=1/2.0*((vy[ii,ll+1,:]-vy[ii,ll,:])/d-(vx[ii,ll+2,:]-vx[ii,ll,:])/d)
            ll=ll+4



# fft of the rotations

    n=vz[0,0,:].size
    dt = 1.000000e-02
    freq = np.fft.fftfreq(n, d=dt)[0:n/2]
    om=2*np.pi*freq                 # omega
    ROT_X=abs(np.fft.fft(rot_x, axis=2)[:,:,0:n/2])
    ROT_Y=abs(np.fft.fft(rot_y, axis=2)[:,:,0:n/2])
    ROT_Z=abs(np.fft.fft(rot_z, axis=2)[:,:,0:n/2])





# calculata the transversal (Love wave) kinetic energy
# calculate acceleration from rotation rate a=2*C*rot where C is phase velocity

    at_5=2*c_l_1*ROT_Y[:,:,10]
    disp=at_5/(om[10]**2)





    # calculate the Rayleigh wave energy
    # fft of the velocity

    VY = abs(np.fft.fft(vy, axis=2)[:, :, 0:n / 2])
    disp_y = VY[:,:,10]/om[10]

    for ii in range(0,24):
        a_m[kk-1,ii]=np.mean(at_5[:,ii])
        disp_m_l[kk-1,ii]=np.mean(disp[:,ii])
        disp_m_r[kk-1,ii]=np.mean(disp_y[:,ii])




a=np.zeros((24))        # mean acceleration of radial and over all seismogram
d_l=np.zeros((24))        # mean displacement of radial and over all seismogram
d_r=np.zeros((24))        # mean displacement of radial and over all seismogram
for ii in range(0,24):
    a[ii]=np.mean(a_m[:,ii])
    d_l[ii] = np.mean(disp_m_l[:, ii])
    d_r[ii] = np.mean(disp_m_r[:, ii])


inte_l,inte_z,inte_r=int_EM(d_l,d_r)  # parameters
E_l=inte_l*om[10]**2      # calculate the velocity from the displacement
E_z=inte_z*om[10]**2      # calculate the velocity from the displacement
E_r=inte_r*om[10]**2      # calculate the velocity from the displacement
E_ray=E_r+E_z

plt.plot(E_l/E_ray,'o')
plt.show()

# 1. Problem: ab wann nehmen wir seismogram -> first peak weg lassen???
# wir nehmen mal ganze energy -> kann aber spater geandert werden

# frequenz wechlsen
# schaue bei welcher frequenz es am meisten scattering gibt und dann berechne dort E_l/E_r

# achtung: du musst eleptisity nochmals nachprufen!!!!!