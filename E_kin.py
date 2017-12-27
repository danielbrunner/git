# calculation of kinetic energy with the mehtod of tanimoto, 2015
# 1. calculate the rotattion rate
#2 calculate the acceleration
# 3. calculate the displacement
# calculate the integrate over the depth with the eigenmodes

import numpy as np
import matplotlib.pyplot as plt

# parameters

c_l=2350              # Love phase velocity at 5 Hz; is writen on file SLDER.TXT (atention!!!! vaules are given in km/s)
c_r=2162              # Rayleigh phase velocity at 5 Hz; is written on the SRDER.TXT (atention!!!! vaules are given in km/s)


d=20.0                # d: distance between microarray-receivers

# gepmetrical spreading
d_r=400
r=np.sqrt(np.arange(0,24)*d_r)      # geometrical spreading
r[0]=1
print(r)
#output von sofi3D sind glaub m/s --->>> achtung mit phase velocity von oben!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# ALLOCATION #############

rot_x = np.zeros((14, 24,1000))
rot_y = np.zeros((14, 24,1000))
rot_z = np.zeros((14, 24,1000))

# nean acceleration and displacement
a_m=np.zeros((10,24))
disp_m=np.zeros((10,24))




#vz=np.empty([14,24*4,1000])
for kk in xrange(1,11):

    stri='/home/djamel/PHD_projects/scatering_Paper/seismogram/model_2/seismogram_2_'+str(kk)+'/Model_2_'+str(kk)+'_vx.bin'
    vx=np.fromfile(stri, dtype='<f4')
    vx=np.reshape(vx, (14,24*4, 1000))

    stri='/home/djamel/PHD_projects/scatering_Paper/seismogram/model_2/seismogram_2_'+str(kk)+'/Model_2_'+str(kk)+'_vy.bin'
    vy=np.fromfile(stri, dtype='<f4')
    vy=np.reshape(vy, (14,24*4, 1000))

    stri='/home/djamel/PHD_projects/scatering_Paper/seismogram/model_2/seismogram_2_'+str(kk)+'/Model_2_'+str(kk)+'_vz.bin'
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


# calculate acceleration from rotation rate a=2*C*rot where C is phase velocity

    at_5=2*c_l*ROT_Y[:,:,50]
    disp=at_5/(om[50]**2)



    for ii in range(0,24):
        a_m[kk-1,ii]=np.mean(at_5[:,ii])*r[ii]
        disp_m[kk-1,ii]=np.mean(disp[:,ii])*r[ii]


aa=np.zeros((24))
dd=np.zeros((24))
for ii in range(0,24):
    aa[ii]=np.mean(a_m[:,ii])
    dd[ii] = np.mean(disp_m[:, ii])

plt.plot(dd,'o')
plt.show()

# bei 100 Hz ist steigung maxial
# displacement und acc mit geosread normalisiert