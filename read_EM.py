# reads eigenmodes from COMPUTER PROGRAM IN SEISMOLOGY OUTPUT

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#aa=np.loadtxt('./SLDER.TXT',skiprows=3,nrows=3)

#### read depth values and calculate the depth vector

temp=pd.read_csv("./SLDER.TXT", skiprows=range(0, 4), nrows=121, delim_whitespace=True)
d=np.array(temp)
d=np.cumsum(d[:,1])     # da wir nur distance haen macht diese fkt. eine plot array



# read eigenmodes values
temp=pd.read_csv("./SLDER.TXT", skiprows=range(0, 132), delim_whitespace=True)
prov=np.array(temp)
UT=prov[:,1]

temp=pd.read_csv("./SRDER.TXT", skiprows=range(0, 132), delim_whitespace=True)
prov=np.array(temp)
UR=prov[:,1]
UZ=prov[:,3]





print(d.size)
print(UT.size)
plt.plot(UR,d)
plt.gca().invert_yaxis()
plt.show()


