import numpy as np

import sqlite3
import matplotlib.pyplot as plt


#local libs
import adapt_conv_sql as acsql


#format_paper()                  # format for plots

sqlite3.register_adapter(np.ndarray, acsql.adapt_array)
sqlite3.register_converter("MATRIX", acsql.converter_array)




connection = sqlite3.connect("scattering.db",detect_types=sqlite3.PARSE_DECLTYPES)
#
cursor = connection.cursor()


cursor.execute("SELECT * FROM model_3_exp")
data = cursor.fetchall()
connection.commit()
connection.close()

ma_E_lr=[]
smfp=[]
freq=[]
lam=[]
for ii in range(0,51):
    smfp=np.append(smfp,data[ii][2])
    ma_E_lr=np.append(ma_E_lr,max(data[ii][1]))
    freq=np.append(freq,data[ii][0])
    lam = np.append(lam, data[ii][4])
#plt.plot(lam,smfp)
plt.plot(smfp[10:51],ma_E_lr[10:51])


plt.show()
