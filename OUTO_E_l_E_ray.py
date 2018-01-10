import numpy as np

import sqlite3
import matplotlib.pyplot as plt


#local libs
import adapt_conv_sql as acsql
from _utils import format_paper

format_paper()


# adapter converter
sqlite3.register_adapter(np.ndarray, acsql.adapt_array)
sqlite3.register_converter("MATRIX", acsql.converter_array)
####



connection = sqlite3.connect("scattering.db",detect_types=sqlite3.PARSE_DECLTYPES)
#
cursor = connection.cursor()

cursor.execute("SELECT * FROM model_3_exp")
data_3_exp = cursor.fetchall()
connection.commit()

cursor.execute("SELECT * FROM model_2_exp")
data_2_exp = cursor.fetchall()
connection.commit()

cursor.execute("SELECT * FROM model_2")
data_2_vK = cursor.fetchall()
connection.commit()


cursor.execute("SELECT * FROM model_2_NL")
data_2_NL = cursor.fetchall()
connection.commit()
connection.close()

ma_E_lr_2=[]
ma_E_lr_vK=[]
ma_E_lr_NL=[]
ma_E_lr_3=[]


smfp_2=[]
smfp_vK=[]
smfp_NL=[]
smfp_3=[]


freq=[]
lam_r=[]
lam_l=[]
cp_4=[]
cs_4=[]
cp_1=[]
cs_1=[]
for ii in range(0,90):
    smfp_2=np.append(smfp_2,data_2_exp[ii][2])
    smfp_3=np.append(smfp_3,data_3_exp[ii][2])
    smfp_vK = np.append(smfp_vK, data_2_vK[ii][2])
    smfp_NL= np.append(smfp_NL, data_2_NL[ii][2])

    ma_E_lr_2=np.append(ma_E_lr_2,max(data_2_exp[ii][1]))
    ma_E_lr_3=np.append(ma_E_lr_3,max(data_3_exp[ii][1]))
    ma_E_lr_vK = np.append(ma_E_lr_vK, max(data_2_vK[ii][1]))
    ma_E_lr_NL = np.append(ma_E_lr_NL, max(data_2_NL[ii][1]))





    freq=np.append(freq,data_2_exp[ii][0])
    lam_l = np.append(lam_l, data_2_exp[ii][3])
    lam_r = np.append(lam_r, data_2_exp[ii][4])
    #
    # cp_4 = np.append(cp_4, data[ii][5])
    # cs_4 = np.append(cs_4, data[ii][6])
    # cp_1 = np.append(cp_1, data[ii][11])
    # cs_1 = np.append(cs_1, data[ii][12])



fig = plt.figure()
ax=fig.add_subplot(211)
line1=ax.plot(lam_r[20::],ma_E_lr_2[20::],'or', label = '$\sigma=15\%$, exponential CF')
plt.hold(True)
line2=plt.plot(lam_r[20::],ma_E_lr_3[20::],'og', label = '$\sigma=20\%$, exponential CF')
plt.hold(True)
line3=plt.plot(lam_r[20::],ma_E_lr_vK[20::],'ob', label = '$\sigma=15\%$, von Karman CF')
plt.hold(True)
line4=plt.plot(lam_r[20::],ma_E_lr_NL[20::],'ok', label = '$\sigma=15\%$, exponential CF, no layer')



# xlabel
plt.xlabel('$\lambda_{norm}$', fontsize=22)
plt.ylabel('$E_{L}/E_{R}$', fontsize=22)



ax=fig.add_subplot(212)
line1=ax.plot(lam_r[20::],smfp_2[20::],'or', label = '$\sigma=15\%$, exponential CF')
plt.hold(True)
line2=plt.plot(lam_r[20::],smfp_3[20::],'og', label = '$\sigma=20\%$, exponential CF')
plt.hold(True)
line3=plt.plot(lam_r[20::],smfp_vK[20::],'ob', label = '$\sigma=15\%$, von Karman CF')
plt.hold(True)
line4=plt.plot(lam_r[20::],smfp_NL[20::],'ok', label = '$\sigma=15\%$, exponential CF, no layer')


# legend
lns = line1+line2+line3+line4
labs = [l.get_label() for l in lns]
ax.legend(lns,labs, fontsize=14, loc=4)

# xlabel
plt.xlabel('$\lambda_{norm}$', fontsize=22)
plt.ylabel('$smfp$', fontsize=22)




# fig.add_subplot(423)
# plt.plot(lam_r[20::2],ma_E_lr[20::2],'o')
#
# fig.add_subplot(424)
# plt.plot(lam_l[20::2],ma_E_lr[20::2],'o')
#
#
# fig.add_subplot(425)
# plt.plot(cp_4[20::2],ma_E_lr[20::2],'o')
#
# fig.add_subplot(426)
# plt.plot(cs_4[20::2],ma_E_lr[20::2],'o')
#
# fig.add_subplot(427)
# plt.plot(cp_1[20::2],ma_E_lr[20::2],'o')
#
# fig.add_subplot(428)
# plt.plot(cs_1[20::2],ma_E_lr[20::2],'o')
###############

#fig.add_subplot(211)
#plt.plot(freq[20::2],smfp[20::2],'o')


# fig.add_subplot(212)
# plt.plot(freq[20::2],cp_4[20::2],'or')
#
# plt.hold(True)
# plt.plot(freq[20::2],cs_4[20::2],'or')
#
# plt.hold(True)
# plt.plot(freq[20::2],cp_1[20::2],'or')
#
# plt.hold(True)
# plt.plot(freq[20::2],cs_1[20::2],'or')
#
# plt.hold(True)
# plt.plot(freq[20::2],lam_l[20::2],'ok')
#
# plt.hold(True)
# plt.plot(freq[20::2],lam_r[20::2],'ok')



plt.show()



# love geht viel weiter runter also rayleigh waves ->

# the higher the frequency the lower the smfp
