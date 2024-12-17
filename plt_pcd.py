import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
sns.set_style("whitegrid")
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np

## ising learning

dula = np.load('./figs/ising_learn/rmse_dula_0.25_100.npy')
egdula = np.load('./figs/ising_learn/rmse_egdula_0.25_100.npy')

dmala = np.load('./figs/ising_learn/rmse_dmala_0.25_100.npy')
egdmala = np.load('./figs/ising_learn/rmse_egdmala_0.25_100.npy')




x= range(len(dula))
x=[t*10000 for t in x]
"""
plt.plot(x,gibbs,lw=2,label='Gibbs-1')
plt.plot(x,gwg,lw=2,label='GWG-1')
plt.plot(x,dmala,lw=2,label='DMALA')
"""
plt.plot(x,np.log(dula),lw=2,label='dula')
plt.plot(x,np.log(egdula),lw=2,label='egdula')


plt.plot(x,np.log(dmala),lw=2,label='dmala')
plt.plot(x,np.log(egdmala),lw=2,label='egdmala')
#plt.plot(x,edmala_glu,lw=2,label='edmala-glu')
plt.xlabel('Iters ',fontsize=10)
plt.legend(fontsize=10)
plt.ylabel('log RMSE',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(left=0)
plt.title('logRMSE vs Iterations',fontsize=10)
plt.savefig('figs/ising_learn/logrmse_.25.pdf')
plt.close()

#time
plt.clf()
"""
gibbst = np.load('./figs/ising_learn/times_gibbs_0.25_100.npy')
gwgt = np.load('./figs/ising_learn/times_gwg_0.25_100.npy')
dmalat = np.load('./figs/ising_learn/times_dmala_0.25_100.npy')

egdulat = np.load('./figs/ising_learn/times_dula_0.25_100.npy')
edulat = np.load('./figs/ising_learn/times_edula_0.25_100.npy')
edula_glut = np.load('./figs/ising_learn/times_edula-glu_0.25_100.npy')

dmalat = np.load('./figs/ising_learn/times_dmala_0.25_100.npy')
edmalat = np.load('./figs/ising_learn/times_edmala_0.25_100.npy')
edmala_glut = np.load('./figs/ising_learn/times_edmala-glu_0.25_100.npy')

plt.plot(gibbst,gibbs,lw=2,label='Gibbs-1')
plt.plot(gwgt,gwg,lw=2,label='GWG-1')
plt.plot(dmalat,dmala,lw=2,label='DMALA')

plt.plot(dulat,dula,lw=2,label='dula')
plt.plot(edulat,edula,lw=2,label='edula')
plt.plot(edula_glut,edula_glu,lw=2,label='edula-glu')

plt.plot(dmalat,dmala,lw=2,label='dmala')
plt.plot(edmalat,edmala,lw=2,label='edmala')
#plt.plot(edmala_glut,edmala_glu,lw=2,label='edmala-glu')

plt.xlabel('Runtime (s)',fontsize=10)
plt.legend(fontsize=10)
plt.ylabel('log RMSE',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title('logRMSE vs Runtime',fontsize=10)
plt.savefig('figs/ising_learn/time_logrmse_.25.pdf')
plt.close()
"""