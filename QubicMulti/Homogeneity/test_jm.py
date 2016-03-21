from pylab import *
import numpy as np
import cosmolopy
from scipy import integrate
from scipy import interpolate
from Cosmology import pyxi

mycosmo=cosmolopy.fidcosmo.copy()
mycosmo['baryonic_effects']=True
mycosmo['h']=0.7
mycosmo['sigma_8']=0.8
mycosmo['n']=0.97
mycosmo['omega_b_0']=0.0227/mycosmo['h']/mycosmo['h']
mycosmo['omega_M_0']=0.27
mycosmo['omega_lambda_0']=0.73
mycosmo['omega_k_0']=0.


#test
z=2.4
xi=pyxi.xith(mycosmo,z)

r=linspace(0,200,2001)
nr=xi.nr(r)
d2=xi.d2(r)

clf()
subplot(211)
ylim(0.98,1.05)
plot(r,nr)
plot(r,r*0+1.01,'--')
plot(r,r*0+1.0,'--')
xlabel('$r\,[h^{-1}.\mathrm{Mpc}]$')
ylabel('$N(<r)$')
subplot(212)
ylim(2.7,3.05)
plot(r,d2)
plot(r,r*0+2.97,'--')
plot(r,r*0+3,'--')
xlabel('$r\,[h^{-1}.\mathrm{Mpc}]$')
ylabel('$d_2(r)$')

f=open('file4JM.txt','w')
f.write('r[h^-1.Mpc]   N(<r)   d_2(r)')
for i in np.arange(r.size):
    f.write(str(r[i])+' '+str(nr[i])+' '+str(d2[i])+'\n')

f.close()





