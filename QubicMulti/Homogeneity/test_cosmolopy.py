from pylab import *
import numpy as np
import cosmolopy
from scipy import integrate
from scipy import interpolate
from Cosmology import pyxi

mycosmo=cosmolopy.fidcosmo.copy()
mycosmo['baryonic_effects']=True

mycosmonob=cosmolopy.fidcosmo.copy()
mycosmonob['baryonic_effects']=True
mycosmonob['omega_M_0']=mycosmonob['omega_M_0']+mycosmonob['omega_b_0']
mycosmonob['omega_b_0']=0.000000001

#test
z=0.5
xi=pyxi.xith(mycosmo,z)
xinobao=pyxi.xith(cosmolopy.fidcosmo,z)
xinob=pyxi.xith(mycosmonob,z)


r=linspace(0,200,1000)
rl=logspace(-1,np.log10(200),1000)

clf()
plot(r,xi(r)*r**2)
plot(r,xinobao(r)*r**2)
plot(r,xinob(r)*r**2)

plot(rl,xi(rl)*rl**2,ls='--',lw=3)

clf()
ylim(0.9,1.1)
plot(r,xi.nr(r))
plot(rl,xi.nr(rl),ls='--',lw=3)
plot(r,r*0+1.01)


clf()
plot(r,xi.d2(r))
plot(rl,xi.d2(rl),ls='--',lw=3)
plot(r,r*0+2.97)


clf()
plot(r,xi.d2(r))
plot(r,xinobao.d2(r))
plot(r,xinob.d2(r))
plot(r,r*0+2.97)

print(xi.rh_d2())



delt=0.1
nb=10
par='omega_lambda_0'
vals=linspace(mycosmo[par]*(1-delt),mycosmo[par]*(1+delt),nb)
thed2=zeros(nb)
for i in arange(nb):
    thecosmo=mycosmo.copy()
    thecosmo[par]=vals[i]
    thexi=pyxi.xith(thecosmo,0.5)
    thed2[i]=thexi.rh_d2()

clf()
plot(vals,thed2)

##### fisher matrix calculation
pars=['h','n','omega_M_0','omega_b_0','sigma_8']
zvals=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,2.3]
zvals=[0.4,0.5,0.6,0.7,2.3]


delt=0.01
nb=3

der=zeros((size(pars),size(zvals)))
for k in arange(size(zvals)):
    print(k)
    for i in arange(size(pars)):
        thecosmo=mycosmo.copy()
        vals=linspace(mycosmo[pars[i]]*(1-delt),mycosmo[pars[i]]*(1+delt),nb)
        dvals=vals[1]-vals[0]
        thed2=zeros(nb)
        for j in arange(nb):
            thecosmo[pars[i]]=vals[j]
            thexi=pyxi.xith(thecosmo,zvals[k])
            thed2[j]=thexi.rh_d2()
        bla=np.gradient(thed2)/dvals
        der[i,k]=bla[nb/2]

fracerror=0.01
xi0=pyxi.xith(mycosmo,0.5)
d20=xi0.rh_d2()
sigma_d2=fracerror*d20

fishermat=zeros((size(pars),size(pars)))
for k in arange(size(zvals)):
    for i in arange(size(pars)):
        for j in arange(size(pars)):
            fishermat[i,j]+=der[i,k]*der[j,k]/sigma_d2**2
   

clf()
partex=['$h$','$n_s$','$\Omega_m$','$\Omega_b$','$\sigma_8$']
imshow(fishermat,interpolation='nearest')
title('Fisher Matrix')
xticks(arange(np.size(partex)),(partex))
yticks(arange(np.size(partex)),(partex))
colorbar()
savefig('fisher.png')

covpar=np.linalg.inv((fishermat))

clf()
imshow(covpar,interpolation='nearest')
title('Parameters covariance matrix')
xticks(arange(np.size(partex)),(partex))
yticks(arange(np.size(partex)),(partex))
colorbar()
savefig('covariance.png')

clf()
imshow(galtools.cov2cor(covpar),interpolation='nearest')
title('Parameters correlation matrix')
xticks(arange(np.size(partex)),(partex))
yticks(arange(np.size(partex)),(partex))
colorbar()
savefig('correlation.png')

sqrt(diagonal(covpar))


nvalues=100
h_planck=0.673
sig_h_planck=0.012
ns_planck=0.9603
sig_ns_planck=0.0073
om_planck=1.-0.685
sig_om_planck=0.018
ob_planck=0.0487
sig_ob_planck=0.0006*sqrt(3)
s8_planck=0.829
sig_s8_planck=0.012

vals=zeros((5,nvalues))
vals[0,:]=np.random.randn(nvalues)*sig_h_planck+h_planck
vals[1,:]=np.random.randn(nvalues)*sig_ns_planck+ns_planck
vals[2,:]=np.random.randn(nvalues)*sig_om_planck+om_planck
vals[3,:]=np.random.randn(nvalues)*sig_ob_planck+ob_planck
vals[4,:]=np.random.randn(nvalues)*sig_s8_planck+s8_planck


valsrh=zeros((nbins,nvalues))
d2val=zeros((nbins,nvalues))
for i in arange(nbins):
    for j in arange(nvalues):
        print(i,j)
        thecosmo=mycosmo.copy()
        for k in arange(size(pars)):
            thecosmo[pars[k]]=vals[k,j]
        
        thexi=pyxi.xith(thecosmo,zmid[i])
        d2val[i,j]=thexi.rh_d2()

d2av=zeros(nbins)
d2s=zeros(nbins)
for i in arange(nbins):
    d2av[i]=mean(d2val[i,:])
    d2s[i]=std(d2val[i,:])

clf()
errorbar(zmid,rhs,xerr=dz,yerr=drhs,fmt='ro',label='DR10 CMASS South',ms=8,elinewidth=2)
errorbar(zmid,rhn,xerr=dz,yerr=drhn,fmt='bo',label='DR10 CMASS North',ms=8,elinewidth=2)
errorbar(zmid,rha,xerr=dz,yerr=drha,fmt='ko',label='DR10 CMASS Both',ms=8,elinewidth=2)
plot(zmid,d2av,color='g',lw=2,label='Planck model')
plot(zmid,d2av+d2s,color='g',lw=2,ls=':')
plot(zmid,d2av-d2s,color='g',lw=2,ls=':')



####### Now growth factor
omega_M_0=mycosmo['omega_M_0']
zz=linspace(0,10,1000)
fg=cosmolopy.perturbation.fgrowth(zz,omega_M_0)


