import scipy.io
from Cosmology import cosmology
from Homogeneity import galtools
from scipy import integrate
from scipy import interpolate
from matplotlib import rc
import copy
import pyfits
import glob
import os
rc('text', usetex=False)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# define z bins
minz=0.43
maxz=0.7

################### Linear - 1 bin in z ############################################
dirbase='/Volumes/Data/SDSS/DR10/PTHaloMocks/'

# run South
rmin=0
rmax=200.
nbins=50
log=False
region='South'
outdir=dirbase+'Pairs/'
galtools.loop_mocks(dirbase,region,outdir,rmin=rmin,rmax=rmax,nbins=nbins,log=log)

# run North
rmin=0
rmax=200.
nbins=50
log=False
region='North'
outdir=dirbase+'Pairs/'
galtools.loop_mocks(dirbase,region,outdir,rmin=rmin,rmax=rmax,nbins=nbins,log=log)


#south
southfiles=glob.glob(outdir+'South/pairs_*.txt')
rs,lss,siglss,covlss,corlss=galtools.read_many_pairs(southfiles)

rs,dds,rrs,drs,ngs,nrs=galtools.read_pairs('dr10_V6_South_pairs_forxi.txt')
xilss=(dds-2*drs+rrs)/rrs

clf()
subplot(121)
errorbar(rs,lss*rs*rs,yerr=siglss*rs*rs,fmt='ro')
plot(rs,xilss*rs*rs)
subplot(122)
pcolor(rs,rs,corlss)
colorbar()

#north
northfiles=glob.glob(outdir+'North/pairs_*.txt')
rn,lsn,siglsn,covlsn,corlsn=galtools.read_many_pairs(northfiles)

rn,ddn,rrn,drn,ngn,nrn=galtools.read_pairs('dr10_V6_North_pairs_forxi.txt')
xilsn=(ddn-2*drn+rrn)/rrn

clf()
subplot(121)
errorbar(rn,lsn*rn*rn,yerr=siglsn*rn*rn,fmt='ro')
plot(rn,xilsn*rn*rn)
subplot(122)
pcolor(rn,rn,corlsn)
colorbar()

#both
nbcommon=min([size(southfiles),size(northfiles)])
r,ls,sigls,covls,corls=galtools.read_many_pairs_combine(northfiles[:nbcommon],southfiles[:nbcommon])

ra,dda,rra,dra,nga,nra=galtools.combine_regions('dr10_V6_South_pairs_forxi.txt','dr10_V6_North_pairs_forxi.txt')
xilssum=(dda-2*dra+rra)/rra

clf()
subplot(121)
errorbar(r,ls*r*r,yerr=sigls*r*r,fmt='ro')
plot(ra,xilssum*ra*ra)
subplot(122)
pcolor(r,r,corls)
colorbar()

clf()
errorbar(ra,xilssum*r*r,yerr=sigls*r*r,fmt='ro')
####################################################################################


################### Log - 1 bin in z ############################################
# south
dirbase='/Volumes/Data/SDSS/DR10/PTHaloMocks/'
rmin=1.
rmax=200.
nbins=50
log=True
region='South'
outdir=dirbase+'Pairs_Log/'
galtools.loop_mocks(dirbase,region,outdir,rmin=rmin,rmax=rmax,nbins=nbins,log=log)

#north
dirbase='/Volumes/Data/SDSS/DR10/PTHaloMocks/'
rmin=1.
rmax=200.
nbins=50
log=True
region='North'
outdir=dirbase+'Pairs_Log/'
galtools.loop_mocks(dirbase,region,outdir,rmin=rmin,rmax=rmax,nbins=nbins,log=log)

# analyse them
northfiles=glob.glob(outdir+'North/pairs_*.txt')
rn,nrn,signrn,d2n,sigd2n,covmatnrn,covmatd2n,cormatnrn,cormatd2n,rhnrn,rhd2n,sigrhnrn,sigrhd2n=galtools.homogeneity_many_pairs(northfiles,2.)

clf()
xlim(10,rmax)
ylim(0.99,1.05)
xlabel('$r [h^{-1}.\mathrm{Mpc}]$')
ylabel('scaled $N(r)$')
title('Mocks North')
plot(ones(200),ls='--',color='k')
plot(ones(200)+0.01,ls=':',color='red')
plot(zeros(10)+rhnrn,arange(10),ls=':',color='red')
errorbar(rn,nrn,yerr=signrn,fmt='ro',label='North $R_H$ = '+str('%.1f'%rhnrn)+' $\pm$ '+str('%.1f'%sigrhnrn)+' $h^{-1}.\mathrm{Mpc}$',color='b')
legend()

clf()
xlim(10,rmax)
ylim(2.8,3.01)
xlabel('$r [h^{-1}.\mathrm{Mpc}]$')
ylabel('$D_2(r)$')
title('Mocks North')
plot(zeros(200)+3,ls='--',color='k')
plot(zeros(200)+3-0.03,ls=':',color='red')
plot(zeros(10)+rhd2n,arange(10),ls=':',color='red')
errorbar(rn,d2n,yerr=sigd2n,fmt='ro',label='North $R_H$ = '+str('%.1f'%rhd2n)+' $\pm$ '+str('%.1f'%sigrhd2n)+' $h^{-1}.\mathrm{Mpc}$',color='b')
legend(loc='lower right')

southfiles=glob.glob(outdir+'South/pairs_*.txt')
rs,nrs,signrs,d2s,sigd2s,covmatnrs,covmatd2s,cormatnrs,cormatd2s,rhnrs,rhd2s,sigrhnrs,sigrhd2s=galtools.homogeneity_many_pairs(southfiles,2.)

clf()
xlim(10,rmax)
ylim(0.99,1.05)
xlabel('$r [h^{-1}.\mathrm{Mpc}]$')
ylabel('scaled $N(r)$')
title('Mocks South')
plot(ones(200),ls='--',color='k')
plot(ones(200)+0.01,ls=':',color='g')
plot(zeros(10)+rhnrs,arange(10),ls=':',color='g')
errorbar(rn,nrn,yerr=signrs,fmt='ro',label='South $R_H$ = '+str('%.1f'%rhnrs)+' $\pm$ '+str('%.1f'%sigrhnrs)+' $h^{-1}.\mathrm{Mpc}$',color='b')
legend()

clf()
xlim(10,rmax)
ylim(2.8,3.01)
xlabel('$r [h^{-1}.\mathrm{Mpc}]$')
ylabel('$D_2(r)$')
title('Mocks South')
plot(zeros(200)+3,ls='--',color='k')
plot(zeros(200)+3-0.03,ls=':',color='g')
plot(zeros(10)+rhd2s,arange(10),ls=':',color='g')
errorbar(rs,d2s,yerr=sigd2s,fmt='ro',label='South $R_H$ = '+str('%.1f'%rhd2s)+' $\pm$ '+str('%.1f'%sigrhd2s)+' $h^{-1}.\mathrm{Mpc}$',color='r')
legend(loc='lower right')




nbcommon=min([size(southfiles),size(northfiles)])
r,nr,signr,d2,sigd2,covmatnr,covmatd2,cormatnr,cormatd2,rhnr,rhd2,sigrhnr,sigrhd2=galtools.homogeneity_many_pairs_combine(northfiles[:nbcommon],southfiles[:nbcommon],2.)

clf()
xlim(1,rmax)
ylim(0.99,1.1)
xlabel('$r [h^{-1}.\mathrm{Mpc}]$')
ylabel('scaled $N(r)$')
title('Mocks')
plot(r,nr)
errorbar(r,nr,yerr=signr,fmt='ro')

clf()
xlim(1,rmax)
ylim(2.8,3.01)
xlabel('$r [h^{-1}.\mathrm{Mpc}]$')
ylabel('$D_2(r)$')
title('Mocks')
plot(r,d2)
errorbar(r,d2,yerr=sigd2,fmt='ro')



clf()
ylim(0.99,1.1)
xlim(10,rmax)
ylim(0.99,1.05)
xlabel('$r [h^{-1}.\mathrm{Mpc}]$')
ylabel('scaled $N(r)$')
title('Mocks')
plot(ones(200),ls='--',color='k')
plot(ones(200)+0.01,ls=':',color='k')
plot(zeros(10)+rhnrn,arange(10),ls=':',color='b')
plot(zeros(10)+rhnrs,arange(10),ls=':',color='r')
plot(zeros(10)+rhnr,arange(10),ls=':',color='g')
plot(rn,nrn,label='North $R_H$ = '+str('%.1f'%rhnrn)+' $\pm$ '+str('%.1f'%sigrhnrn)+' $h^{-1}.\mathrm{Mpc}$',color='b')
plot(rs,nrs,label='South $R_H$ = '+str('%.1f'%rhnrs)+' $\pm$ '+str('%.1f'%sigrhnrs)+' $h^{-1}.\mathrm{Mpc}$',color='r')
plot(rs,nrs,label='Both $R_H$ = '+str('%.1f'%rhnr)+' $\pm$ '+str('%.1f'%sigrhnr)+' $h^{-1}.\mathrm{Mpc}$',color='g')
legend()

clf()
xlim(10,rmax)
ylim(2.8,3.01)
xlabel('$r [h^{-1}.\mathrm{Mpc}]$')
ylabel('$D_2(r)$')
title('Mocks South')
plot(zeros(200)+3,ls='--',color='k')
plot(zeros(200)+3-0.03,ls=':',color='k')
plot(zeros(10)+rhd2n,arange(10),ls=':',color='b')
plot(zeros(10)+rhd2s,arange(10),ls=':',color='r')
plot(zeros(10)+rhd2,arange(10),ls=':',color='g')
plot(rs,d2n,label='North $R_H$ = '+str('%.1f'%rhd2n)+' $\pm$ '+str('%.1f'%sigrhd2n)+' $h^{-1}.\mathrm{Mpc}$',color='b')
plot(rs,d2s,label='South $R_H$ = '+str('%.1f'%rhd2s)+' $\pm$ '+str('%.1f'%sigrhd2s)+' $h^{-1}.\mathrm{Mpc}$',color='r')
plot(r,d2,label='Both $R_H$ = '+str('%.1f'%rhd2)+' $\pm$ '+str('%.1f'%sigrhd2)+' $h^{-1}.\mathrm{Mpc}$',color='g')
legend(loc='lower right')




clf()
subplot(131)
pcolor(rn,rn,cormatnrn)
subplot(132)
pcolor(rs,rs,cormatnrs)
subplot(133)
pcolor(r,r,cormatnr)

clf()
subplot(131)
#xscale('log')
#yscale('log')
pcolor(rn,rn,cormatd2n)
subplot(132)
pcolor(rs,rs,cormatd2s)
subplot(133)
pcolor(r,r,cormatd2)


####################################################################################



################### Log - 5 bins in z ############################################
import scipy.io
from Cosmology import cosmology
from Homogeneity import galtools
from scipy import integrate
from scipy import interpolate
from matplotlib import rc
import copy
import pyfits
import glob
import os
rc('text', usetex=False)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

zmin=0.43
zmax=0.7
nbins=5
zedge=linspace(zmin,zmax,nbins+1)

dirbase='/Volumes/Data/SDSS/DR10/PTHaloMocks/'
dirout='/Pairs_Log_Zbins_Test/'


subprocess.call(["mkdir",dirbase])
subprocess.call(["mkdir",dirbase+dirout])

bins=np.array(zeros(nbins),dtype='|S20')
for i in arange(nbins):
    bins[i]='z_'+str(zedge[i])+'_'+str(zedge[i+1])

for i in arange(nbins):
    subprocess.call(["mkdir",dirbase+dirout+bins[i]])


nmax=600
rmin=1.
rmax=200.
nbins=50
log=True

#South
for i in arange(nbins):
    region='South'
    outdir=dirbase+dirout+bins[i]+'/'
    galtools.loop_mocks(dirbase,region,outdir,rmin=rmin,rmax=rmax,nbins=nbins,log=log,nproc=10,zmin=zedge[i],zmax=zedge[i+1],nmax=nmax)

#north
for i in arange(nbins):
    region='North'
    outdir=dirbase+dirout+bins[i]+'/'
    galtools.loop_mocks(dirbase,region,outdir,rmin=rmin,rmax=rmax,nbins=nbins,log=log,nproc=10,zmin=zedge[i],zmax=zedge[i+1],nmax=nmax)









