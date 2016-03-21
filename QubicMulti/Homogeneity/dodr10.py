import scipy.io
from Cosmology import cosmology
from Homogeneity import galtools
from Homogeneity import fitting
from scipy import integrate
from scipy import interpolate
from matplotlib import rc
import copy
import glob
import pyfits
rc('text', usetex=False)
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import mpfit
import iminuit

# define z bins
minz=0.43
maxz=0.7
nbins=5
zedge=linspace(minz,maxz,nbins+1)





# define fiducial cosmology
cosmo=[0.27,0.73,-1,0,0.7]



#dirbase='/Users/hamilton/SDSS/Data/DR10/LRG/'
#dirbase='/Volumes/Data/SDSS/'
dirbase='/Volumes/Data/SDSS/DR10/LRG/'

fdnorth='CMASS-DR10_v6-N-Anderson.fits'
fdsouth='CMASS-DR10_v6-S-Anderson.fits'
frnorth='CMASS-DR10_v6-N-Anderson.ran.fits'
frsouth='CMASS-DR10_v6-S-Anderson.ran.fits'

ratiorandom=5.

#### South
data,hdr_data=pyfits.getdata(dirbase+fdsouth,header=True)
wok=where(data.field('z')>minz)
data=data[wok]
wok=where(data.field('z')<maxz)
data=data[wok]

dataraS=data.field('RA')
datadecS=data.field('DEC')
datazS=data.field('z')
data_wfkpS=data.field('WEIGHT_FKP')
data_wcpS=data.field('WEIGHT_CP')
data_wnozS=data.field('WEIGHT_NOZ')
data_wstarS=data.field('WEIGHT_STAR')
data_wS=data_wfkpS*data_wstarS*(data_wnozS+data_wcpS-1)


rnd,hdr_data=pyfits.getdata(dirbase+frsouth,header=True)
wok=where(rnd.field('z')>minz)
rnd=rnd[wok]
wok=where(rnd.field('z')<maxz)
rnd=rnd[wok]
rnd=rnd[0:size(dataraS)*ratiorandom]

randomraS=rnd.field('RA')
randomdecS=rnd.field('DEC')
randomzS=rnd.field('z')
random_wfkpS=rnd.field('WEIGHT_FKP')
random_wS=random_wfkpS





######### North
data,hdr_data=pyfits.getdata(dirbase+fdnorth,header=True)
wok=where(data.field('z')>minz)
data=data[wok]
wok=where(data.field('z')<maxz)
data=data[wok]

dataraN=data.field('RA')
datadecN=data.field('DEC')
datazN=data.field('z')
data_wfkpN=data.field('WEIGHT_FKP')
data_wcpN=data.field('WEIGHT_CP')
data_wnozN=data.field('WEIGHT_NOZ')
data_wstarN=data.field('WEIGHT_STAR')
data_wN=data_wfkpN*data_wstarN*(data_wnozN+data_wcpN-1)


rnd,hdr_data=pyfits.getdata(dirbase+frnorth,header=True)

wok=where(rnd.field('z')>minz)
rnd=rnd[wok]
wok=where(rnd.field('z')<maxz)
rnd=rnd[wok]
rnd=rnd[0:size(dataraN)*ratiorandom]

randomraN=rnd.field('RA')
randomdecN=rnd.field('DEC')
randomzN=rnd.field('z')
random_wfkpN=rnd.field('WEIGHT_FKP')
random_wN=random_wfkpN




##### Homogeneity
# define r bins
#rmin=1.
#rmax=200.
#nbins=50

#r,dd,rr,dr=galtools.paircount_data_random(dataraS,datadecS,datazS,randomraS,randomdecS,randomzS,cosmo,rmin,rmax,nbins,log=1,file='dr10South_pairs.txt',nproc=12)

#r,dd,rr,dr=galtools.paircount_data_random(dataraN,datadecN,datazN,randomraN,randomdecN,randomzN,cosmo,rmin,rmax,nbins,log=1,file='dr10North_pairs.txt',nproc=12)


#weighted
#r,dd,rr,dr=galtools.paircount_data_random(dataraS,datadecS,datazS,randomraS,randomdecS,randomzS,cosmo,rmin,rmax,nbins,log=1,file='dr10South_pairs_weighted.txt',nproc=2,wdata=data_wS,wrandom=random_wS)

#r,dd,rr,dr=galtools.paircount_data_random(dataraN,datadecN,datazN,randomraN,randomdecN,randomzN,cosmo,rmin,rmax,nbins,log=1,file='dr10North_pairs_weighted.txt',nproc=14,wdata=data_wN,wrandom=random_wN)



#### z bins
# define r bins
rmin=1.
rmax=200.
nbinsr=50

bins=np.array(zeros(nbins),dtype='|S20')
for i in arange(nbins):
    bins[i]='z_'+str(zedge[i])+'_'+str(zedge[i+1])

# South
for i in arange(np.size(zedge)-1):
    print(' ')
    print(bins[i])
    mini=zedge[i]
    maxi=zedge[i+1]
    wdata=where((datazS >= mini) & (datazS <= maxi))
    wrandom=where((randomzS >= mini) & (randomzS <= maxi))
    r,dd,rr,dr=galtools.paircount_data_random(dataraS[wdata],datadecS[wdata],datazS[wdata],randomraS[wrandom],randomdecS[wrandom],randomzS[wrandom],cosmo,rmin,rmax,nbinsr,log=1,file='dr10South_'+bins[i]+'_pairs_weighted.txt',nproc=8,wdata=data_wS,wrandom=random_wS)



# North
for i in arange(np.size(zedge)-1):
    print(' ')
    print(bins[i])
    mini=zedge[i]
    maxi=zedge[i+1]
    wdata=where((datazN >= mini) & (datazN <= maxi))
    wrandom=where((randomzN >= mini) & (randomzN <= maxi))
    r,dd,rr,dr=galtools.paircount_data_random(dataraN[wdata],datadecN[wdata],datazN[wdata],randomraN[wrandom],randomdecN[wrandom],randomzN[wrandom],cosmo,rmin,rmax,nbinsr,log=1,file='dr10North_'+bins[i]+'_pairs_weighted.txt',nproc=16,wdata=data_wS,wrandom=random_wS)






#### Analyse Data and Mocks
#datafileNorth='/Users/hamilton/SDSS/Homogeneity/dr10North_pairs_weighted.txt'
#datafileSouth='/Users/hamilton/SDSS/Homogeneity/dr10South_pairs_weighted.txt'
#mockdirNorth='/Users/hamilton/SDSS/DR10/PTHaloMocks/Pairs_Log/North/'
#mockdirSouth='/Users/hamilton/SDSS/DR10/PTHaloMocks/Pairs_Log/South/'

#rhs,drhs=galtools.getd2_datamocks(datafileSouth,mockdirSouth,nbspl=8)
#rhn,drhn=galtools.getd2_datamocks(datafileNorth,mockdirNorth,nbspl=12)
#rha,drha=galtools.getd2_datamocks([datafileNorth,datafileSouth],[mockdirNorth,mockdirSouth],combine=True,nbspl=12)





#################### with z bins ###############################
# define z bins
minz=0.43
maxz=0.7
nbins=5
zedge=linspace(minz,maxz,nbins+1)
bins=np.array(zeros(nbins),dtype='|S20')
for i in arange(nbins):
    bins[i]='z_'+str(zedge[i])+'_'+str(zedge[i+1])

dz=zeros(nbins)+(zedge[1]-zedge[0])/2
zmid=(zedge[arange(nbins)]+zedge[arange(nbins)+1])/2

#growth factor
import cosmolopy
fg=cosmolopy.perturbation.fgrowth(zmid,0.27)

#bias : from an email by Eisenstein, forwarded by Jean-Marc on 03/04/2013
thebias=1.7/fg§


datafileNorth=[]
datafileSouth=[]
mockdirNorth=[]
mockdirSouth=[]
for i in bins:
    datafileNorth.append('/Users/hamilton/SDSS/Homogeneity/dr10North_'+np.str(i)+'_pairs_weighted.txt')
    datafileSouth.append('/Users/hamilton/SDSS/Homogeneity/dr10South_'+np.str(i)+'_pairs_weighted.txt')
    mockdirNorth.append('/Volumes/Data/SDSS/DR10/PTHaloMocks/Pairs_Log_Zbins/'+np.str(i)+'/North/')
    mockdirSouth.append('/Volumes/Data/SDSS/DR10/PTHaloMocks/Pairs_Log_Zbins/'+np.str(i)+'/South/')


rhs=np.zeros(nbins)
drhs=np.zeros(nbins)
rhn=np.zeros(nbins)
drhn=np.zeros(nbins)
rha=np.zeros(nbins)
drha=np.zeros(nbins)
mockrhs=np.zeros(nbins)
nbspl=6
for i in arange(nbins):
    covmatd2S,meand2S,all_nrS,all_d2S,r=galtools.read_mocks(mockdirSouth[i],bias=thebias[i])
    rhs[i],drhs[i],toto,rd,d2_rS=galtools.getd2_datamocks(datafileSouth[i],covmatd2S,nbspl=nbspl,bias=thebias[i])
    nm,nb=shape(all_d2S)
    for j in arange(nm):
        plot(r,all_d2S[j,:],color='green',alpha=0.01)
    title('South - '+bins[i])
    draw()
    savefig('Result_South_'+bins[i]+'.png',dpi=300)
    covmatd2N,meand2N,all_nrN,all_d2N,r=galtools.read_mocks(mockdirNorth[i],bias=thebias[i])
    rhn[i],drhn[i],toto,rd,d2_rN=galtools.getd2_datamocks(datafileNorth[i],covmatd2N,nbspl=nbspl,bias=thebias[i])
    nm,nb=shape(all_d2N)
    for j in arange(nm):
        plot(r,all_d2N[j,:],color='green',alpha=0.01)
    title('North - '+bins[i])
    draw()
    savefig('Result_North_'+bins[i]+'.png',dpi=300)
    covmatd2A,meand2A,all_nrA,all_d2A,r=galtools.read_mocks([mockdirNorth[i],mockdirSouth[i]],combine=True,bias=thebias[i])
    rha[i],drha[i],toto,rd,d2_rA=galtools.getd2_datamocks([datafileNorth[i],datafileSouth[i]],covmatd2A,combine=True,nbspl=nbspl,bias=thebias[i])
    nm,nb=shape(all_d2A)
    for j in arange(nm):
        plot(r,all_d2A[j,:],color='green',alpha=0.01)
    title('North+South - '+bins[i])
    draw()
    savefig('Result_Both_'+bins[i]+'.png',dpi=300)










savez('data_saved.npz',nbins=nbins,zedge=zedge,rhs=rhs,drhs=drhs,rhn=rhn,drhn=drhn,rha=rha,drha=drha)
toto=np.load('data_saved.npz')
nbins=toto['nbins']
zedge=toto['zedge']
rhs=toto['rhs']
drhs=toto['drhs']
rhn=toto['rhn']
drhn=toto['drhn']
rha=toto['rha']
drha=toto['drha']


import cosmolopy
zvals=linspace(min(zedge),max(zedge),100)
allrh_d2=zeros(100)
# Fiducial cosmology used for DD,RR calculation
thecosmo=cosmolopy.fidcosmo.copy()
thecosmo['baryonic_effects']=True
thecosmo['omega_M_0']=0.27
thecosmo['h']=0.7
thecosmo['omega_lambda_0']=1-thecosmo['omega_M_0']
# Planck 2013 Cosmology
#thecosmo=cosmolopy.fidcosmo.copy()
#thecosmo['baryonic_effects']=True
#thecosmo['omega_M_0']=0.3175      #Planck value
#thecosmo['sigma_8']=0.8344      #Planck value
#thecosmo['h']=0.6711      #Planck value
#thecosmo['omega_b_0']=0.049      #Planck value
#thecosmo['omega_lambda_0']=1-thecosmo['omega_M_0']

import pyxi
for i in arange(100):
    thexi=pyxi.xith(thecosmo,zvals[i])
    allrh_d2[i]=thexi.rh_d2()


dz=zeros(nbins)+(zedge[1]-zedge[0])/2
zmid=(zedge[arange(nbins)]+zedge[arange(nbins)+1])/2

ax=clf()
ax=xlabel('z')
ax=ylabel('$r_H(r)$ [$h^{-1}.\mathrm{Mpc}$]')
ax=ylim(40,80)
ax=xlim(min(zedge)-0.01,max(zedge)+0.01)
ax=plot(zvals,allrh_d2,label='$\Lambda$CDM',color='g',lw=2)
ax=errorbar(zmid,rhs,xerr=dz,yerr=drhs,fmt='ro',label='DR10 CMASS South',ms=8,elinewidth=2)
ax=errorbar(zmid,rhn,xerr=dz,yerr=drhn,fmt='bo',label='DR10 CMASS North',ms=8,elinewidth=2)
ax=errorbar(zmid,rha,xerr=dz,yerr=drha,fmt='ko',label='DR10 CMASS Both',ms=8,elinewidth=2)
ax=legend(loc='lower right',frameon=False)
#savefig('results_z.png',dpi=300)




### need to do the MCMC on mocks
nmocks=599

rhn_mocks=np.zeros((nbins,nmocks))
drhn_mocks=np.zeros((nbins,nmocks))
rhs_mocks=np.zeros((nbins,nmocks))
drhs_mocks=np.zeros((nbins,nmocks))
rha_mocks=np.zeros((nbins,nmocks))
drha_mocks=np.zeros((nbins,nmocks))
chi2n_mocks=np.zeros((nbins,nmocks))
ndfn_mocks=np.zeros((nbins,nmocks))
chi2s_mocks=np.zeros((nbins,nmocks))
ndfs_mocks=np.zeros((nbins,nmocks))
chi2a_mocks=np.zeros((nbins,nmocks))
ndfa_mocks=np.zeros((nbins,nmocks))
nbspl=6
for i in arange(nbins):
    print(bins[i])
    allmocksS=glob.glob(mockdirSouth[i]+'pairs_*.txt')
    allmocksN=glob.glob(mockdirNorth[i]+'pairs_*.txt')
    covmatd2N,meand2N=galtools.read_mocks(mockdirNorth[i])
    covmatd2S,meand2S=galtools.read_mocks(mockdirSouth[i])
    covmatd2A,meand2A=galtools.read_mocks([mockdirNorth[i],mockdirSouth[i]],combine=True)
    num=0
    for j in arange(nmocks):
        print(allmocksS[j])
        rhn_mocks[i,num],drhn_mocks[i,num],res=galtools.getd2_datamocks(allmocksN[j],covmatd2N,nbspl=nbspl,bias=thebias[i])
        chi2n_mocks[i,num]=res.chi2
        ndfn_mocks[i,num]=res.ndf
        draw()
        rhs_mocks[i,num],drhs_mocks[i,num],res=galtools.getd2_datamocks(allmocksS[j],covmatd2S,nbspl=nbspl,bias=thebias[i])
        chi2s_mocks[i,num]=res.chi2
        ndfs_mocks[i,num]=res.ndf
        draw()
        rha_mocks[i,num],drha_mocks[i,num],res=galtools.getd2_datamocks([allmocksN[j],allmocksS[j]],covmatd2A,nbspl=nbspl,combine=True,bias=thebias[i])
        chi2a_mocks[i,num]=res.chi2
        ndfa_mocks[i,num]=res.ndf
        draw()
        num=num+1

ndf=ndfa_mocks[0,0]
clf()
hist(chi2a_mocks[0,:]/ndf,30)
hist(chi2a_mocks[1,:]/ndf,30)
hist(chi2a_mocks[2,:]/ndf,30)
hist(chi2a_mocks[3,:]/ndf,30)
hist(chi2a_mocks[4,:]/ndf,30)

for i in arange(nbins): print(mean(chi2a_mocks[i,:]/ndf),std(chi2a_mocks[i,:]/ndf)/sqrt(nmocks))


savez('mock_saved.npz',nbins=nbins,zedge=zedge,rhs_mocks=rhs_mocks,drhs_mocks=drhs_mocks,rhn_mocks=rhn_mocks,drhn_mocks=drhn_mocks,rha_mocks=rha_mocks,drha_mocks=drha_mocks,chi2n_mocks=chi2n_mocks, ndfn_mocks=ndfn_mocks,chi2s_mocks=chi2s_mocks,ndfs_mocks=ndfs_mocks,chi2a_mocks=chi2a_mocks,ndfa_mocks=ndfa_mocks)
toto=np.load('mock_saved.npz')
nbins=toto['nbins']
zedge=toto['zedge']
rhs_mocks=toto['rhs_mocks']
drhs_mocks=toto['drhs_mocks']
chi2s_mocks=toto['chi2s_mocks']
ndfs_mocks=toto['ndfs_mocks']
rhn_mocks=toto['rhn_mocks']
drhn_mocks=toto['drhn_mocks']
chi2n_mocks=toto['chi2n_mocks']
ndfn_mocks=toto['ndfn_mocks']
rha_mocks=toto['rha_mocks']
drha_mocks=toto['drha_mocks']
chi2a_mocks=toto['chi2a_mocks']
ndfa_mocks=toto['ndfa_mocks']

                


mean_rhn_mocks=zeros(nbins)
sig_rhn_mocks=zeros(nbins)
mean_rhs_mocks=zeros(nbins)
sig_rhs_mocks=zeros(nbins)
mean_rha_mocks=zeros(nbins)
sig_rha_mocks=zeros(nbins)
for i in arange(nbins):
    mean_rhn_mocks[i],sig_rhn_mocks[i]=galtools.meancut(rhn_mocks[i,:])
    mean_rhs_mocks[i],sig_rhs_mocks[i]=galtools.meancut(rhs_mocks[i,:])
    mean_rha_mocks[i],sig_rha_mocks[i]=galtools.meancut(rha_mocks[i,:])


dz=zeros(nbins)+(zedge[1]-zedge[0])/2
zmid=(zedge[arange(nbins)]+zedge[arange(nbins)+1])/2

clf()
errorbar(zmid,mean_rha_mocks,xerr=dz,yerr=sig_rha_mocks/sqrt(600),fmt='ko')
errorbar(zmid,mean_rhn_mocks,xerr=dz,yerr=sig_rhn_mocks/sqrt(600),fmt='bo')
errorbar(zmid,mean_rhs_mocks,xerr=dz,yerr=sig_rhs_mocks/sqrt(600),fmt='ro')


dz=zeros(nbins)+(zedge[1]-zedge[0])/2
zmid=(zedge[arange(nbins)]+zedge[arange(nbins)+1])/2
clf()
xlabel('z')
ylabel('$r_H(r)$ [$h^{-1}.\mathrm{Mpc}$]')
ylim(40,80)
xlim(min(zedge)-0.01,max(zedge)+0.01)
plot(zmid,mean_rha_mocks,color='k',lw=3,label='Mocks average')
plot(zmid,mean_rha_mocks+sig_rha_mocks,color='k',lw=2,ls=':',label='Mocks +/1 $\sigma$')
plot(zmid,mean_rha_mocks-sig_rha_mocks,color='k',lw=2,ls=':')
plot(zmid,mean_rhn_mocks,color='b',lw=2)
plot(zmid,mean_rhn_mocks+sig_rhn_mocks,color='b',lw=2,ls=':')
plot(zmid,mean_rhn_mocks-sig_rhn_mocks,color='b',lw=2,ls=':')
plot(zmid,mean_rhs_mocks,color='r',lw=2)
plot(zmid,mean_rhs_mocks+sig_rhs_mocks,color='r',lw=2,ls=':')
plot(zmid,mean_rhs_mocks-sig_rhs_mocks,color='r',lw=2,ls=':')
errorbar(zmid,rhs,xerr=dz,yerr=drhs,fmt='ro',label='DR10 CMASS South',ms=8,elinewidth=2)
errorbar(zmid,rhn,xerr=dz,yerr=drhn,fmt='bo',label='DR10 CMASS North',ms=8,elinewidth=2)
errorbar(zmid,rha,xerr=dz,yerr=drha,fmt='ko',label='DR10 CMASS Both',ms=8,elinewidth=2)
legend(loc='lower right',frameon=False)
savefig('results_z_withmocks.png',dpi=300)

clf()
hist(rha_mocks[0,:],30,histtype='stepfilled')


##### check dependency with nbspl
nmocks=50
allnbspl=arange(8)+6
nn=size(allnbspl)

rha_mocks=np.zeros((nn,nbins,nmocks))
drha_mocks=np.zeros((nn,nbins,nmocks))
chi2a_mocks=np.zeros((nn,nbins,nmocks))
ndfa_mocks=np.zeros((nn,nbins,nmocks))
for k in arange(nn):
    for i in arange(nbins):
        print(bins[i])
        allmocksS=glob.glob(mockdirSouth[i]+'pairs_*.txt')
        allmocksN=glob.glob(mockdirNorth[i]+'pairs_*.txt')
        covmatd2N,meand2N=galtools.read_mocks(mockdirNorth[i])
        covmatd2S,meand2S=galtools.read_mocks(mockdirSouth[i])
        covmatd2A,meand2A=galtools.read_mocks([mockdirNorth[i],mockdirSouth[i]],combine=True)
        num=0
        for j in arange(nmocks):
            print(allmocksS[j])
            rha_mocks[k,i,num],drha_mocks[k,i,num],res=galtools.getd2_datamocks([allmocksN[j],allmocksS[j]],covmatd2A,nbspl=allnbspl[k],combine=True)
            chi2a_mocks[k,i,num]=res.chi2
            ndfa_mocks[k,i,num]=res.ndf
            draw()
            num=num+1

mchi2=zeros((nn,nbins))
schi2=zeros((nn,nbins))
mrh=zeros((nn,nbins))
srh=zeros((nn,nbins))
for k in arange(nn):
    for i in arange(nbins):
        ndf=ndfa_mocks[k,i,0]
        mchi2[k,i]=mean(chi2a_mocks[k,i,:]/ndf)
        schi2[k,i]=std(chi2a_mocks[k,i,:]/ndf)/sqrt(nmocks)
        mrh[k,i]=mean(rha_mocks[k,i,:])
        srh[k,i]=mean(rha_mocks[k,i,:])/sqrt(nmocks)

decal=linspace(-0.1,0.1,nbins)
clf()
subplot(211)
for i in arange(nbins): errorbar(allnbspl+decal[i],mchi2[:,i],yerr=schi2[:,i],label=bins[i])
ylabel('$\chi^2/ndf$')
xlabel('Number of Spline nodes')
legend(loc='upper right',frameon=False)
subplot(212)
for i in arange(nbins): errorbar(allnbspl+decal[i],mrh[:,i],yerr=srh[:,i])
xlabel('Number of Spline nodes')
ylabel('$r_H$')
savefig('number_of_spline_nodes.png')









