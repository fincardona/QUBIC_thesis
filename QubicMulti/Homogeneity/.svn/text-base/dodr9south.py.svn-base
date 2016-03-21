import scipy.io
import cosmology as cosmo
import galtools
from scipy import integrate
from scipy import interpolate
from matplotlib import rc
import copy
rc('text', usetex=False)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# define z bins
minz=0.43
maxz=0.7
nz=5
zedges=linspace(minz,maxz,nz+1)
zmin=zedges[0:nz]
zmax=zedges[1:nz+1]

# define fiducial cosmology
cosmo=[0.27,0.73,-1,0,0.7]

# define r bins
rmin=10.
rmax=500.
nbins=50


#dirbase='/Users/hamilton/SDSS/Data/'
dirbase='/Volumes/Data/SDSS/'

# restore IDL-made file for DR9 South ##################################################################
dr9in=scipy.io.readsav(dirbase+'/DR9/random_and_data_and_geom.save',verbose=True)
#--------------------------------------------------
#Available variables:
# - zdist [<type 'numpy.ndarray'>]
# - ppsub [<class 'numpy.core.records.recarray'>]
# - randomgal [<class 'numpy.core.records.recarray'>]
# - datacut [<class 'numpy.core.records.recarray'>]
#--------------------------------------------------

wg=where(dr9in.datacut.z < 10)
wr=where(dr9in.randomgal.z < 10)
dr9=copy.copy(dr9in)
dr9.datacut=dr9in.datacut[wg]
dr9.randomgal=dr9in.randomgal[wr]

datara=dr9.datacut.ra
datadec=dr9.datacut.dec
dataz=dr9.datacut.z

randomra=dr9.randomgal.ra
randomdec=dr9.randomgal.dec
randomz=dr9.randomgal.z


r,dd,rr,dr=galtools.paircount_data_random(datara,datadec,dataz,randomra,randomdec,randomz,cosmo,rmin,rmax,nbins,log=1,file='dr9South_pairs.txt',nproc=24)
                                 
##########################################################################################################



# North Data (from Mariana Vargas) #######################################################################
data=np.loadtxt(dirbase+'/DR9/FromMariana/gal_CMASS_North_ra_dec_z_weight_wmcmcF.dat')
datara=data[:,0]
datadec=data[:,1]
dataz=data[:,2]
sd=shape(data)

rand=np.loadtxt(dirbase+'/DR9/FromMariana/randoms_CMASS_North_ra_dec_z_weight.dat')
rand=rand[0:3*sd[0],:]
randomra=rand[:,0]
randomdec=rand[:,1]
randomz=rand[:,2]

r,dd,rr,dr=galtools.paircount_data_random(datara,datadec,dataz,randomra,randomdec,randomz,cosmo,rmin,rmax,nbins,log=1,file='dr9North_pairs.txt',nproc=24)

##########################################################################################################





##### read north and south
north=np.loadtxt('dr9North_pairs.txt')
rn=north[:,0]
ddn=north[:,1]
rrn=north[:,2]
drn=north[:,3]

south=np.loadtxt('dr9South_pairs.txt')
rs=south[:,0]
dds=south[:,1]
rrs=south[:,2]
drs=south[:,3]

nn=galtools.scalednr(ddn,rrn,2.)
ns=galtools.scalednr(dds,rrs,2.)
rhn=galtools.rhomo_nr(rn,nn)
rhs=galtools.rhomo_nr(rs,ns)

r=rn

clf()
xlim(10,rmax)
ylim(0.98,1.3)
xscale('log')
xlabel('$r [h^{-1}.\mathrm{Mpc}]$')
ylabel('scaled $N(r)$')
title('DR9')
plot(rn,nn,label='North $R_H$ = '+str('%.1f'%rhn)+' $h^{-1}.\mathrm{Mpc}$',color='b')
plot(rs,ns,label='South $R_H$ = '+str('%.1f'%rhs)+' $h^{-1}.\mathrm{Mpc}$',color='r')
plot(r,ddn*0+1,linestyle='--',color='k')
plot(r,ddn*0+1.01,linestyle=':',color='k')
plot([rhn,rhn],[0,2],linestyle=':',color='b')
plot([rhs,rhs],[0,2],linestyle=':',color='r')
legend()


d2n=galtools.d2(r,ddn,rrn,2.)
d2s=galtools.d2(r,dds,rrs,2.)
rhd2n=galtools.rhomo_d2(r,d2n)
rhd2s=galtools.rhomo_d2(r,d2s)


clf()
dlogr=np.log(r[1])-np.log(r[0])
xlim(10,rmax)
xscale('log')
ylim(2.5,3.1)
xlabel('$r [h^{-1}.\mathrm{Mpc}]$')
ylabel('$D_2(r)$')
title('DR9')
plot(r,d2n,color='b',label='North $R_H$ = '+str('%.1f'%rhd2n)+' $h^{-1}.\mathrm{Mpc}$')
plot(r,d2s,color='r',label='South $R_H$ = '+str('%.1f'%rhd2s)+' $h^{-1}.\mathrm{Mpc}$')
plot(r,ddn*0+3,linestyle='--',color='k')
plot(r,ddn*0+2.97,linestyle=':',color='k')
legend(loc='lower right')


