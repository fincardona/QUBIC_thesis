import scipy.io
from Cosmology import cosmology
from Homogeneity import galtools
from scipy import integrate
from scipy import interpolate
from matplotlib import rc
import copy
import glob
import pyfits
rc('text', usetex=False)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# define z bins
minz=0.43
maxz=0.7

# define fiducial cosmology
cosmo=[0.27,0.73,-1,0,0.7]



#dirbase='/Users/hamilton/SDSS/Data/'
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





# define r bins
rmin=0.
rmax=200.
nbins=50
nproc=8

#weighted
r,dd,rr,dr=galtools.paircount_data_random(dataraS,datadecS,datazS,randomraS,randomdecS,randomzS,cosmo,rmin,rmax,nbins,log=None,file='dr10_V6_South_pairs_forxi.txt',nproc=nproc,wdata=data_wS,wrandom=random_wS)

r,dd,rr,dr=galtools.paircount_data_random(dataraN,datadecN,datazN,randomraN,randomdecN,randomzN,cosmo,rmin,rmax,nbins,log=None,file='dr10_V6_North_pairs_forxi.txt',nproc=nproc,wdata=data_wN,wrandom=random_wN)




##### read north and south for xi
rs,dds,rrs,drs,ngs,nrs=galtools.read_pairs('dr10_V6_South_pairs_forxi.txt')

rn,ddn,rrn,drn,ngn,nrn=galtools.read_pairs('dr10_V6_North_pairs_forxi.txt')

ra,dda,rra,dra,nga,nra=galtools.combine_regions('dr10_V6_South_pairs_forxi.txt','dr10_V6_North_pairs_forxi.txt')

r=rn

xilsn=(ddn-2*drn+rrn)/rrn
xilss=(dds-2*drs+rrs)/rrs
xilssum=(dda-2*dra+rra)/rra

clf()
ylim(-50,120)
plot(r,xilsn*r*r,label='North')
plot(r,xilss*r*r,label='South')
plot(r,xilssum*r*r,label='All')
legend()






