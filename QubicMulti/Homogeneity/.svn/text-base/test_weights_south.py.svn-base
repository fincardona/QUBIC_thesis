import scipy.io
from Cosmology import cosmology
from Homogeneity import galtools
from scipy import integrate
from scipy import interpolate
from matplotlib import rc
import copy
import pyfits
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



#dirbase='/Users/hamilton/SDSS/Data/'
#dirbase='/Volumes/Data/SDSS/'
dirbase='/Volumes/Data/SDSS/DR10/LRG/'

#### South
data,hdr_data=pyfits.getdata(dirbase+'cmass-dr10v5-S-Anderson.dat.fits',header=True)
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


rnd,hdr_data=pyfits.getdata(dirbase+'cmass-dr10v5-S-Anderson.ran.fits',header=True)
wok=where(rnd.field('z')>minz)
rnd=rnd[wok]
wok=where(rnd.field('z')<maxz)
rnd=rnd[wok]
rnd=rnd[0:size(dataraS)*3]

randomraS=rnd.field('RA')
randomdecS=rnd.field('DEC')
randomzS=rnd.field('z')
random_wfkpS=rnd.field('WEIGHT_FKP')
random_wS=random_wfkpS

# define r bins
rmin=0.
rmax=200.
nbins=50

r,dd,rr,dr=galtools.paircount_data_random(dataraS,datadecS,datazS,randomraS,randomdecS,randomzS,cosmo,rmin,rmax,nbins,log=None,file='dr10South_pairs_forxi_weighted.txt',nproc=8,wdata=data_wS,wrandom=random_wS)

r,dd,rr,dr=galtools.paircount_data_random(dataraS,datadecS,datazS,randomraS,randomdecS,randomzS,cosmo,rmin,rmax,nbins,log=None,file='dr10South_pairs_forxi.txt',nproc=8)




south=np.loadtxt('dr10South_pairs_forxi.txt')
rs_nw=south[:,0]
dds_nw=south[:,1]
rrs_nw=south[:,2]
drs_nw=south[:,3]
south=np.loadtxt('dr10South_pairs_forxi_weighted.txt')
rs=south[:,0]
dds=south[:,1]
rrs=south[:,2]
drs=south[:,3]

r=rs_nw
xilss_nw=(dds_nw-2*drs_nw+rrs_nw)/rrs_nw
xilss=(dds-2*drs+rrs)/rrs

clf()
ylim(-50,350)
plot(r,xilss_nw*r*r,label='South No Weights',linewidth=3)
plot(r,xilss*r*r,label='South all Weights',linewidth=3)
legend(loc='upper left')
savefig('south.pdf')


