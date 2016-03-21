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



##### read data
rds,dds,rrs,drs,ngs,nrs=galtools.read_pairs('dr10South_pairs_weighted.txt')

ns=galtools.scalednr(dds,rrs,2.)
rhs=galtools.rhomo_nr(rds,ns)

d2sd=galtools.d2(rds,dds,rrs,2.)
rhd2sd=galtools.rhomo_d2(rds,d2sd)


# get error bars from mocks
dirbase='/Users/hamilton/SDSS/Data/DR10/PTHaloMocks/'
outdir=dirbase+'Pairs_Log/'
southfiles=glob.glob(outdir+'South/pairs_*.txt')
rs,nrs,signrs,d2sd,sigd2s,covmatnrs,covmatd2s,cormatnrs,cormatd2s,rhnrs,rhd2s,sigrhnrs,sigrhd2s=galtools.homogeneity_many_pairs(southfiles,2.)


#save
savez('save4pierre.npz',rds=rds,d2sd=d2sd,covmatd2s=covmatd2s)

toto=np.load('save4pierre.npz')
rds=toto['rds']
d2sd=toto['d2sd']
covmatd2s=toto['covmatd2s']

# try polynmial fitting with scipy.optimize.leastsq
w=where((rds >= 30) & (rds<=200))
x=rds[w]
y=d2sd[w]
yerr=np.sqrt(covmatd2s[w[0],w[0]])
cov=(covmatd2s[w[0],:])[:,w[0]]

cov=cov*0
cov[arange(x.size),arange(x.size)]=yerr**2

invcov=np.linalg.inv(cov)
#invcov=np.array(np.matrix(cov).I)


#first simple polynomial fitting
clf()
errorbar(x,y,yerr=yerr,fmt='ro')
poltry=np.polyfit(x,y,5)
p=np.poly1d(poltry)
plot(x,p(x))


import scipy.optimize as opt

def functominimize(params,x,y,invcovar):
    thep=np.poly1d(params)
    delta=y-thep(x)
    chi2=np.dot(np.dot(np.transpose(delta),invcovar),delta)
    return(chi2)

def functominimize_forleastsq(params,args):
    x,y,invcovar=args
    thep=np.poly1d(params)
    delta=(y-thep(x))/yerr
    chi2=np.dot(np.dot(np.transpose(delta),invcovar),delta)
    tata=np.zeros(y.size)
    tata[0]=np.sqrt(chi2)
    #tata=tata+np.sqrt(chi2/y.size)
    return(tata)

def fmodel(x,*params):
    thep=np.poly1d(params)
    return(thep(x))

fitpars0,parscov0=opt.curve_fit(fmodel,x,y,p0=poltry,sigma=yerr)
sqrt(diagonal(parscov0))

p=np.poly1d(fitpars0)
plot(x,p(x))


fitpars1=opt.fmin(functominimize,poltry,args=[x,y,invcov])
p=np.poly1d(fitpars1)
plot(x,p(x))

fitpars2,covpar,infodict,mesg,ier=opt.leastsq(functominimize_forleastsq,poltry,args=[x,y,invcov],full_output=True,maxfev=100000)
p=np.poly1d(fitpars2)
plot(x,p(x))

args=[x,y,invcov]
covpar=covpar*(functominimize_forleastsq(fitpars2,args)**2).sum()/(len(y)-len(poltry))

errs=np.sqrt(np.diagonal(covpar))

clf()
errorbar(x,y,yerr=yerr,fmt='ro')
poltry=np.polyfit(x,y,7)
p=np.poly1d(poltry)
plot(x,p(x))
pp=np.poly1d(fitpars1)
plot(x,pp(x))

