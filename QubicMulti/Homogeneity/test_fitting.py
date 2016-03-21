#voir: http://nbviewer.ipython.org/urls/raw.github.com/iminuit/probfit/master/tutorial/tutorial.ipynb


from probfit import describe, Chi2Regression
import iminuit
import numpy as np
import numpy.random as npr

npr.seed(0)
x = linspace(0,10,20)
y = 3*x+15+ randn(20)
err = np.array([1]*20)
errorbar(x,y,err,fmt='.');

#lets define our line
#first argument has to be independent variable
#arguments after that are shape parameters
def line(x,m,c): #define it to be parabolic or whatever you like
    return m*x+c
#We can make it faster but for this example this is plentily fast.
#We will talk about speeding things up later(you will need cython)

describe(line)

#cost function
chi2 = Chi2Regression(line,x,y,err)
#Chi^2 regression is just a callable object nothing special about it
describe(chi2)

#minimize it
#yes it gives you a heads up that you didn't give it initial value
#we can ignore it for now
minimizer = iminuit.Minuit(chi2) #see iminuit tutorial on how to give initial value/range/error
minimizer.migrad(); #very stable robust minimizer
#you can look at your terminal to see what it is doing;


#lets see our results
print minimizer.values
print minimizer.errors

chi2.draw(minimizer)






#### essais JC

### without probfit but uses global variables... useless...
from iminuit import Minuit
import numpy as np
import numpy.random as npr

npr.seed(0)
x = linspace(0,10,20)
y = 3*x+15+ randn(20)
err = np.array([1]*20)

clf()
errorbar(x,y,err,fmt='.');

def f(x, a, b): return a + b*x

def chi2(a, b):
    c2 = 0.
    for i in np.arange(np.size(x)):
        c2 += (f(x[i], a, b) - y[i])**2 / err[i]**2

    return(c2)

print(chi2(15,3))


m = Minuit(chi2,a=1,b=1,fix_a=False,fix_b=False)
m.migrad()
fit=m.args

plot(x,f(x,fit[0],fit[1]))

####### try mpfit
import mpfit
import numpy as np
import numpy.random as npr

deginit=4
coeffs=npr.randn(deginit+1)
x = linspace(0,10,20)
pin=np.poly1d(coeffs)
y = pin(x)
noise=(max(y)-min(y))/30
y=y+noise*npr.randn(np.size(x))
err = np.array([noise]*20)

clf()
errorbar(x,y,err,fmt='.');

def myfunct(pars, fjac=None, x=None, y=None, err=None):
    pol=np.poly1d(pars)
    model = pol(x)
    status = 1
    resid=(y-model)/err
    print(resid)
    return([status, resid])
    
deg=4
p0=np.zeros(deg+1)
fa={'x':x,'y':y,'err':err}
mpf = mpfit.mpfit(myfunct, p0, functkw=fa)
print 'status = ', mpf.status
if (mpf.status <= 0): print 'error message = ', mpf.errmsg
print 'parameters = ', mpf.params

f=poly1d(mpf.params)
plot(x,f(x))








#### essai suite au mail de Piti (developpeur de iminuit)
from probfit import describe, Chi2Regression, Polynomial
import iminuit
import numpy as np
import numpy.random as npr

deginit=4
coeffs=npr.randn(deginit+1)
x = linspace(0,10,20)
pin=np.poly1d(coeffs)
y = pin(x)
noise=(max(y)-min(y))/30
y=y+noise*npr.randn(np.size(x))
err = np.array([noise]*20)

errorbar(x,y,err,fmt='.');

deg=4
init_param={'c_%d'%i:0 for i in range(deg+1)}
p=Polynomial(deg)
chi2 = Chi2Regression(p,x,y,error=err)
describe(chi2)

minimizer = iminuit.Minuit(chi2,**init_param)
minimizer.migrad()

print minimizer.values
print minimizer.errors

clf()
chi2.draw(minimizer)




##### autre maniere (de piti) avec une fonction
from probfit import describe, Chi2Regression, Polynomial
import iminuit
import numpy as np
import numpy.random as npr

deginit=4
coeffs=npr.randn(deginit+1)
x = linspace(0,10,20)
pin=np.poly1d(coeffs)
y = pin(x)
noise=(max(y)-min(y))/30
y=y+noise*npr.randn(np.size(x))
err = np.array([noise]*20)

clf()
errorbar(x,y,yerr=err,fmt='ro')

deg=4
parnames=[]
for i in range(deg+1): parnames.append('c_'+np.str(i))

def mypol(x,*pars):
    f=np.poly1d(pars)
    return(f(x))

chi2 = Chi2Regression(mypol,x,y,error=err)
describe(chi2)

minimizer = iminuit.Minuit(chi2,forced_parameters=parnames)
minimizer.migrad()

print minimizer.values
print minimizer.errors

clf()
chi2.draw(minimizer)


#### on essaie sans probfit
import iminuit
import numpy as np
import numpy.random as npr

deginit=6
coeffs=npr.randn(deginit+1)
x = linspace(0,10,20)
pin=np.poly1d(coeffs)
y = pin(x)
noise=(max(y)-min(y))/30
y=y+noise*npr.randn(np.size(x))
err = np.array([noise]*20)

clf()
errorbar(x,y,yerr=err,fmt='ro')

def mypol(x,pars):
    print(pars)
    f=np.poly1d(pars)
    return(f(x))

class MyChi2:
    def __init__(self,x,y,err,functname):
        self.x=x
        self.y=y
        self.err=err
        self.functname=functname
    def __call__(self,*pars):
        val=self.functname(x,pars)
        chi2=np.sum( ((self.y-val)/self.err)**2 )
        return(chi2)

chi2=MyChi2(x,y,err,mypol)
iminuit.describe(chi2)

deg=6
parnames=[]
for i in range(deg+1): parnames.append('c_'+np.str(i))

m = iminuit.Minuit(chi2,forced_parameters=parnames)
m.migrad()
print m.values

parfit=[]
for i in parnames: parfit.append(m.values[i])


thepol=np.poly1d(parfit)
plot(x,thepol(x))






############ test my fitting
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
import mpfit
import iminuit
import FitPoly
from Homogeneity import fitting

datafileNorth='/Users/hamilton/SDSS/Homogeneity/dr10North_pairs_weighted.txt'
datafileSouth='/Users/hamilton/SDSS/Homogeneity/dr10South_pairs_weighted.txt'
mockdirNorth='/Users/hamilton/SDSS/DR10/PTHaloMocks/Pairs_Log/North/'
mockdirSouth='/Users/hamilton/SDSS/DR10/PTHaloMocks/Pairs_Log/South/'


rd,d2_r,covmatd2,rhd2=galtools.read_datamocks(datafileNorth,mockdirNorth)
x=rd
y=d2_r
cov=covmatd2

#### right way to do the fit: 28/03/2013
rh,drh=galtools.get_rh_spline(rd,d2_r,covmatd2,thresh=2.97,nbspl=12)


#### older tests below...


# get desired sub array
xstart=30
xstop=200
w=np.where((x >= xstart) & (x <= xstop))
thex=x[w]
they=y[w]
theyerr=np.sqrt(cov[w[0],w[0]])
thecov=(cov[w[0],:])[:,w[0]]
theinvcov=np.array(np.matrix(thecov).I)
theinvcovdiag=zeros((np.size(w),np.size(w)))
theinvcovdiag[np.arange(np.size(w)),np.arange(np.size(w))]=1./theyerr**2




errsz=2
thex=linspace(0,1,50)
theyerr=np.zeros(np.size(thex))+errsz
they=30*thex+10+randn(np.size(thex))*errsz
thecov=np.zeros((np.size(thex),np.size(thex)))
thecov[arange(np.size(thex)),arange(np.size(thex))]=theyerr**2
clf()
errorbar(thex,they,yerr=theyerr,fmt='ro')

deg=7
guess=np.polyfit(thex, they, deg)


fitbusca=FitPoly.FitPoly(thex,they,thecov,deg)
pars_nico=fitbusca.pars
cov_nico=fitbusca.covpar
err_nico=np.sqrt(np.diagonal(cov_nico))

m,parfit_m,errfit_m,covar_m=fitting.do_minuit(thex,they,thecov,guess)

mp,parfit_mp,errfit_mp,covar_mp=fitting.do_mpfit(thex,they,thecov,guess)

chain,parfit_mc,errfit_mc,covar_mc=fitting.do_emcee(thex,they,thecov,guess)

clf()
subplot(211)
errorbar(thex,they,yerr=theyerr,fmt='ro')
plot(thex,fitting.thepolynomial(thex,parfit_m),label='Minuit',lw=5)
plot(thex,fitting.thepolynomial(thex,parfit_mp),label='MPFIT',lw=3)
plot(thex,fitting.thepolynomial(thex,parfit_mc),label='MCMC',lw=3,ls='--')
plot(thex,fitting.thepolynomial(thex,pars_nico),label='Exact',lw=3,ls=':')
errorbar(thex,they,yerr=theyerr,fmt='ro')
legend(loc='lower right',frameon=False)

subplot(223)
errorbar(arange(deg+1)-0.1,parfit_m,yerr=errfit_m,label='Minuit',fmt='ro')
errorbar(arange(deg+1),parfit_mp,yerr=errfit_mp,label='MPFIT',fmt='bo')
errorbar(arange(deg+1)+0.1,parfit_mc,yerr=errfit_mc,label='MCMC',fmt='go')
errorbar(arange(deg+1)+0.2,pars_nico,yerr=err_nico,label='Exact',fmt='ko')
legend(loc='lower right',frameon=False)


subplot(224)
ylim(0,1.1)
plot(errfit_m/err_nico,label='Minuit/Exact')
plot(errfit_mp/err_nico,label='MPFit/Exact')
plot(errfit_mc/err_nico,label='MCMC/Exact')
legend(loc='lower right',frameon=False)



#### residuals
clf()
subplot(211)
errorbar(thex,they,yerr=theyerr,fmt='ro')
newx=linspace(thex.min(),thex.max(),100)
plot(newx,fitting.thepolynomial(newx,pars_nico))
subplot(212)
errorbar(thex,they-fitting.thepolynomial(thex,pars_nico),yerr=theyerr,fmt='ro')
plot(thex,thex*0,ls=':')




clf()
subplot(221)
imshow(galtools.cov2cor(np.abs(covar_m)),interpolation='nearest',aspect='equal',vmin=0,vmax=1)
title('MPFIT')
colorbar()
subplot(222)
imshow(galtools.cov2cor(np.abs(covar_mp)),interpolation='nearest',aspect='equal',vmin=0,vmax=1)
title('Minuit')
colorbar()
subplot(223)
imshow(galtools.cov2cor(np.abs(covar_mc)),interpolation='nearest',aspect='equal',vmin=0,vmax=1)
title('MCMC')
colorbar()
subplot(224)
imshow(galtools.cov2cor(np.abs(cov_nico)),interpolation='nearest',aspect='equal',vmin=0,vmax=1)
title('Exact')
colorbar()



#### explore chi2 with Minuit and others

chi2=fitting.MyChi2(thex,they,thecov,fitting.thepolynomial)

parsnum=[deg-1,deg]
nsigmc=3
nb=64
par0range=linspace(parfit_m[parsnum[0]]-nsigmc*errfit_mc[parsnum[0]], parfit_m[parsnum[0]]+nsigmc*errfit_mc[parsnum[0]],nb)
par1range=linspace(parfit_m[parsnum[1]]-nsigmc*errfit_mc[parsnum[1]], parfit_m[parsnum[1]]+nsigmc*errfit_mc[parsnum[1]],nb)
u,s,v=np.linalg.svd(thecov)


chi2minuitmap=zeros((nb,nb))
chi2mpmap=zeros((nb,nb))
chi2mcmcmap=zeros((nb,nb))
for i in arange(nb):
    print(i)
    for j in arange(nb):
        pp=parfit_m.copy()
        pp[parsnum[0]]=par0range[i]
        pp[parsnum[1]]=par1range[j]
        chi2minuitmap[i,j]=chi2(*pp)
        chi2mpmap[i,j]=np.sum((fitting.chi2svd(pp,x=thex,y=they,svdvals=s,v=v,functname=fitting.thepolynomial)[1])**2)
        chi2mcmcmap[i,j]=-2*fitting.lnprobcov(pp,thex,they,s,v,fitting.thepolynomial)

clf()

cs=contour(par0range,par1range,chi2minuitmap-chi2minuitmap.min(),[1,4,9],linewidths=10,colors='red')
contour(par0range,par1range,chi2mpmap-chi2mpmap.min(),[1,4,9],linewidths=7,colors='green')
contour(par0range,par1range,chi2mcmcmap-chi2mcmcmap.min(),[1,4,9],linewidths=2,colors='k')
plot(30,10,marker='+',markersize=50,mew=5)



#### look at chi2/ndf as a function of degree
alldegs=arange(8)+1

chi2min=zeros(np.size(alldegs))
ndf=zeros(np.size(alldegs))
for i in arange(np.size(alldegs)):
    guess=np.polyfit(thex, they, alldegs[i])
    m,parfit_m,errfit_m,covar_m=fitting.do_minuit(thex,they,thecov,guess)
    chi2min[i]=m.fval
    ndf[i]=np.size(thex)-m.narg

plot(alldegs,chi2min/ndf)




#### try other functions with minuit
def fctarctan(x,pars):
    val=pars[0]+pars[1]*np.arctan((x-pars[2])/pars[3])
    return(val)

fct=fctarctan
guess=[2.7,0.2,20.,10.]
clf()
errorbar(thex,they,yerr=theyerr,fmt='ro')
plot(thex,fct(thex,guess))

def sigmoid(x,pars):
    val=pars[0]+pars[1]/(1+np.exp(-(x-pars[2])/pars[3]))
    return(val)

fct=sigmoid
guess=np.array([2.7,0.3,20,15.])
clf()
errorbar(thex,they,yerr=theyerr,fmt='ro')
plot(thex,fct(thex,guess))

clf()
subplot(211)
errorbar(thex,they,yerr=theyerr,fmt='ro')
m,parfit,errfit,covar=fitting.do_minuit(thex,they,thecov,guess,functname=fct)
plot(thex,fct(thex,parfit))
subplot(212)
errorbar(thex,they-fct(thex,parfit),yerr=theyerr,fmt='ro')
plot(thex,thex*0,ls=':')





######################### spline fitting
import SplineFitting

#### How many nodes ?
allnbspl=5+np.arange(15)
chi2=zeros(np.size(allnbspl))
ndf=zeros(np.size(allnbspl))
for i in arange(np.size(allnbspl)):
    print(allnbspl[i])
    spl=SplineFitting.MySplineFitting(thex,they,thecov,allnbspl[i],logspace=True)
    chi2[i]=spl.chi2
    ndf[i]=spl.ndf

clf()
ylim(0,10)
plot(allnbspl,chi2/ndf)
######## 9 looks reasonnable



nbspl=12
spl=SplineFitting.MySplineFitting(thex,they,thecov,nbspl,logspace=True)

clf()
subplot(211)
errorbar(thex,they,yerr=theyerr,fmt='ro')
newx=linspace(thex.min(),thex.max(),100)
plot(newx,spl(newx))
subplot(212)
errorbar(thex,they-spl(thex),yerr=theyerr,fmt='ro')
plot(thex,thex*0,ls=':')

clf()
imshow(spl.covout,interpolation='nearest')

clf()
imshow(galtools.cov2cor(spl.covout),interpolation='nearest')

# get rh and error bar
newx=linspace(thex.min(),thex.max(),1000)
ff=interpolate.interp1d(spl(newx),newx)
rh=ff(2.97)

clf()
errorbar(thex,they,yerr=theyerr,fmt='ro')
plot(newx,spl(newx))
plot(newx,newx*0+2.97,ls=':',c='k')
plot(newx*0+rh,spl(newx),ls=':',c='k')

# Error
# derivative w.r.t. coefficients
thepartial=np.zeros(spl.nbspl)
for i in arange(spl.nbspl):
    pval=linspace(spl.alpha[i]-0.01*spl.dalpha[i],spl.alpha[i]+0.01*spl.dalpha[i],2)
    yyy=zeros(np.size(pval))
    for j in arange(np.size(pval)):
        thepars=np.copy(spl.alpha)
        thepars[i]=pval[j]
        yyy[j]=spl.with_alpha(rh,thepars)
    
    thepartial[i]=np.diff(yyy)/np.diff(pval)

err_on_funct=np.sqrt(dot(dot(thepartial,spl.covout),thepartial))

newx=linspace(thex.min(),thex.max(),1000)
deriv_spl=interpolate.interp1d(newx[1:1000],np.diff(spl(newx))/np.diff(newx))

rh_m=rh
drh_m=err_on_funct/deriv_spl(rh)

clf()
xv=linspace(min(thex),max(thex),1000)
xlabel('$r [h^{-1}.\mathrm{Mpc}]$')
ylabel('$d_2(r)$')
plot(arange(200),zeros(200)+2.97,'--')
errorbar(thex,they,yerr=theyerr,fmt='ko')
plot(xv,spl(xv),label='Spline Fit (with covariance)',color='b')
errorbar(rh_m,2.97,xerr=drh_m,label='$R_H$ = '+str('%.1f'%rh_m)+' $\pm$ '+str('%.1f'%drh_m)+' $h^{-1}.\mathrm{Mpc}$',fmt='ro')
legend(loc='lower right')




##### test
from Homogeneity import fitting
import numpy as np
import numpy.random as npr
npr.seed(0)
x = linspace(0,10,20)
y = 3*x**2+15+ randn(20)
err = np.array([1]*20)
errorbar(x,y,err,fmt='.');

toto=fitting.dothefit(x,y,err,[0,0,0],functname=fitting.thepolynomial)

plot(x,fitting.thepolynomial(x,toto[1]))
