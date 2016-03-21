import numpy as np
from pylab import *
import iminuit

####### Generic polynomial function ##########
def thepolynomial(x,pars):
    f=np.poly1d(pars)
    return(f(x))


####### Generic fitting function ##############
def dothefit(x,y,covarin,guess,functname=thepolynomial,method='minuit'):
    if method == 'minuit':
        print('Fitting with Minuit')
        return(do_minuit(x,y,covarin,guess,functname))
    else:
        print('method must be among: minuit')
        return(0,0,0,0)




################### Fitting with Minuit #######################
# Class defining the minimizer and the data
class MyChi2:
    def __init__(self,xin,yin,covarin,functname):
        self.x=xin
        self.y=yin
        self.covar=covarin
        self.invcov=np.linalg.inv(covarin)
        self.functname=functname
            
    def __call__(self,*pars):
        val=self.functname(self.x,pars)
        chi2=dot(dot(self.y-val,self.invcov),self.y-val)
        return(chi2)

def do_minuit(x,y,covarin,guess,functname=thepolynomial):
    # check if covariance or error bars were given
    covar=covarin
    if np.size(np.shape(covarin)) == 1:
        err=covarin
        covar=np.zeros((np.size(err),np.size(err)))
        covar[np.arange(np.size(err)),np.arange(np.size(err))]=err**2
                                    
    # instantiate minimizer
    chi2=MyChi2(x,y,covar,functname)
    # variables
    ndim=np.size(guess)
    parnames=[]
    for i in range(ndim): parnames.append('c_'+np.str(i))
    # initial guess
    theguess=dict(zip(parnames,guess))
    # Run Minuit
    print('Fitting with Minuit')
    m = iminuit.Minuit(chi2,forced_parameters=parnames,errordef=0.01,**theguess)
    m.migrad()
    m.migrad()
    # build np.array output
    parfit=[]
    for i in parnames: parfit.append(m.values[i])
    errfit=[]
    for i in parnames: errfit.append(m.errors[i])
    covariance=np.zeros((ndim,ndim))
    for i in arange(ndim):
        for j in arange(ndim):
            covariance[i,j]=m.covariance[(parnames[i],parnames[j])]

    print('Chi2=',chi2(*parfit))
    print('ndf=',np.size(x)-ndim)
    return(m,np.array(parfit), np.array(errfit), np.array(covariance))

################################################################



