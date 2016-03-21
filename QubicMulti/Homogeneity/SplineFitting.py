import numpy as np
from pylab import *
from scipy import interpolate

class MySplineFitting:
    def __init__(self,xin,yin,covarin,nbspl,logspace=False):
        # input parameters
        self.x=xin
        self.y=yin
        self.nbspl=nbspl
        covar=covarin
        if np.size(np.shape(covarin)) == 1:
            err=covarin
            covar=np.zeros((np.size(err),np.size(err)))
            covar[np.arange(np.size(err)),np.arange(np.size(err))]=err**2
        
        self.covar=covar
        self.invcovar=np.linalg.inv(covar)
        
        # Prepare splines
        xspl=np.linspace(np.min(self.x),np.max(self.x),nbspl)
        if logspace==True: xspl=np.logspace(np.log10(np.min(self.x)),np.log10(np.max(self.x)),nbspl)
        self.xspl=xspl
        F=np.zeros((np.size(xin),nbspl))
        self.F=F
        for i in np.arange(nbspl):
            self.F[:,i]=self.get_spline_tofit(xspl,i,xin)
        
        # solution of the chi square
        ft_cinv_y=np.dot(np.transpose(F),np.dot(self.invcovar,self.y))
        covout=np.linalg.inv(np.dot(np.transpose(F),np.dot(self.invcovar,F)))
        alpha=np.dot(covout,ft_cinv_y)
        fitted=np.dot(F,alpha)
        
        # output
        self.residuals=self.y-fitted
        self.chi2=np.dot(np.transpose(self.residuals), np.dot(self.invcovar, self.residuals))
        self.ndf=np.size(xin)-np.size(alpha)
        self.alpha=alpha
        self.covout=covout
        self.dalpha=np.sqrt(np.diagonal(covout))
    
    def __call__(self,x):
        theF=np.zeros((np.size(x),self.nbspl))
        for i in np.arange(self.nbspl): theF[:,i]=self.get_spline_tofit(self.xspl,i,x)
        return(dot(theF,self.alpha))

    def with_alpha(self,x,alpha):
        theF=np.zeros((np.size(x),self.nbspl))
        for i in np.arange(self.nbspl): theF[:,i]=self.get_spline_tofit(self.xspl,i,x)
        return(dot(theF,alpha))
            
    def get_spline_tofit(self,xspline,index,xx):
        yspline=zeros(np.size(xspline))
        yspline[index]=1.
        tck=interpolate.splrep(xspline,yspline)
        yy=interpolate.splev(xx,tck,der=0)
        return(yy)


