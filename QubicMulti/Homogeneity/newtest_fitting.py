from Homogeneity import fitting
import numpy as np


nn=10000
x0=1.5
sigma=3.
xvalues = np.random.randn(nn)*sigma+x0

res = hist(xvalues, bins=100, range = [-10,15])

xcenter = (res[1][:-1]+res[1][1:])/2
yvals = res[0]
errors = np.sqrt(yvals)


def mygaussian(x,pars):
	return(pars[0]*np.exp(-(x-pars[1])**2/(2*pars[2]**2)))


avguess = np.sum(xcenter*yvals)/np.sum(yvals)
av2guess = np.sum(xcenter**2*yvals)/np.sum(yvals)
varguess = av2guess-avguess**2

guess = [np.max(yvals), avguess, np.sqrt(varguess)]
ok = errors != 0
result_minuit = fitting.dothefit(xcenter[ok],yvals[ok],errors[ok],guess,functname=mygaussian,method='minuit')
result_mpfit = fitting.dothefit(xcenter[ok],yvals[ok],errors[ok],guess,functname=mygaussian,method='mpfit')
result_mcmc = fitting.dothefit(xcenter[ok],yvals[ok],errors[ok],guess,functname=mygaussian,method='mcmc')

clf()
errorbar(xcenter,yvals,yerr=errors,fmt='ro')
xxx = np.linspace(-10,15,1000)
plot(xxx, mygaussian(xxx, result_minuit[1]))
plot(xxx, mygaussian(xxx, result_mpfit[1]))
plot(xxx, mygaussian(xxx, result_mcmc[1]))



