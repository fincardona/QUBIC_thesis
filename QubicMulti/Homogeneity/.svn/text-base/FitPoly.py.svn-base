

import pylab as pl


class FitPoly:
	def __init__(self,x,y,cov,deg):
		self.deg = deg
		self.x = x
		self.y = y
		self.cov = cov
		self.icov = pl.linalg.inv(cov)
		
		self.M = pl.zeros([deg+1,len(x)])
		for i in range(deg+1):
			self.M[i,:]=x**(deg-i)

		self.MC = pl.dot(self.M,self.icov)
		self.MCM = pl.dot(self.M,pl.transpose(self.MC))

		self.fit()

	def fit(self):
		MCy = pl.dot(self.MC,self.y)
		iMCM = pl.linalg.inv(self.MCM)
		self.pars = pl.dot(iMCM,MCy)
		self.covpar = iMCM
		
	def yfit(self):
		return pl.dot(pl.transpose(self.M),self.pars)

	def covpar():
		return self.covpar

