from pylab import *
import numpy as np
import cosmolopy
from scipy import integrate
from scipy import interpolate
from Cosmology import pyxi
import cosmolopy.distance as cd
from Cosmology import cosmology

mycosmo=cosmolopy.fidcosmo.copy()
mycosmo['baryonic_effects']=True
mycosmo['h']=0.7
mycosmo['omega_M_0']=0.3
mycosmo['omega_lambda_0']=0.7
mycosmo['omega_k_0']=0.


print(cd.comoving_distance(0.001,**mycosmo))



print(cosmology.properdistance(np.array([0.001,0.002]))*3e5/70)