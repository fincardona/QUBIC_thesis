import numpy as np
from scipy import linalg

def mySVD(matrix,doCheck=0,kk=False):
    '''
    Singular Value decomposition                                   
    method to compute the inverse
    of matrix        
    option: doCheck=0,1,2
    0: return only the inverted SVD matrix
    1: return previous plus composite matrices
    2: return previous plus checks
           0,1                         ,2
    USAGE: s,sU,sUt,sV,sVh,sSig,sinvSig,sCheck = numMath.mySVD(a,doCheck=2)
    '''
    ### Compute the SVD parts
    M,N       = matrix.shape
    U,s,Vh    = linalg.svd(matrix)
    Sig       = linalg.diagsvd(s,M,N)

    V = np.matrix(Vh).H
    Ut = np.matrix(U).T
    
    # invSig = linalg.inv( np.matrix(Sig) )
    invSig = np.matrix(Sig).I
    
    ### Correct for ill-ness
    w = np.where(Sig<=10**(-14))
    invSig[w]=0.0
    
    ### Compute the Inverse of the matrix 
    invSVD = V.dot(invSig).dot(Ut)  

    ### and check the matrix
    checkMatrix = U.dot(Sig.dot(Vh))
    checkProduct = invSVD.dot(matrix)

    #print ' invSVD = \n', invSVD
    #print ' sCheck = \n', checkMatrix

    #print ' invSVD.dot(matrix)     = \n  ', checkProduct

    if(doCheck==0):
        return(invSVD)
    elif(doCheck==1):
        return(invSVD,U,Ut,V,Vh,Sig,invSig)
    elif(doCheck==2):
        if(kk==True):  
            print 'V.dot(Vh)= \n',V.dot(Vh)
            print 'U.dot(Ut)= \n',U.dot(Ut)  
            print 'Sig.dot(invSig)= \n',Sig.dot(invSig)
        return(invSVD,U,Ut,V,Vh,Sig,invSig,checkMatrix)

    else:
        print'Read the description of numMath.mySVD'
