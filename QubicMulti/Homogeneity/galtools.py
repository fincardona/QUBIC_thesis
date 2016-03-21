from Cosmology import cosmology
from pylab import *
import numpy as np
import pyfits
import string
import random
import subprocess
import os
import glob
from scipy import integrate
from scipy import interpolate
from scipy import linalg
import emcee
import scipy.optimize as opt
import SplineFitting
from Homogeneity import numMath

host = os.environ['HOST']
if host[0:6] == 'dapint':
    docfmpi = "/home/usr202/mnt/burtin/Cosmo/analyse/python_from_jch/Homogeneity/docfmpi.py"
    mpirun = "mpirun"
elif host == 'MacBook-Pro-de-Jean-Christophe.local' or host == 'apcmc191.in2p3.fr' :
    docfmpi = "/Users/hamilton/idl/pro/SDSS-APC/python/Homogeneity/docfmpi.py"
    mpirun = "openmpirun"
elif host == 'apcmcqubic' :
    docfmpi = "/Users/hamilton/idl/pro/SDSS-APC/python/Homogeneity/docfmpi.py"
    mpirun = "/opt/local/bin/openmpirun"
else :
    message = "********* in galtools.py, unknown host : " + host + "  *******************"
    print(message)

def meancut(array,nsig=3,niter=3):
    thearray=array
    for i in np.arange(niter):
        m=np.mean(thearray)
        s=np.std(thearray)
        w=np.where(np.abs(thearray-m) <= nsig*s)
        thearray=thearray[w[0]]

    return(m,s)
    
def cov2cor(mat):
    cor=np.zeros((mat.shape[0],mat.shape[1]))
    for i in np.arange(mat.shape[0]):
        for j in np.arange(mat.shape[1]):
            cor[i,j]=mat[i,j]/np.sqrt(mat[i,i]*mat[j,j])

    return(cor)

def pick_indices(done,nfiles):
    okrand=False
    icount=0
    while okrand is False:
        thenumrand=int(np.floor(random.random()*nfiles))
        if thenumrand not in done:
            okrand=True
            numrand=thenumrand
            #print('OK for ',numrand)
        else:
            #print('    ',thenumrand,' was already in the list')
            okrand=False
            icount=icount+1
        if icount >= nfiles:
            print('Error in pick_indices: Cannot find enough free indices')
            stop
    return numrand


def read_pthalo_data(file,zmin=0.43,zmax=0.7):
    data=np.loadtxt(file)
    wok=np.where(data[:,2] >= zmin)
    data=data[wok[0],:]
    wok=np.where(data[:,2] <= zmax)
    data=data[wok[0],:]

    datawboss=data[:,4]    #weight from target completeness
    datawcp=data[:,5]      #weight from close pairs
    datawred=data[:,6]     #redshift completeness weight
    w=np.where(datawboss*datawcp*datawred > 0)
    data=data[w]
    datara=data[:,0]
    datadec=data[:,1]
    dataz=data[:,2]
    datawboss=data[:,4]    #weight from target completeness
    datawcp=data[:,5]      #weight from close pairs
    datawred=data[:,6]     #redshift completeness weight
    dataw=(datawcp+datawred-1)   # this is the weight to be used (does not include FKP)
    return datara,datadec,dataz,dataw

def read_pthalo_random(randfiles,nrandomreq,zmin=0.43,zmax=0.7):
    nfiles=np.size(randfiles)
    randra=np.array([])
    randdec=np.array([])
    randz=np.array([])
    randw=np.array([])
    while np.size(randra) < nrandomreq :
        # Pick one
        done=[]
        randnum=pick_indices(done,nfiles)
        done.append(randnum)
        #print('Reading '+randfiles[randnum])
        therandra,theranddec,therandz,therandw=read_pthalo_data(randfiles[randnum],zmin=zmin,zmax=zmax)
        randra=np.concatenate((randra,therandra))
        randdec=np.concatenate((randdec,theranddec))
        randz=np.concatenate((randz,therandz))
        randw=np.concatenate((randw,therandw))

    return randra,randdec,randz,randw


def random_string(nchars):
    lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(nchars)]
    str = "".join(lst)
    return(str)


def rthph2xyz(r,th,ph):
    """
    for r, theta (rad) and phi (rad), returns the x,y,z in Euclidean coordinates
    """
    x=r*np.sin(th)*np.cos(ph)
    y=r*np.sin(th)*np.sin(ph)
    z=r*np.cos(th)
    return(x,y,z)


def rthphw2fits(fitsname,r,th,ph,w=None):
    """
    for r, theta (rad) and phi (rad) and optionnaly w (weights), writes them in euclidean coordinates into a fits file with name fitsname
    """
    x,y,z=rthph2xyz(r,th,ph)
    nb=x.size
    if w is None: w=np.ones(nb)
    
    #weights need to be normalized to an average of 1
    w=w/np.mean(w)

    #Writing Fits file
    col0=pyfits.Column(name='x',format='E',array=x)
    col1=pyfits.Column(name='y',format='E',array=y)
    col2=pyfits.Column(name='z',format='E',array=z)
    col3=pyfits.Column(name='w',format='E',array=w)
    cols=pyfits.ColDefs([col0,col1,col2,col3])
    tbhdu=pyfits.new_table(cols)
    tbhdu.writeto(fitsname,clobber=True)
    return(x,y,z,w)


def run_kdtree(datafile1,datafile2,binsfile,resfile,nproc=None):
    if nproc is None:
        subprocess.call(["python",
                         docfmpi,
                         "--counter=euclidean",
                         "-b",binsfile,
                         "-o",resfile,
                         "-w",
                         datafile1,
                         datafile2])
    else:
        subprocess.call([mpirun,
                         "-np",str(nproc),
                         "python",
                         docfmpi,
                         "--counter=euclidean",
                         "-b",binsfile,
                         "-o",resfile,
                         "-w",
                         datafile1,
                         datafile2])


def read_pairs(file,normalize=True):
    bla=np.loadtxt(file,skiprows=1)
    r=bla[:,0]
    dd=bla[:,1]
    rr=bla[:,2]
    dr=bla[:,3]
    f=open(file)
    a=f.readline()
    ng=int(a.split("=")[1].split(" ")[0])
    nr=int(a.split("=")[2].split(" ")[0])
    f.close()
    if normalize is True:
        dd=dd/(ng*(ng-1)/2)
        rr=rr/(nr*(nr-1)/2)
        dr=dr/(ng*nr)

    return r,dd,rr,dr,ng,nr


def combine_regions(file1,file2,normalize=True):
    r1,dd1,rr1,dr1,ng1,nr1=read_pairs(file1,normalize=False)
    r2,dd2,rr2,dr2,ng2,nr2=read_pairs(file2,normalize=False)
    ng=ng1+ng2
    nr=nr1+nr2
    dd=dd1+dd2
    rr=rr1+rr2
    dr=dr1+dr2
    if normalize is True:
        dd=dd/(ng*(ng-1)/2)
        rr=rr/(nr*(nr-1)/2)
        dr=dr/(ng*nr)

    return r1,dd,rr,dr,ng,nr


def read_many_pairs(files):
    r,dd,rr,dr,ng,nr=read_pairs(files[0])
    nsim=np.size(files)
    nbins=np.size(r)

    all_ls=np.zeros((nsim,nbins))
    i=0
    for file in files:
        r,dd,rr,dr,ng,nr=read_pairs(file)
        ls=(dd-2*dr+rr)/rr
        all_ls[i,:]=ls
        i=i+1

    meanls,sigls,covmat,cormat=average_realisations(all_ls)
    return(r,meanls,sigls,covmat,cormat)

def read_many_pairs_combine(filesS,filesN):
    r,dd,rr,dr,ng,nr=read_pairs(filesS[0])
    nsim=np.size(filesN)
    nbins=np.size(r)
    
    all_ls=np.zeros((nsim,nbins))
    i=0
    for fileN,fileS in zip(filesN,filesN):
        r,dd,rr,dr,ng,nr=combine_regions(fileN,fileS)
        ls=(dd-2*dr+rr)/rr
        all_ls[i,:]=ls
        i=i+1

    meanls,sigls,covmat,cormat=average_realisations(all_ls)
    return(r,meanls,sigls,covmat,cormat)



def get_pairs(rdata,thdata,phdata,rrandom,thrandom,phrandom,rmin,rmax,nbins,wdata=None,wrandom=None,nproc=None,log=None):
    """
    dd,rr,dr=get_pairs(rdata,thdata,phidata,rrandom,thrandom,phirandom,rmin,rmax,nbins,wdata=None,wrandom=None,nproc=None,log=None)
    """
    # Need a random string for temporary files
    rndstr=random_string(10)

    # Prepare filenames
    datafile="/tmp/data_"+rndstr+".fits"
    randomfile="/tmp/random_"+rndstr+".fits"
    binsfile="/tmp/bins_"+rndstr+".txt"
    ddfile="/tmp/dd_"+rndstr+".txt"
    rrfile="/tmp/rr_"+rndstr+".txt"
    drfile="/tmp/dr_"+rndstr+".txt"

    # write fits files with data and randoms
    xd,yd,zd,wd=rthphw2fits(datafile,rdata,thdata,phdata,w=wdata)
    xr,yr,zr,wr=rthphw2fits(randomfile,rrandom,thrandom,phrandom,w=wrandom)
    # write bins file
    if log is None:
        edges=np.linspace(rmin,rmax,nbins+1)
    else:
        edges=10.**(np.linspace(np.log10(rmin),np.log10(rmax),nbins+1))

    outfile=open(binsfile,'w')
    for x in edges:
        outfile.write("%s\n" % x)

    outfile.close()

    # do DD
    print('       - Doing DD : '+str(xd.size)+' elements')
    run_kdtree(datafile,datafile,binsfile,ddfile,nproc=nproc)
    alldd=np.loadtxt(ddfile,skiprows=1)
    # do RR
    print('       - Doing RR : '+str(xr.size)+' elements')
    run_kdtree(randomfile,randomfile,binsfile,rrfile,nproc=nproc)
    allrr=np.loadtxt(rrfile,skiprows=1)
    # do DR
    print('       - Doing DR : '+str(xd.size)+'x'+str(xr.size)+' pairs')
    run_kdtree(datafile,randomfile,binsfile,drfile,nproc=nproc)
    alldr=np.loadtxt(drfile,skiprows=1)

    dd=alldd[:,2]
    rr=allrr[:,2]
    dr=alldr[:,2]

    # correct DD and RR for double counting
    dd=dd/2
    rr=rr/2

    r=(alldd[:,0]+alldd[:,1])/2
    subprocess.call(["rm","-f",datafile,randomfile,binsfile,ddfile,rrfile,drfile])
    return(r,dd,rr,dr)


def landy_szalay(rdata,thdata,phdata,rrandom,thrandom,phrandom,rmin,rmax,nbins,wdata=None,wrandom=None,nproc=None,log=None):
    """
    Gives the Landy-Szalay estimator for 2pts correlation function
    r,cf=landy_szalay(rdata,thdata,phdata,rrandom,thrandom,phrandom,rmin,rmax,nbins,wdata=None,wrandom=None,nproc=None,log=None)
    """
    # normalize the results
    ng=rdata.size
    nr=rrandom.size
    dd=dd/(ng*(ng-1)/2)
    rr=rr/(nr*(nr-1)/2)
    dr=dr/(ng*nr)

    r,dd,rr,dr=get_pairs(rdata,thdata,phdata,rrandom,thrandom,phrandom,rmin,rmax,nbins,wdata=wdata,wrandom=wrandom,nproc=nproc,log=log)
    return(r,(dd-2*dr+rr)/rr)
    

def paircount_data_random(datara,datadec,dataz,randomra,randomdec,randomz,cosmo_model,rmin,rmax,nbins,nproc=None,log=None,file=None,wdata=None,wrandom=None):
    """
    r,dd,rr,dr=paircount_data_random(datara,datadec,dataz,randomra,randomdec,randomz,cosmo,rmin,rmax,nbins,log=None,file=None,wdata=None,wrandom=None)
    Returns the R,DD,RR,DR for a given set of data and random and a given cosmology
    """

    # calculate proper distance for each object (data and random)
    pi=np.pi
    params=cosmo_model[0:4]
    h=cosmo_model[4]
    rdata=cosmology.get_dist(dataz,type='prop',params=params,h=h)*1000*h
    thdata=(90-datadec)*pi/180
    phdata=datara*pi/180
    rrandom=cosmology.get_dist(randomz,type='prop',params=params,h=h)*1000*h
    thrandom=(90-randomdec)*pi/180
    phrandom=randomra*pi/180
    # Count pairs
    r,dd,rr,dr=get_pairs(rdata,thdata,phdata,rrandom,thrandom,phrandom,rmin,rmax,nbins,nproc=nproc,log=log,wdata=wdata,wrandom=wrandom)

    # Write to file if required
    if file is not None:
        outfile=open(file,'w')
        outfile.write("Ng=%s Nr=%s \n" % (np.size(datara), np.size(randomra)))
        for xr,xdd,xrr,xdr in zip(r,dd,rr,dr):
            outfile.write("%s %s %s %s\n" % (xr,xdd,xrr,xdr))

    outfile.close()

    # return result
    return(r,dd,rr,dr)


def scalednr(dd,rr,bias):
    ddint=dd*0
    rrint=rr*0
    ddint[0]=dd[0]
    rrint[0]=rr[0]
    for i in np.arange(dd.size-1)+1:
        ddint[i]=ddint[i-1]+dd[i]
        rrint[i]=rrint[i-1]+rr[i]

    result=np.ndarray(ddint.size)
    result=((ddint/rrint-1)/bias**2)+1
    return(result)

def rhomo_nr(r,nrvec):
    w=np.where(r >= 10)
    ther=r[w]
    thenrvec=nrvec[w]
    f=interpolate.interp1d(thenrvec[::-1],ther[::-1],bounds_error=False)
    return(f(1.01))


def d2(r,dd,rr,bias):
    nr=scalednr(dd,rr,bias)
    dlogr=np.log(r[1])-np.log(r[0])
    d2=np.gradient(np.log(nr),dlogr)+3
    return(d2)


def rhomo_d2(r,d2):
    w=np.where(r >= 10)
    f=interpolate.interp1d(d2[w],r[w])
    return(f(2.97))


def loop_mocks(dirbase,region,outdir,objectFileName='cmass_dr11*.wghtv.txt',weightDirName='/Users/pntelis/Volumes/Data/SDSS/DR11/LRG/',nprocfile=False,ratiorandom=5,rmin=0,rmax=200,nbins=50,log=None,cosmo=[0.27,0.73,-1,0,0.7],nproc=8,nmax=False,zmin=0.43,zmax=0.7):

    dat_ext='Data_'+region+'/'
    ran_ext='Random_'+region+'/'
    datafiles=glob.glob(dirbase+dat_ext+objectFileName)
    randfiles=glob.glob(dirbase+ran_ext+objectFileName)
    donefile=outdir+'donefile_'+region+'.txt'
    subprocess.call(["mkdir",outdir])
    subprocess.call(["mkdir",outdir+region])
    subprocess.call(["touch",donefile])
    
    if nprocfile is False:
        nprocfile=outdir+'nproc_'+region+'.txt'
        if not os.path.isfile(nprocfile):
            file=open(nprocfile,'w')
            file.write("%s\n" % (nproc))
            file.close()

    if nmax is False:
        nmax=np.size(datafiles)

    for thefile in datafiles[:nmax-1]:
        print('Doing region '+region+' '+str(zmin)+' < z < '+str(zmax)+' - file = '+os.path.basename(thefile))
        list_done=np.loadtxt(donefile,dtype='string')
        if 'pairs_'+os.path.basename(thefile) in list_done:
            print('    File was already done according to \n    '+donefile)
        else:
            # read data
            print('    * reading data')
            datara,datadec,dataz,dataw=read_pthalo_data(thefile,zmin=zmin,zmax=zmax)
            ndata=datara.size
            print('      read '+str(ndata)+' objects')
            
            # read random
            nrandomreq=ratiorandom*ndata
            nfiles=np.size(randfiles)
            print('    * reading random')
            randra,randdec,randz,randw=read_pthalo_random(randfiles,nrandomreq,zmin=zmin,zmax=zmax)
            nrandom=randra.size
            print('      read '+str(nrandom)+' objects')
            
            # calculate FKP weights
            if region is 'North':
                filenz=weightDirName+'nbar-cmass-dr11v1remake-N-Anderson.dat'
            else:
                filenz=weightDirName+'nbar-cmass-dr11v1remake-S-Anderson.dat'

            truc=np.loadtxt(filenz,skiprows=2)
            zcen=truc[:,0]
            wfkp=truc[:,4]
            f=interpolate.interp1d(zcen,wfkp)
            data_wfkp=f(dataz)
            rand_wfkp=f(randz)
            dataw=dataw*data_wfkp
            randw=randw*rand_wfkp
            # count pairs
            outfile=outdir+region+'/'+'pairs_'+os.path.basename(thefile)
            nproc=np.loadtxt(nprocfile,dtype='int')
            print('    * Counting pairs on '+str(nproc)+' threads')
            r,dd,rr,dr=paircount_data_random(datara,datadec,dataz,randra,randdec,randz,cosmo,rmin,rmax,nbins,log=log,file=outfile,nproc=nproc,wdata=dataw,wrandom=randw)
            list_done=np.append(list_done,os.path.basename(outfile))
            file=open(donefile,'w')
            for fnames in list_done:
                file.write("%s\n" % fnames)

            file.close()


def homogeneity_many_pairs(files,bias):
    r,dd,rr,dr,ng,nr=read_pairs(files[0])
    nsim=np.size(files)
    nbins=np.size(r)
    
    all_nr=np.zeros((nsim,nbins))
    all_d2=np.zeros((nsim,nbins))
    all_rhnr=np.zeros(nsim)
    all_rhd2=np.zeros(nsim)
    i=0
    for file in files:
        r,dd,rr,dr,ng,nr=read_pairs(file)
        thenr=scalednr(dd,rr,bias)
        all_nr[i,:]=thenr
        all_rhnr[i]=rhomo_nr(r,thenr)
        thed2=d2(r,dd,rr,bias)
        all_d2[i,:]=thed2
        all_rhd2[i]=rhomo_d2(r,thed2)
        i=i+1

    rhnr=np.mean(all_rhnr)
    rhd2=np.mean(all_rhd2)
    sigrhnr=np.std(all_rhnr)
    sigrhd2=np.std(all_rhd2)
    mean_nr,sig_nr,covmat_nr,cormat_nr=average_realisations(all_nr)
    mean_d2,sig_d2,covmat_d2,cormat_d2=average_realisations(all_d2)
    return(r,mean_nr,sig_nr,mean_d2,sig_d2,covmat_nr,covmat_d2,cormat_nr,cormat_d2,rhnr,rhd2,sigrhnr,sigrhd2,all_nr,all_d2)


def homogeneity_many_pairs_combine(filesN,filesS,bias):
    r,dd,rr,dr,ng,nr=read_pairs(filesN[0])
    nsim=np.size(filesN)
    nbins=np.size(r)
    
    all_nr=np.zeros((nsim,nbins))
    all_d2=np.zeros((nsim,nbins))
    all_rhnr=np.zeros(nsim)
    all_rhd2=np.zeros(nsim)
    i=0
    for fileN,fileS in zip(filesN,filesS):
        r,dd,rr,dr,ng,nr=combine_regions(fileN,fileS)
        thenr=scalednr(dd,rr,bias)
        all_nr[i,:]=thenr
        all_rhnr[i]=rhomo_nr(r,thenr)
        thed2=d2(r,dd,rr,bias)
        all_d2[i,:]=thed2
        all_rhd2[i]=rhomo_d2(r,thed2)
        i=i+1
    
    rhnr=np.mean(all_rhnr)
    rhd2=np.mean(all_rhd2)
    sigrhnr=np.std(all_rhnr)
    sigrhd2=np.std(all_rhd2)
    mean_nr,sig_nr,covmat_nr,cormat_nr=average_realisations(all_nr)
    mean_d2,sig_d2,covmat_d2,cormat_d2=average_realisations(all_d2)
    return(r,mean_nr,sig_nr,mean_d2,sig_d2,covmat_nr,covmat_d2,cormat_nr,cormat_d2,rhnr,rhd2,sigrhnr,sigrhd2,all_nr,all_d2)


def average_realisations(datasim):
    dims=np.shape(datasim)
    nsim=dims[0]
    nbins=dims[1]
    meansim=np.zeros(nbins)
    sigsim=np.zeros(nbins)
    for i in np.arange(nbins):
        meansim[i]=np.mean(datasim[:,i])
        sigsim[i]=np.std(datasim[:,i])
    
    covmat=np.zeros((nbins,nbins))
    for i in np.arange(nbins):
        for j in np.arange(nbins):
            covmat[i,j]=np.mean((datasim[:,i]-meansim[i])*(datasim[:,j]-meansim[j]))
    
    cormat=np.zeros((nbins,nbins))
    for i in np.arange(nbins):
        for j in np.arange(nbins):
            cormat[i,j]=covmat[i,j]/np.sqrt(covmat[i,i]*covmat[j,j])

    return(meansim,sigsim,covmat,cormat)



###### Fitting d2 ###################################################

# polynomial fitting with inverse covariance matrix
def lnprobcov(thepars, xvalues, yvalues, invcovmat):
    pol=np.poly1d(thepars)
    delta=yvalues-pol(xvalues)
    chi2=np.dot(np.dot(np.transpose(delta),invcovmat),delta)
    return(-chi2)

# polynomial model
def polymodel(x,*params):
    thep=np.poly1d(params)
    return(thep(x))

def get_rh_mcmc(x,y,cov,thresh=2.97,poldeg=5,xstart=30,xstop=500,nburn=1000,nbmc=1000,nthreads=0,doplot=True,diagonly=False):
    # get desired sub array
    w=np.where((x >= xstart) & (x <= xstop))
    thex=x[w]
    they=y[w]
    theyerr=np.sqrt(cov[w[0],w[0]])
    thecov=(cov[w[0],:])[:,w[0]]
    theinvcov=np.array(np.matrix(thecov).I)
    if diagonly:
        print('Using only diagonal part of the covariance matrix')
        theinvcov=zeros((np.size(w),np.size(w)))
        theinvcov[np.arange(np.size(w)),np.arange(np.size(w))]=1./theyerr**2
                
    # Simple polynomial fitting (no errors)
    polfit=np.polyfit(thex,they,poldeg)
    # 
    polfit2,parscov=opt.curve_fit(polymodel,thex,they,p0=polfit,sigma=theyerr)
    err_polfit2=sqrt(diagonal(parscov))

    nok=0
    while nok <= nbmc/2:
        ######################### MCMC using emcee #############################################
        nok=0
        ndim=poldeg+1
        nwalkers=ndim*2
        print('\nStart emcee with '+np.str(ndim)+' dimensons and '+np.str(nwalkers)+' walkers')
        # initial guess
        p0=emcee.utils.sample_ball(polfit,err_polfit2*3,nwalkers)
        # Initialize emcee
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobcov, args=[thex,they,theinvcov],threads=nthreads)
        # Burn out
        print('   - Burn-out with:')
        pos=p0
        okburn=0
        niterburn=0
        while okburn == 0:
            pos, prob, state = sampler.run_mcmc(pos, nburn)
            niterburn=niterburn+nburn
            chains=sampler.chain
            sz=chains[0,:,0].size
            largesig=np.zeros([nwalkers,ndim])
            smallsig=np.zeros([nwalkers,ndim])
            for j in arange(ndim):
                for i in arange(nwalkers):
                    largesig[i,j]=np.std(chains[i,sz-nburn:sz-101,j])
                    smallsig[i,j]=np.std(chains[i,sz-100:sz-1,j])
    
            ratio=largesig/smallsig
            bestratio=zeros(ndim)
            for i in arange(ndim):
                bestratio[i]=ratio[:,i].min()
                    
            worsebestratio=bestratio.max()
            wbest=np.where(bestratio == worsebestratio)
            print('     niter='+np.str(niterburn)+' : Worse ratio for best walker :'+np.str(worsebestratio))
            if worsebestratio < 2:
                okburn=1
                print('     OK burn-out done')
        
        sampler.reset()
        # now run MCMC
        print('   - MCMC with '+np.str(nbmc)+' iterations')
        sampler.run_mcmc(pos, nbmc)
        # find chain for best walker
        chains=sampler.chain
        fractions=sampler.acceptance_fraction
        #########################################################################################
        frac_threshold=0.4
        print('     Best fraction: '+np.str(max(fractions)))
        wfrac=where(fractions >= frac_threshold)
        print('     '+np.str(np.size(wfrac))+' walkers are above f='+np.str(frac_threshold))
        if max(fractions) > frac_threshold:
            best=np.where(fractions == max(fractions))
            #bestwalker=best[0]
            #thechain=chains[bestwalker[0],:,:]
            thechain=chains[wfrac[0],:,:]
            sp=np.shape(thechain)
            thechain=np.reshape(thechain,[sp[0]*sp[1],sp[2]])
            # find roots
            nelts=sp[0]*sp[1]
            vals=np.zeros(nelts)
            for i in np.arange(nelts):
                roots=(np.poly1d((thechain[i,:]).flatten())-thresh).r
                w0=np.where((roots > np.min(x)) & (roots < np.max(x)) & (np.imag(roots)==0))
                if np.size(w0)==1:
                    vals[i]=np.min((np.real(roots[w0])).flatten())

            wok=where(vals != 0)
            nok=np.size(wok)

        if nok < nbmc/2:
            print('       -> chain was not good (nok='+np.str(nok)+')... retrying...')

    meanrh=np.mean(vals[wok])
    sigrh=np.std(vals[wok])

    bla=zeros((x.size,nelts))
    for i in arange(nelts):
        aa=np.poly1d(thechain[i,:].flatten())
        bla[:,i]=aa(x)

    avpol=zeros(x.size)
    sigpol=zeros(x.size)
    for i in arange(x.size):
        avpol[i]=np.mean(bla[i,:])
        sigpol[i]=np.std(bla[i,:])
    
    #show a plot if needed
    if(doplot):
        clf()
        subplot(2,1,1)
        xlim(min(thex)*0.9,max(thex)*1.05)
        ylim(min(they)-(3-min(they))*0.1,3+(3-min(they))*0.1)
        xscale('log')
        
        plot([xstart,xstart],[-10,10],'--',color='black')
        plot([xstop,xstop],[-10,10],'--',color='black')
        plot(x,x*0+3,'--',color='black')
        plot(x,x*0+thresh,'--',color='red')
        plot(x,avpol,color='g',label='Average polynomial (degree='+np.str(poldeg)+')')
        plot(x,avpol+sigpol,color='g',ls=':')
        plot(x,avpol-sigpol,color='g',ls=':')

        errorbar(thex,they,yerr=theyerr,fmt='ro',label='Data')
        errorbar(meanrh,thresh,xerr=sigrh,fmt='bo',label='$R_H$ = '+str('%.1f'%meanrh)+' $\pm$ '+str('%.1f'%sigrh)+' $h^{-1}.\mathrm{Mpc}$')
        xlabel('r [$h^{-1}.\mathrm{Mpc}$]')
        ylabel('$d_2(r)$')
        legend(loc='lower right')
        subplot(2,1,0)
        hist(vals[wok],100)
        xlabel('Homogegeity scale [$h^{-1}.\mathrm{Mpc}$]')

    del(sampler)
    # return R_H
    returnchains=thechain[wok[0],:]
    print('Fit OK : R_H = '+str('%.1f'%meanrh)+' \pm '+str('%.1f'%sigrh)+' h^{-1}.\mathrm{Mpc}')
    return(meanrh,sigrh,vals[wok[0]],returnchains)




def get_rh_spline(x,y,cov,thresh=2.97,nbspl=12,xstart=30,xstop=500,doplot=True,diagonly=False,logspace=True):
    # get desired sub array
    w=np.where((x >= xstart) & (x <= xstop))
    thex=x[w]
    they=y[w]
    theyerr=np.sqrt(cov[w[0],w[0]])
    thecov=(cov[w[0],:])[:,w[0]]
    #theinvcov=np.array(np.matrix(thecov).I)
    theinvcov = numMath.mySVD(thecov)
    ###theinvcov = linalg.pinv2(thecov)
    
    print 'the shape of thecov   ', thecov.shape
    print 'the shape of theinvcov', theinvcov.shape
    print 'the Identity is:', thecov.dot(theinvcov)

    if diagonly:
        print('Using only diagonal part of the covariance matrix')
        theinvcov=zeros((np.size(w),np.size(w)))
        theinvcov[np.arange(np.size(w)),np.arange(np.size(w))]=1./theyerr**2
    

    # Fit with splines
    print(SplineFitting.__init__)
    spl=SplineFitting.MySplineFitting(thex,they,thecov,nbspl,logspace=logspace)
 
    # rh
    newx=linspace(thex.min(),thex.max(),1000)
    ff=interpolate.interp1d(spl(newx),newx)
    rh=ff(2.97)

    # Error on rh
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
    drh=err_on_funct/deriv_spl(rh)


    #show a plot if needed
    if(doplot):
        clf()
        xlim(min(thex)*0.9,max(thex)*1.05)
        ylim(2.85,3.01)
        plot(x,x*0+3,'--',color='black')
        plot(x,x*0+thresh,'--',color='red')
        plot(newx,spl(newx),color='b',lw=2,label='Best Fit Spline ('+np.str(nbspl)+' nodes): $\chi^2/ndf=$'+str('%.1f'%spl.chi2)+'/'+np.str(np.size(thex)-nbspl))
        errorbar(thex,they,yerr=theyerr,fmt='ko',label='Data')
        errorbar(rh,thresh,xerr=drh,fmt='ro',label='$R_H$ = '+str('%.1f'%rh)+' $\pm$ '+str('%.1f'%drh)+' $h^{-1}.\mathrm{Mpc}$')
        xlabel('r [$h^{-1}.\mathrm{Mpc}$]')
        ylabel('$d_2(r)$')
        legend(loc='lower right',frameon=False)

    print('Fit OK : R_H = '+str('%.1f'%rh)+' \pm '+str('%.1f'%drh)+' h^{-1}.\mathrm{Mpc}')
    return(rh,drh,spl)

def read_data(datafile,bias=2,combine=False):
    # Read data
    if combine is False:
        rd,dd,rr,dr,ngal,nrand=read_pairs(datafile)
    else:
        rd,dd,rr,dr,ngal,nrand=combine_regions(datafile[0],datafile[1])
    
    # calculate n(r) and d2(r)
    n_r=scalednr(dd,rr,bias)
    d2_r=d2(rd,dd,rr,bias)
    return(rd,d2_r)

def read_mocks(mockdir,bias=2,pairfiles='pairs_*.txt',combine=False):
    # Read Mocks to get covariance matrix
    if mockdir is None:
        covmatd2=np.zeros((np.size(rd),np.size(rd)))
        covmatd2[np.arange(np.size(rd)),np.arange(np.size(rd))]=1e-4
    else:
        if combine is False:
            mockfiles=glob.glob(mockdir+pairfiles)
            r,mean_nr,sig_nr,mean_d2,sig_d2,covmatnr,covmatd2,cormatnr,cormatd2,rhnr,rhd2,sigrhnr,sigrhd2,all_nr,all_d2=homogeneity_many_pairs(mockfiles,bias)
        else:
            mockfiles0=glob.glob(mockdir[0]+pairfiles)
            mockfiles1=glob.glob(mockdir[1]+pairfiles)
            nbcommon=min([size(mockfiles0),size(mockfiles1)])
            r,mean_nr,sig_nr,mean_d2,sig_d2,covmatnr,covmatd2,cormatnr,cormatd2,rhnr,rhd2,sigrhnr,sigrhd2,all_nr,all_d2=homogeneity_many_pairs_combine(mockfiles0[:nbcommon],mockfiles1[:nbcommon],bias)
    
    return(covmatd2,rhd2,all_nr,all_d2,r)


def read_datamocks(datafile,mockdir,bias=2.,combine=False):
    rd,d2_r=read_data(datafile,bias=bias,combine=combine)
    covmatd2,rhd2=read_mocks(mockdir,bias=bias,combine=combine)
    return(rd,d2_r,covmatd2,rhd2)

def getd2_datamocks(datafile,covmatd2,bias=2.,combine=False,deg=7,r0=30,nbmc=5000,nbspl=12,mcmc=False):
    #read data and mocks
    rd,d2_r=read_data(datafile,bias=bias,combine=combine)
    
    if mcmc==True:
        rha,drha,result,toto=get_rh_mcmc(rd,d2_r,covmatd2,poldeg=deg,xstart=r0,nbmc=nbmc)
    else:
        rha,drha,result=get_rh_spline(rd,d2_r,covmatd2,nbspl=nbspl,xstart=r0)
        print 'res from get_rh_spli',result
    
    return(rha,drha,result,rd,d2_r)


def NameZ(minz,maxz,nbins):
    # returns 2 arrays which contain
    # the edges of bins of redshift
    # in double and string array 
    
    zedge = np.linspace(minz,maxz,nbins+1)
    znames = np.array(np.zeros(nbins), dtype='|S20')
    
    for i in np.arange(nbins):
        znames[i]='z_'+str(zedge[i])+'_'+str(zedge[i+1])

    return(zedge,znames)

def corrCoeffd2(covmatd2,r,nbspl=9,name='',save=False):
    cc = np.ones((np.size(r),np.size(r)))

    for i in range(np.size(r)):
        for j in range(np.size(r)):
            cc[i][j] = covmatd2[i][j]/np.sqrt(covmatd2[i][i]*covmatd2[j][j])

    #X,Y = meshgrid(rS,rS)
    X,Y = r,r
    Z = cc

    fig, ax = plt.subplots()
    plt.ylabel('$r_i$ ($h^{-1}\ Mpc$)')
    plt.xlabel('$r_j$ ($h^{-1}\ Mpc$)')
    p = ax.pcolor(X, Y, Z, cmap=cm.RdBu, vmin=Z.min(), vmax=Z.max())
    cb = fig.colorbar(p, ax=ax)
    plt.suptitle(name+' at $0.626<$ z $<0.7$')
    if save == True:
        print 'Saving ...'
        plt.savefig('images/corr_z_0.646_0.7_'+name+'_nbspl_'+str(nbspl)+'.png',dpi=100)
