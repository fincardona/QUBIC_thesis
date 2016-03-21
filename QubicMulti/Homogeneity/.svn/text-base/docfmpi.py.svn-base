#!/usr/bin/env python
# copied and modified from John K Parejko mr_2pc_bin.py

import sys
import numpy as np
import time

from mpi4py import MPI

from pairs import mr_wpairs_wp,mr_wpairs
import pyfits


def do_counting(comm,rank,bins,data1,data2,counter,weight1=None,weight2=None):
    """Count pairs in bins between data1 and data2.
    To count a dataset against itself, you must pass data.copy() as data2.
    Counter determines which code is used for bin counting.
    Specify weights to weight points by that amount."""
    t1 = time.time()
    #print '%d of %d: Setting up...'%(comm.rank,comm.size)
    if counter == 'euclidean':
        wpairs = mr_wpairs.radial_wpairs(comm, data1, data2, w1=weight1, w2=weight2, minpart=200, minboxwidth=10.0, ntrunc=1000)
    elif counter == 'rppi':
        wpairs = mr_wpairs_wp.wp_wpairs(comm, data1, data2, w1=weight1, w2=weight2, minpart=200, minboxwidth=10.0, ntrunc=1000)
    t2 = time.time()
    #print '%d of %d: Setup time = %8.5f'%(comm.rank,comm.size,t2-t1)
    sys.stdout.flush()

    # TBD: it would be nice to get some more status information here, if possible
    # e.g. number of processors in use, processing balance between processors
    # a progress bar, etc.
    t1 = time.time()
    #print '%d of %d: Computing counts in bins...'%(comm.rank,comm.size)
    if counter == 'euclidean':
        counts = wpairs(bins)
    elif counter == 'rppi':
        counts = wpairs(bins[0],bins[1])
    t2 = time.time()
    #print '%d of %d: Processing time = %8.5f'%(comm.rank,comm.size,t2-t1)
    sys.stdout.flush()

    return counts
#...

def write_1d(bins,counts,outfilename):
    """Write the (1-dimensional) results to outfilename."""
    outfile = file(outfilename,'wb')
    header = ('Start','Stop','Count\n')
    outfile.write('\t'.join(header))
    for x in zip(bins[:-1],bins[1:],counts):
        outfile.write(('%10f\t%10f\t%10f\n')%x)
    #print 'Wrote results to:',outfilename
    sys.stdout.flush()
#...

def write_2d(bins,counts,outfilename):
    """Write the (2-dimensional) results to outfilename."""
    outfile = file(outfilename,'wb')
    header = [['#first row is rp bins'],]
    header.append(['#second row is pll bins'])
    header.append(['#subsequent rows are the grid of bin counts'])
    for h in header:
        outfile.write('\t'.join(h)+'\n')
    outfile.write('\t'.join([str(x) for x in bins[0]])+'\n')
    outfile.write('\t'.join([str(x) for x in bins[1]])+'\n')
    for c in counts:
        outfile.write('\t'.join([str(x) for x in c])+'\n')
    #print 'Wrote results to:',outfilename
    sys.stdout.flush()
#...

def main(argv=None):
    if argv is None: argv = sys.argv[1:]
    from optparse import OptionParser, OptionGroup

    usage = '%prog [OPTION] datafile1 [datafile2]'
    usage += '\nCompute the counts-in-bins for datafile1 vs. either itself, or datafile2 (if specified).'
    usage += '\n\nTo run with MPI (use multiple processors) use:'
    usage += '\n\tmpirun -np N_CPU python mr_2pc_bin.py [OPTS]'
    usage += '\nWhere N_CPU is the number of threads you want to run with.'
    parser = OptionParser(usage)

    parser.add_option('-o','--output',dest='output',default='results.dat',
                      help='File to write the results to (%default).')
    parser.add_option('-b','--bins',dest='binfile',default='distancebins.txt',
                      help='file containing the bins to compute pairs in (%default)')
    parser.add_option('-w',dest='weight',default=False,action='store_true',
                      help='Use the fourth colum of the data as a weight (%default).')
    counterChoices = ('euclidean','rppi')
    parser.add_option('--counter',dest='counter',default='euclidean',choices=counterChoices,
                      help='Which counting metric to use, from: ['+','.join(counterChoices)+'] (%default)')

    (opts,args) = parser.parse_args(argv)

    # read in the data file(s)
    try:
        datafile1 = args[0]
        try:
            #print 'Reading data 1: %s'%(datafile1,)
            #data1 = np.loadtxt(datafile1)
            data,hdr_data=pyfits.getdata(datafile1,header=True)
            nb=(data.field('x')).size
            data1=[]
            data1=np.append([data.field('w')],data1)
            data1=np.append([data.field('z')],data1)
            data1=np.append([data.field('y')],data1)
            data1=np.append([data.field('x')],data1)
            data1=data1.reshape(4,nb)
            data1=data1.transpose()
            data=0
            #print 'Finished reading data 1: %s'%(datafile1,)
        except IOError:
            parser.error('Could not find file: '+datafile1)
    except IndexError:
        parser.error('At least one data file must be specified!')
    sys.stdout.flush()
    try:
        datafile2 = args[1]
        try:
            #print 'Reading data 2: %s'%(datafile2,)
            #data2 = np.loadtxt(datafile2)
            data,hdr_data=pyfits.getdata(datafile2,header=True)
            nb=(data.field('x')).size
            data2=[]
            data2=np.append([data.field('w')],data2)
            data2=np.append([data.field('z')],data2)
            data2=np.append([data.field('y')],data2)
            data2=np.append([data.field('x')],data2)
            data2=data2.reshape(4,nb)
            data2=data2.transpose()
            data=0
            #print 'Finished reading data 2: %s'%(datafile2,)
        except IOError:
            parser.error('Could not find file: '+datafile2)
    except IndexError:
        # if we don't get the 2nd file, use a copy of data1
        data2 = data1.copy()
    sys.stdout.flush()


    if opts.weight:
        weight1 = data1.T[3]
        data1 = data1.T[:3].T
        weight2 = data2.T[3]
        data2 = data2.T[:3].T
    else:
        weight1 = None
        weight2 = None

    try:
        if opts.counter == 'euclidean':
            # The bins should be a 1d array out of loadtxt by default.
            bins=np.loadtxt(opts.binfile)
        elif opts.counter == 'rppi':
            # in this case, assume the bins have two components.
            #infile=file(opts.binfile,'r')
            #bins = []
            #print opts.binfile
            #bins.append(np.array(infile.readline().split(' '),dtype='float'))
            #print 'bins1'
            #print bins
            #bins.append(np.array(infile.readline().split(' '),dtype='float'))
            toto=np.loadtxt(opts.binfile)
            bins=[toto[:,0],toto[:,1]]
    except IOError:
        parser.error('Could not find file: '+opts.binfile)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    counts = do_counting(comm,rank,bins,data1,data2,opts.counter,weight1,weight2)
    # We should only write the results to the output file one time.
    if comm.rank == 0:
        if opts.counter == 'euclidean':
            write_1d(bins,counts,opts.output)
        elif opts.counter == 'rppi':
            write_2d(bins,counts,opts.output)
    MPI.Finalize()
#...

if __name__ == "__main__":
    sys.exit(main())
