import numpy as np
import random as rand
import scipy.special as special
import scipy.optimize as opt
from general import *

def rand_discrete(alpha,xmin):
    x2 = float(xmin)
    r = rand.random()
    while True:
        x1 = x2
        x2 = 2*x1
        if (discrete_CCDF(x2,alpha,xmin) < 1-r):
            break
    while (x2-x1>=1):
        if (discrete_CCDF(x1+ ((x2-x1)/2),alpha,xmin) < 1-r):
            x2 = x2 - ((x2-x1)/2)
        else:
            x1 = x1 + ((x2-x1)/2)
    return int(x1)

def rand_continuous(alpha,xmin):
    r = rand.random()
    return  xmin*(1-r)**(-1/(alpha-1))

def discrete_neg_log_likelihood(a,xmin,xs):
    return float(len(xs)*np.log(special.zeta(a,xmin)) + a*sum([np.log(x) for x in xs]))

def continuous_max_likelihood(xs,xmin):
    return 1.0 + len(xs)*(1/sum([np.log(float(x)/xmin) for x in xs]))

def continuous_CCDF(x,alpha,xmin):
    return (float(x)/xmin)**(-alpha + 1.0)
    
def discrete_CCDF(x,alpha,xmin):
    return special.zeta(alpha,x)/special.zeta(alpha,xmin)

def discrete_PDF(x,alpha,xmin):
    return 1./((x**alpha)*special.zeta(alpha,xmin))

def discrete_exponent_estimator(xs,xmin=1,a0=2.5):
    """ Returns the exponent of the power law of the degree distribution
    or at least attempts to find one.

    If the degree distribution follows the power law

        p(x) \prop x^-a

    where 2 < a < 3 generally.[1]

        '... one can estimate a by direct numeric maximization of the
        likelihood function itself, or equivalently of its logarithm
        (which is usually simpler):

        L(a) = -n ln(zeta(a,xmin)) - a sum(ln(xi))'[1]

    in the discrete case.

    We actually minimize the the -log likelihood as this is the same

    While this function will actually return the exponent for the graph
    this doesn't necessarily mean the data has a power law
    distribution. Other statistical tests should be run to build evidence
    for that hypothesis.
    
    Parameters
    ----------
    xs   : list of collected values
    xmin : minimum value for which to estimate power law value
    a0   : starting guess at exponent

    Returns
    ----------
    a : float
        The estimation of the scaling parameter

    [1] A. Clauset, C.R. Shalizi, and M.E.J. Newman,
        'Power-law distributions in empirical data'
        SIAM Review 51(4), 661-703 (2009). (arXiv:0706.1062)

    A little explaining about this function opt.fmin. It takes a
    function that represents the function we are trying to minimize,
    in this case the negative log likelihood of the discrete power
    law function. It takes an initial guess 2.5, which is where most
    power laws are anyway and args which tells it what arguments to
    pass to the function we are optimizing besides a, and finally we
    turn the display off with disp=0
    """
    a = opt.fmin(discrete_neg_log_likelihood,a0,args=(xmin,xs),disp=0)
    return float(a)

def continuous_estimator(xs,a=None,xmin=None,a0=2.5):
    """ Return the minimum degree for which the degree distribution
    holds about the value.
    """
    numZeros = 0
    while True:
        try:
            xs.remove(0)
            numZeros+=1
        except ValueError:
            break
    
    if not xmin is None:
        alpha = continuous_max_likelihood(xs,xmin)
        xs_xmin = [x for x in xs if x>=xmin]
        Sxmin = comp_cum_distribution(xs_xmin)
        P = dict([(x,continuous_CCDF(x,alpha,xmin)) for x in Sxmin.keys()])
        D = KSStat(None,None,cdf_x=Sxmin,cdf_y=P)
        return (alpha,xmin,D)
    xmins = list(set(xs))
    xmins.sort()
    xmins = xmins[:-1]
    alphas={}
    Ds = {}
    xminhat = min(xmins)
    for xmin in xmins:
        xs_xmin = [x for x in xs if x>=xmin]
        if (a==None):
            alphas[xmin] = continuous_max_likelihood(xs_xmin,xmin)
        else:
            alphas[xmin] = a
        Sxmin = comp_cum_distribution(xs_xmin)
        P = dict([(x,continuous_CCDF(x,alphas[xmin],xmin)) for x in Sxmin.keys()])
        Ds[xmin] = KSStat(None,None,cdf_x=Sxmin,cdf_y=P)
        if (Ds[xmin] <= np.min(Ds.values())):
            xminhat = xmin 
    return (alphas[xminhat],xminhat,Ds[xminhat])

def discrete_estimator(xs,a=None,xmin=None,a0=2.5):
    """ Return the minimum degree for which the degree distribution
    holds about the value.
    """
    numZeros = 0
    while True:
        try:
            xs.remove(0)
            numZeros+=1
        except ValueError:
            break
    
    if not xmin is None:
        alpha = discrete_exponent_estimator(xs,xmin,a0)
        xs_xmin = [x for x in xs if x>=xmin]
        Sxmin = comp_cum_distribution(xs_xmin,data_range='ext_data')
        pdf_P = dict([(x,discrete_PDF(x,alpha,xmin)) for x in Sxmin.keys()])
        P = comp_cum_distribution(None,pdf=pdf_P)
#        P = dict([(x,discrete_CCDF(x,alpha,xmin)) for x in Sxmin.keys()])
        D = KSStat(None,None,cdf_x=Sxmin,cdf_y=P)
        return (alpha,xmin,D)
    xmins = sorted([x for x in list(set(xs)) if x < max(xs)*.9])
    alphas={}
    Ds = {}
    xminhat = min(xmins)
    for xmin in xmins:
        xs_xmin = [x for x in xs if x>=xmin]
        if a is None:
            alphas[xmin] = discrete_exponent_estimator(xs_xmin,xmin,a0)
        else:
            alphas[xmin] = a
        Sxmin = comp_cum_distribution(xs_xmin,data_range='ext_data')
        pdf_P = dict([(x,discrete_PDF(x,alphas[xmin],xmin)) for x in Sxmin.keys()])
        P = comp_cum_distribution(None,pdf=pdf_P)
        #P = dict([(x,discrete_CCDF(x,alphas[xmin],xmin)) for x in Sxmin.keys()])
        Ds[xmin] = KSStat(None,None,cdf_x=Sxmin,cdf_y=P)
        if (Ds[xmin] <= np.min(Ds.values())):
            xminhat = xmin
    #print Ds
    return (alphas[xminhat],xminhat,Ds[xminhat])

def discrete_power_law_test(xs,alpha=None,xmin=None,D=None,epsilon=.01,disp=False):
    numZeros = 0
    while True:
        try:
            xs.remove(0)
            numZeros+=1
        except ValueError:
            print 'Removed ' + str(numZeros) + ' Zeros.'
            break

    numTests = .25*(1/epsilon**2)
    if (alpha==None or xmin==None or D==None):
        alpha,xmin,D = discrete_estimator(xs)
    if disp:
        print "(alpha,xmin,D) = (" + str(alpha) + ", " + str(xmin) + ", "+ str(D) + ")"
    xs_lt_xmin = [x for x in xs if x<xmin]
    alpha_syn = {}
    xmin_syn = {}
    D_syn = {}
    print "Number of tests : " + str(int(numTests+.5))
    for n in range(int(numTests+.5)):
        if disp:
            print "Test : " + str(n)
        xs_synth = []
        for i in range(len(xs)):
            xminRand = rand.random()
            if (xminRand < float(len(xs_lt_xmin))/len(xs)):
                xs_synth.append(rand.choice(xs_lt_xmin))
            else:
                xs_synth.append(rand_discrete(alpha,xmin))
        alpha_syn[n],xmin_syn[n],D_syn[n] = discrete_estimator(xs_synth,a0=alpha)
        if disp:
            print "Synthetic (alpha,xmin,D) : (" + str(alpha_syn[n]) + ", " + str(xmin_syn[n]) + ", " + str(D_syn[n]) + ")"

    p = len([d for d in D_syn.values() if d>D])/float(numTests)
    if disp:
        if (p >= .1):
            print "There is evidence the provided data is a power law"
        else:
            print "There is evidence the provided data is not a power law"
    return (alpha,xmin,p)    

def continuous_power_law_test(xs,alpha=None,xmin=None,D=None,epsilon=.01,disp=False):
    numZeros = 0
    while True:
        try:
            xs.remove(0)
            numZeros+=1
        except ValueError:
            print 'Removed ' + str(numZeros) + ' Zeros.'
            break

    numTests = .25*(1/epsilon**2)
    if (alpha==None or xmin==None or D==None):
        alpha,xmin,D = continuous_estimator(xs)
    if disp:
        print "(alpha,xmin,D) = (" + str(alpha) + ", " + str(xmin) + ", "+ str(D) + ")"
    xs_lt_xmin = [x for x in xs if x<xmin]
    alpha_syn = {}
    xmin_syn = {}
    D_syn = {}
    if disp:
        print "Number of tests : " + str(int(numTests+.5))
    for n in range(int(numTests+.5)):
        if disp:
            print "Test : " + str(n)
        xs_synth = []
        for i in range(len(xs)):
            xminRand = rand.random()
            if (xminRand < float(len(xs_lt_xmin))/len(xs)):
                xs_synth.append(rand.choice(xs_lt_xmin))
            else:
                xs_synth.append(rand_continuous(alpha,xmin))
        alpha_syn[n],xmin_syn[n],D_syn[n] = continuous_estimator(xs_synth,a0=alpha)
        if disp:
            print "Synthetic (alpha,xmin,D) : (" + str(alpha_syn[n]) + ", " + str(xmin_syn[n]) + ", " + str(D_syn[n]) + ")"
    p = len([d for d in D_syn.values() if d>D])/float(numTests)
    if disp:
        if (p >= .1):
            print "There is evidence the provided data is a power law"
        else:
            print "There is evidence the provided data is not a power law"
    return (alpha,xmin,p)
