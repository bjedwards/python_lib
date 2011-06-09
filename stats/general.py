import numpy as np

def prob_density_func(xs,norm=True,data_range='data'):
    """ Return the probability density function of a set of data xs.

    Parameters:
    -----------

    xs: list of numeric
        data to turn into pdf
    norm: Boolean
          Whether to normalize the data so it sums to 1, making it
          a density functin
    data_range: String or list of numeric
                May be set to any of three values
                'data': Indicates that only values that appear in
                        data, are keys in the returned dictionary
                        Note that this should be used in the
                        continuous data case.
                'ext_data': Indicates that all values between
                            the minimum and maximum value included
                            in the data are included. With 0, for
                            values which do not appear. Note that
                            this assumes discrete integer values,
                            use custom for other types
                other : If another data range is provided it is used
                        as keys for the data.
    Returns:
    --------
    pdf : dict of numeric
          probability density function of data returned as a dictionary
          keyed as set by data_range above.
    """
    if data_range=='data':
        dist_keys = set(xs)
    elif data_range=='ext_data':
        dist_keys = range(min(xs),max(xs)+1)
    else:
        dist_keys = data_range
    
    pdf = dict([(k,0.0) for k in dist_keys])
    for x in xs:
        pdf[x] += 1.0
    if norm:
        pdf.update([(k,pdf[k]/sum(pdf.values())) for k in pdf.keys()])
    return pdf

def cum_density_func(xs,norm=True,rank=False,data_range='data',pdf=None):
    """ Return the cumulative density function of a set of data xs

    Parameters:
    -----------

    xs: list of numeric
        Data for distribution to be calculated
    norm: Boolean
          If True normalize the data making it a true cumulative
          density function, if False keep raw frequency values
    rank: Boolean
          If the data is normalized, whether to do so by absolute
          total values or by rank. If by value, this can lead to
          stepped functions if data_range='ext_data' is selected
          where some values may not occur. If True data is
          normalized by its absolute rank in order. This is a
          relatively standard trick to make CDF's appear
          continuous
    data_range: String or list
                May be set to any of three values
                'data': Indicates that only values that appear in
                        data, are keys in the returned dictionary
                        Note that this should be used in the
                        continuous data case.
                'ext_data': Indicates that all values between
                            the minimum and maximum value included
                            in the data are included. With 0, for
                            values which do not appear. Note that
                            this assumes discrete integer values,
                            use custom for other types
                other : If another data range is provided it is used
                        as keys for the data.
    pdf : dict of numeric
          Pre computed pdf
    Returns:
    --------
    cdf: Dict of numeric keyed by data_range
         the cumulative distrbution function
    """
    if pdf is None:
        pdf = prob_density_func(xs,False,data_range)
    pdfk = sorted(pdf.keys())
    pdfv = map(pdf.get,pdfk)
    if not rank:
        cdfv = np.cumsum(pdfv)
        if norm:
            cdfv = cdfv/np.sum(pdfv)
    else:
        cdfv = np.arange(1,len(pdfk)+1)
        if norm:
            cdfv = cdfv/float((len(pdfk)+1))
    return dict(zip(pdfk,cdfv))

def comp_cum_distribution(xs,norm=True,rank=False,data_range='data',pdf=None):
    """Return the complement cumulative distribution(CCD) of a list dist

    Returns the CCD given by
            P(X>x) = 1 - P(X<=x)
    where P(X<=x) is the cumulative distribution func given by
            P(X <= x) = sum_{xi<x} p(xi)
    where p(xi) is the probaility density function calculated as
            p(xi) = xi/(sum_i xi)

    Parameters
    ----------
    dist : list of Numeric
           list of values representing the frequency of a value at
           the index occuring
    norm : Boolean
           Whether or not to normalize the values by the sum
    rank : Boolean
           If the data is normalized, whether to do so by absolute
           total values or by rank. If by value, this can lead to
           stepped functions if data_range='ext_data' is selected
           where some values may not occur. If True data is
           normalized by its absolute rank in order. This is a
           relatively standard trick to make CDF's appear
           continuous
    data_range : String or list
                 May be set to any of three values
                 'data': Indicates that only values that appear in
                         data, are keys in the returned dictionary
                         Note that this should be used in the
                         continuous data case.
                 'ext_data': Indicates that all values between
                             the minimum and maximum value included
                             in the data are included. With 0, for
                             values which do not appear. Note that
                             this assumes discrete integer values,
                             use custom for other types
                 other : If another data range is provided it is used
                         as keys for the data.
    pdf : dict of numeric
          Pre computed pdf

    Returns
    -------
    ccdh : dict of floats, keyed by occurance
           A dict of the same length, as the cumulative complementary
           distribution func.
    """
    cdf = cum_density_func(xs,norm,rank,data_range,pdf)
    max_v = np.max(cdf.values())
    return dict([(k,max_v - cdf[k]) for k in cdf.keys()]) 

def KSStat(xs,ys,reweight=False,cdf_x=None,cdf_y=None,data_range=None):
    """ Return the Kolomogorov-Smirnov statistic given two cumulative
    density functions, S of observed data, and P of the model.

    Returns the Kolomogorov-Smirnov statistic, defined as

        '    D=max|S(x)-P(x)|
        here S(x) is the CDF[cumulative distribution function] of the data
        for the observations...and P(x) is the CDF for the... model that
        bests fits the data...'

    On the matter of reweighting

        ' The KS statistic is, for instance, known to be relatively
        insensitive to to differences between distributions at the extreme
        limits of the range of x because in these limits the CDFs necessarily
        tend to zero and one. It can be reweighted to avoid this problem
        and be uniformly sensitive across the range; the appropriate re
        weighting is

               D* = max |S(x) -P(x)|/sqrt(P(x)(1-P(x)))' 

    Parameters
    ----------
    xs : list of numeric
         observed data values
    ys : list of numeric
         observed data values
    reweight : Boolean
               May provide more accurate statistic though probabily not
    cdf_x : dict of numeric
            Precomputed cdf for xs to save time
    cdf_y : dict of numeric
            Precomputed cdf for ys to save time
    data_range: string or list of numeric
                Range over which to calculate the cdfs. See cdf for more
                details.
    Returns
    -------
    KSStat: float
    """
    if cdf_x is None and cdf_y is None and data_range is None:
        data_range = list(set(xs)) + list(set(ys))
    if cdf_x is None:
        cdf_x = cum_density_func(xs,norm=True,rank=False,data_range=data_range)
    if cdf_y is None:
        cdf_y = cum_density_func(ys,norm=True,rank=False,data_range=data_range)
    keys = set(cdf_x.keys()+cdf_y.keys())
    SP = []
    for k in keys:
        if k in cdf_x and k in cdf_y:
            SP.append((cdf_x[k],cdf_y[k]))
    if reweight:
        return np.max([np.abs(s-p)/np.sqrt(p*(1.0-p)) for (s,p) in SP])
    else:
        return np.max([np.abs(s-p) for (s,p) in SP])

def KL_divergence(xs,ys,pdf_x=None,pdf_y=None,data_range=None):
    """Return the Kullback-Liebler Divergence of two probability density
    functions P constructed from xs and Q constructed from ys. The divergence
    is defined as

    D_kl(P||Q) = \sum_i P(i) log(P(i)/Q(i))

    This also assumes that if P(i) or Q(i) = 0 then there is no
    contribution to the sum, and that P(i) and Q(i) sum to 1.

    Generally P is measured or observed data and Q is a data
    generated by some model.
    
    The value can be interpretted as the amount of information required
    to code samples from P when using a code base Q.

    Paramters:
    ----------
    xs: list of numeric
        data of distribution P
    ys: list of numeric
        data of distribution Q
    pdf_x: dict of numeric
           pre-computed pdf of xs for speed
    pdf_y: dict of numeric
           pre-computed pdf of ys for speed
    data_range: string or list of numeric
                Range over which to calculate the cdfs. See cdf for more
                details.
    Returns:
    --------
    KL_divergence: float
    """
    if data_range is None:
        data_range = list(set(xs)) + list(set(ys))
    if pdf_x is None:
        pdf_x = prob_density_func(xs,norm=True,data_range=data_range)
    if pdf_y is None:
        pdf_y = prob_density_func(ys,norm=True,data_range=data_range)
    keys = set(pdf_x.keys()+pdf_y.keys())
    PQ = []
    for k in keys:
        if k in pdf_x and k in pdf_y:
            PQ.append((pdf_x[k],pdf_y[k]))
    return np.sum([p*np.log(float(p)/float(q)) for (p,q) in PQ if q>0 and p>0])

def JS_divergence(xs,ys,pdf_x=None,pdf_y=None,data_range=None):
    """Return the Jensen-Shannon Divergence of two probability density
    functions P and Q. The divergence is defined as

    JSD(P||Q) = .5*D(P||M) + .5D(Q||M)

    Where D(X||Y) is the Kullback-Leibler divergence and M is defined as

    M = .5*(P+Q)

    This assumes that P(i) and Q(i) sum to 1, also observe that M sums
    to 1.

    This is like the KL-Divergence except that it maintains nonnegativity,
    finiteness, semiboundedness and boundedness.

    Generally P is measured or observed data and Q is a data
    generated by some model.
    
    The value can be interpretted as the amount of information required
    to code samples from P when using a code base Q.

    Paramters:
    ----------
    xs: dict of numeric
       A probability density funciton
    ys: dict of numeric
       A probability density function
    pdf_x: dict of numeric
           pre-computed pdf of xs for speed
    pdf_y: dict of numeric
           pre-computed pdf of ys for speed
    data_range: string or list of numeric
                Range over which to calculate the cdfs. See cdf for more
                details.
    Returns:
    --------
    JS_divergence: float
    """
    if data_range is None:
        data_range = list(set(xs)) + list(set(ys))
    if pdf_x is None:
        pdf_x = prob_density_func(xs,norm=True,data_range=data_range)
    if pdf_y is None:
        pdf_y = prob_density_func(ys,norm=True,data_range=data_range)
    M = {}
    for i in pdf_x:
        if i not in pdf_y:
            pdf_y[i] = 0.0
    for i in pdf_y:
        if i not in pdf_x:
            pdf_y[i] = 0.0
    for i in pdf_x:
        M[i] = .5*(pdf_x[i] + pdf_y[i])
    return .5*KL_divergence(None,
                            None,
                            pdf_x=pdf_x,
                            pdf_y=M,
                            data_range=data_range) + \
           .5*KL_divergence(None,
                            None,
                            pdf_x=pdf_y,
                            pdf_y=M,
                            data_range=data_range)
def rmse(xs,ys,include_absent=True,default_x_value=None,default_y_value=None):
    """ Calculate the root mean square error between two dictionairs
    calculated as:

        rms = sqrt((1/n)*sum((x[i]-y[i])^2))

    Parameters
    ----------

    xs: Dictionary of Numeric
        First distribution of values
    ys: Dictionary of Numeric
        Second distribution of values

    include_absent: Boolean
        If True, uses the default_value in the case that a key
        exists in one dictionary but not in the other. If false
        only includes keys that occur in both dictionaries

    default_value: Numeric
        The value to use as a default if a key is included in one
        dictionary but not in the other. If None, will calculate the
        mean over the values and use that.

    Returns:
    --------

    rmse: Numeric
         The root mean square error
    """

    if include_absent and default_x_value is None:
        default_x_value = np.mean(xs.values())
    if include_absent and default_y_value is None:
        default_y_value = np.mean(ys.values())

    keys = xs.keys() + ys.keys()
    rmse = 0.0
    n = 0.0
    
    for k in keys:
        if k in xs and k in ys:
            rmse += (xs[k] - ys[k])**2.0
            n += 1.0
        elif include_absent and k in xs and not k in ys:
            rmse += (xs[k] - default_y_value)**2.0
            n += 1.0
        elif include_absent and not k in xs and k in ys:
            rmse += (default_x_value - ys[k])**2.0
            n += 1.0

    return np.sqrt(float(rmse)/float(n))

def rmse_pdf(xs,ys):
    """ Calculate the root mean square error between to sets of data
    when calculated against their probability density functions

    Parameters:
    -----------
    xs: List of numeric
    ys: List of numeric

    Returns:
    --------
    rms: Numeric
    """

    data_range = list(set(xs+ys))
    pdf_x = prob_density_func(xs, norm=True, data_range=data_range)
    pdf_y = prob_density_func(ys, norm=True, data_range=data_range)

    return rmse(pdf_x, pdf_y, include_absent=False)

def rmse_cdf(xs,ys):
    """ Calculate the root mean square error between to sets of data
    when calculated against their cumulative density functions

    Parameters:
    -----------
    xs: List of numeric
    ys: List of numeric

    Returns:
    --------
    rms: Numeric
    """

    data_range = list(set(xs+ys))
    cdf_x = cum_density_func(xs, norm=True, data_range=data_range)
    cdf_y = cum_density_func(ys, norm=True, data_range=data_range)
    return rmse(cdf_x, cdf_y, include_absent=False)

def area_diff_ccdf(xs,ys):
    data_range = sorted(list(set(xs + ys)))
    ccdf_x = comp_cum_distribution(xs, norm=True, data_range=data_range)
    ccdf_y = comp_cum_distribution(ys, norm=True, data_range=data_range)
    return area_diff(ccdf_x,ccdf_y,interpolate=False,ccdf=True)

def area_diff(X1,X2,interpolate=True,ccdf=False):
    if interpolate:
        Y1,Y2 = interpolate_dicts(X1,X2,ccdf)
    else:
        if sorted(X1.keys()) != sorted(X2.keys()):
            raise Exception("Dictionaries must have same keys if not \
                             interpolating!")
        else:
            Y1 = X1
            Y2 = X2
    data_range = sorted(Y1.keys())
    max_value = max(Y1.values()+Y2.values())
    min_value = min(Y1.values()+Y2.values())
    total_area = (max(data_range)-min(data_range))*(max_value-min_value)
    area_bet = 0.0
    for i in range(1,len(data_range)):
        x0 = data_range[i-1]
        x1 = data_range[i]
        b0 = abs(Y1[x0] - Y2[x0])
        b1 = abs(Y1[x1] - Y2[x1])
        h = x1-x0
        if Y1[x0]>=Y2[x0] and Y1[x1]>=Y2[x1]:
            #Trapezoid
            area_bet += .5 * h * (b0 + b1)
        elif Y1[x0]<Y2[x0] and Y1[x1]<Y2[x1]:
            #Trapezoid
            area_bet += .5 * h * (b0 + b1)
        elif Y1[x0]>=Y2[x0] and Y1[x1] < Y2[x1]:
            #Two Triangles
            area_bet += .5*b0*h*(b0/(b0+b1))
            area_bet += .5*b1*h*(b1/(b0+b1))
        else:
            #Two Triangles
            area_bet += .5*b0*h*(b0/(b0+b1))
            area_bet += .5*b1*h*(b1/(b0+b1))
        area_bet
    return area_bet/total_area
    
def interpolate_dicts(X1,X2,ccdf=False):
    data_range = sorted(list(set(X1.keys()+X2.keys())))
    Y1 = {}
    Y2 = {}
    K1 = sorted(X1.keys())
    K2 = sorted(X2.keys())
    i1 = 0
    i2 = 0
    for k in data_range:
        # If already there add them
        if k in K1:
            Y1[k] = X1[k]
            i1 += 1
        else: #interpolate
            if i1 == 0:
                if ccdf:
                    Y1[k] = 1.0
                else:
                    Y1[k] = X1[K1[0]]
            elif i1 == len(K1):
                if ccdf:
                    Y1[k] = 0.0
                else:
                    Y1[k] = X1[K1[-1]]
            else:
                m = (X1[K1[i1]] - X1[K1[i1-1]])/(K1[i1]-K1[i1-1])
                b = X1[K1[i1]] - m*K1[i1]
                Y1[k] = m*k + b
        if k in K2:
            Y2[k] = X2[k]
            i2 += 1
        else:
            if i2 == 0:
                if ccdf:
                    Y2[k] = 1.0
                else:
                    Y2[k] = X2[K2[0]]
            elif i2 == len(K2):
                if ccdf:
                    Y2[k] = 0.0
                else:
                    Y2[k] = X2[K2[-1]]
            else:
                m = (X2[K2[i2]] - X2[K2[i2-1]])/(K2[i2]-K2[i2-1])
                b = X2[K1[i2]] - m*K1[i2]
                Y2[k] = m*k + b
    return Y1,Y2
