import matplotlib.pyplot as plt
import stats.general as st

def plot_tuples(l,*args,**kwargs):
    X = [x[0] for x in l]
    Y = [y[1] for y in l]
    return plt.plot(X,Y,*args,**kwargs)

def plot_dict(d,*args,**kwargs):
    X = sorted(d.keys())
    Y = [d[x] for x in X]
    return plt.plot(X,Y,*args,**kwargs)

def plot_dict_loglog(d,*args,**kwargs):
    X = sorted(d.keys())
    Y = [d[x] for x in X]
    return plt.loglog(X,Y,*args,**kwargs)

def plot_dict_semilogx(d,*args,**kwargs):
    X = sorted(d.keys())
    Y = [d[x] for x in X]
    return plt.semilogx(X,Y,*args,**kwargs)

def plot_dict_semilogy(d,*args,**kwargs):
    X = sorted(d.keys())
    Y = [d[x] for x in X]
    return plt.semilogy(X,Y,*args,**kwargs)

def plot_ccdf(xs,*args,**kwargs):
    ccdf = st.comp_cum_distribution(xs)
    return plot_dict(ccdf,*args,**kwargs)

def plot_ccdf_semilogx(xs,*args,**kwargs):
    ccdf = st.comp_cum_distribution(xs)
    return plot_dict_semilogx(ccdf,*args,**kwargs)

def plot_ccdf_semilogy(xs,*args,**kwargs):
    ccdf = st.comp_cum_distribution(xs)
    return plot_dict_semilogy(ccdf,*args,**kwargs)

def plot_ccdf_loglog(xs,*args,**kwargs):
    ccdf = st.comp_cum_distribution(xs)
    return plot_dict_loglog(ccdf,*args,**kwargs)

def plot_pdf(xs,*args,**kwargs):
    pdf = st.prob_density_func(xs)
    return plot_dict(pdf,*args,**kwargs)

def plot_pdf_semilogx(xs,*args,**kwargs):
    pdf = st.prob_density_func(xs)
    return plot_dict_semilogx(pdf,*args,**kwargs)

def plot_pdf_semilogy(xs,*args,**kwargs):
    pdf = st.prob_density_func(xs)
    return plot_dict_semilogy(pdf,*args,**kwargs)

def plot_pdf_loglog(xs,*args,**kwargs):
    pdf = st.prob_density_func(xs)
    return plot_dict_loglog(pdf,*args,**kwargs)

def plot_dict_dual_y(d1,
                     d2,
                     args1=(),
                     args2=(),
                     kwargs1={},
                     kwargs2={},
                     ax1=None,
                     ax2=None):
    xs = sorted(list(set(sorted(d1.keys())+sorted(d2.keys()))))
    ys1 = [d1[x] for x in xs if x in d1]
    ys2 = [d2[x] for x in xs if x in d2]
    return plot_dual_y(xs,ys1,ys2,args1,args2,kwargs2,kwargs2,ax1,ax2)

def plot_dual_y(xs,
                ys1,
                ys2,
                args1=(),
                args2=(),
                kwargs1={},
                kwargs2={},
                ax1=None,
                ax2=None):
    if ax1 is None:
        plt.clf()
        fig = plt.gcf()
        ax1 = fig.add_subplot(111)
    if ax2 is None:
        ax2 = ax1.twinx()
    ax1.plot(xs,ys1,*args1,**kwargs1)
    ax2.plot(xs,ys2,*args2,**kwargs2)
    return ax1,ax2

def plot_dual_y_date(dates,
                     ys1,
                     ys2,
                     args1=(),
                     args2=(),
                     kwargs1={},
                     kwargs2={},
                     ax1=None,
                     ax2=None):
    if ax1 is None:
        plt.clf()
        fig = plt.gcf()
        ax1 = fig.add_subplot(111)
    if ax2 is None:
        ax2 = ax1.twinx()
    ax1.plot_date(dates,ys1,*args1,**kwargs1)
    ax2.plot_date(dates,ys2,*args2,**kwargs2)
    return ax1,ax2
