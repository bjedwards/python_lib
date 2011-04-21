import matplotlib.pyplot as plt
import stats.general as st

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
