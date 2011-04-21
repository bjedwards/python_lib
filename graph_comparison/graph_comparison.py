import functools
import numpy as np
import os
import matplotlib
#matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from .utilities import plot_utils as pltu
from .utilities import p_utils
from .utilities.stats import general as gn
from .metrics import metrics

def _empt(*args,**kwargs):
    return None

def _empt_dict(models):
    ed = {}
    for m in models:
        ed[m] = None
    return ed

def write_to_file(results,*args,**kwargs):
    metric_name = kwargs['metric_name']
    label = kwargs['label']
    if os.path.isfile('./results/' + metric_name + '.txt'):
        f = open('./results/' + metric_name + '.txt','a')
    else:
        f = open('./results/' + metric_name + '.txt','w')
        f.write(metric_name + '\n')
    f.write('\t' + str(label) + '\t' + str(results) + '\n')
    f.close()
    return None

num_edges_kw = {'metric_func':lambda G: G.number_of_edges(),
                'metric_name':'Number of Edges',
                'comp_funcs':{'Absolute Error':lambda x1,x2: abs(x1-x2),
                              'Relative Error':lambda x1,x2: abs(x1-x2)/float(abs(x1))},
                'metric_plot_func':functools.partial(write_to_file,
                                                     metric_name='NumEdges'),
                'plot_xlabel':'',
                'plot_ylabel':'',
                'plot_title':'',
                'plot_legend_loc':'best',
                'parallel':False,
                'processes':None}

max_deg_kw = {'metric_func':lambda G: max(G.degree().values()),
              'metric_name':'Max Degree',
              'comp_funcs':{'Absolute Error':lambda x1,x2: abs(x1-x2),
                            'Relative Error':lambda x1,x2: abs(x1-x2)/float(abs(x1))},
              'metric_plot_func':functools.partial(write_to_file,
                                                   metric_name='MaxDegree'),
              'plot_xlabel':'',
              'plot_ylabel':'',
              'plot_title':'',
              'plot_legend_loc':'best',
              'parallel':False,
              'processes':None}

avg_deg_kw = {'metric_func':lambda G: float(sum(G.degree().values()))/G.order(),
              'metric_name':'Average Degree',
              'comp_funcs':{'Absolute Error':lambda x1,x2: abs(x1-x2),
                            'Relative Error':lambda x1,x2: abs(x1-x2)/float(abs(x1))},
              'metric_plot_func':functools.partial(write_to_file,
                                                   metric_name='AvgDegree'),
              'plot_xlabel':'',
              'plot_ylabel':'',
              'plot_title':'',
              'plot_legend_loc':'best',
              'parallel':False,
              'processes':None}

deg_cent_kw = {'metric_func':lambda G: G.degree().values(),
               'metric_name':'Degree Centrality',
               'comp_funcs':{'Jenson Shannon Divergence':gn.JS_divergence,
                             'Kolomgorov-Smirnov Statistic':gn.KSStat,
                             'Root Mean Square Error (CDF)':gn.rmse_cdf,
                             'Root Mean Square Error (PDF)':gn.rmse_pdf},
               'metric_plot_func':pltu.plot_ccdf_loglog,
               'plot_xlabel':'Degree $k$',
               'plot_ylabel':'$Pr[deg(v) > k]$',
               'plot_title':'Degree Centrality Comparison',
               'plot_legend_loc':'best',
               'parallel':False,
               'processes':None}

pr_cent_kw = {'metric_func':metrics.pagerank,
              'metric_name':'Page Rank Centrality',
              'comp_funcs':{'Kolomgorov-Smirnov Statistic':gn.KSStat,
                            'Root Mean Square Error (CDF)':gn.rmse_cdf},
              'metric_plot_func':pltu.plot_ccdf_loglog,
              'plot_xlabel':'Page Rank $r$',
              'plot_ylabel':'$Pr[pr(v) > r]$',
              'plot_title':'Page Rank Comparison',
              'plot_legend_loc':'best',
              'parallel':True,
              'processes':None}

bt_cent_kw = {'metric_func':metrics.between_cent,
              'metric_name':'Betweenness Centrality',
              'comp_funcs':{'Kolomgorov-Smirnov Statistic':gn.KSStat,
                            'Root Mean Square Error (CDF)':gn.rmse_cdf},
              'metric_plot_func':pltu.plot_ccdf_loglog,
              'plot_xlabel':'Betweenness $b$',
              'plot_ylabel':'$Pr[bet_cent(v) > b]$',
              'plot_title':'Betweenness Centrality Comparison',
              'plot_legend_loc':'lower left',
              'parallel':False,
              'processes':None}

avg_path_kw = {'metric_func':metrics.avg_path_len,
               'metric_name':'Average Path Length',
               'comp_funcs':{'Kolomgorov-Smirnov Statistic':gn.KSStat,
                             'Root Mean Square Error (CDF)':gn.rmse_cdf},
               'metric_plot_func':pltu.plot_ccdf_semilogy,
               'plot_xlabel':'Average Path Length $l$',
               'plot_ylabel':'$Pr[path_length(v) > l]$',
               'plot_title':'Average Path Length Comparison',
               'plot_legend_loc':'lower left',
               'parallel':False,
               'processes':None}

rich_club_kw = {'metric_func':metrics.rich_club,
                'metric_name':'Rich Club Coefficient',
                'comp_funcs':{'Root Mean Square Error':gn.rmse},
                'metric_plot_func':pltu.plot_dict_loglog,
                'plot_xlabel':'Degree $k$',
                'plot_ylabel':'$\phi(k)$',
                'plot_title':'Rich Club Coefficient Comparison',
                'plot_legend_loc':'lower right',
                'parallel':True,
                'processes':None}

n_rich_club_kw = {'metric_func':metrics.rich_club_norm,
                  'metric_name':'Normalized Rich Club Coefficient',
                  'comp_funcs':{'Root Mean Square Error':functools.partial(gn.rmse,include_absent=False)},
                  'metric_plot_func':pltu.plot_dict_semilogx,
                  'plot_xlabel':'Degree $k$',
                  'plot_ylabel':'$phi(k)$',
                  'plot_title':'Normalized Rich Club Coefficient Comparison',
                  'plot_legend_loc':'upper left',
                  'parallel':True,
                  'processes':None}

cluster_kw = {'metric_func':metrics.clust,
              'metric_name':'Clustering Coefficient',
              'comp_funcs':{'Kolomgorov-Smirnov Statistic':gn.KSStat,
                            'Root Mean Square Error (CDF)':gn.rmse_cdf},
              'metric_plot_func':pltu.plot_ccdf_semilogx,
              'plot_xlabel':'Clustering Coefficient $cc$',
              'plot_ylabel':'$Pr[clust(v) > cc]$',
              'plot_title':'Clustering Coefficient Comparison',
              'plot_legend_loc':'center left',
              'parallel':True,
              'processes':None}


k_cores_kw = {'metric_func':metrics.cores,
              'metric_name':'K-Cores',
              'comp_funcs':{'Jenson Shannon Divergence':gn.JS_divergence,
                            'Kolomgorov-Smirnov Statistic':gn.KSStat,
                            'Root Mean Square Error (CDF)':gn.rmse_cdf,
                            'Root Mean Square Error (PDF)':gn.rmse_pdf},
              'metric_plot_func':pltu.plot_ccdf_loglog,
              'plot_xlabel':'Core $k$',
              'plot_ylabel':'$Pr[core(v) > k]$',
              'plot_title':'K-Cores Comparison Comparison',
              'plot_legend_loc':'best',
              'parallel':True,
              'processes':None}

smet_kw = {'metric_func':nx.s_metric,
           'metric_name':'S-Metric',
           'comp_funcs':{'Absolute Error':lambda x1,x2: abs(x1-x2),
                         'Relative Error':lambda x1,x2: abs(x1-x2)/float(abs(x1))},
           'metric_plot_func':functools.partial(write_to_file,
                                                metric_name='S-Metric'),
           'plot_xlabel':'',
           'plot_ylabel':'',
           'plot_title':'',
           'plot_legend_loc':'best',
           'parallel':True,
           'processes':None}

assort_kw = {'metric_func':nx.degree_assortativity,
             'metric_name':'Degree Assortativity',
             'comp_funcs':{'Absolute Error':lambda x1,x2: abs(x1-x2),
                         'Relative Error':lambda x1,x2: abs(x1-x2)/float(abs(x1))},
             'metric_plot_func':functools.partial(write_to_file,
                                                  metric_name='DegreeAssort'),
             'plot_xlabel':'',
             'plot_ylabel':'',
             'plot_title':'',
             'plot_legend_loc':'best',
             'parallel':True,
             'processes':None}

def compare_graphs(models,
                   comp_model=None,
                   filename='',
                   metric_func=_empt,
                   metric_name='',
                   comp_funcs={},
                   metric_plot_func=None,
                   plot_xlabel='',
                   plot_ylabel='',
                   plot_title='',
                   plot_legend_loc='best',
                   parallel=True,
                   processes=None):
    """ A wrapper function to compare a dictionary of models on a
    variety of given topological metrics. Will produce output files
    of any given nature, in the results folder.

    Parameters
    ----------
    models: dictionary of networkx Graphs
            Dictionary of networkx graphs to be analyzed, keyed by
            anything that can be converted into str() for plotting
            purposes
    metric_func: function
                 A function on which to evaluate the graph, must
                 take a networkx graph object, but can return
                 anything
    comp_funcs: Dictionary of functions
                This dictionary contains the possible functions for use
                in comparing metrics. 
    comp_model: key in models.keys()
                This is a comparison model for which all the other models
                are compared to.
    filename: String
              String to append to outputed result filenames.                
    metric_plot_func: function
                      A function which takes the result of metric_func
                      and produces a plot, or writes to a file, if None
                      no output will be produced

    plot_xlabel: String
                 A label for the xaxis if the plot function produces
                 a matplotlib plot.
    plot_ylabel: String
                 A label for the yaxis if the plot function produces
                 a matplotlib plot.
    plot_title:  String
                 A label for the title if the plot function produces
                 a matplotlib plot.
    plot_legend_loc: String
                     A location for for the legend if a matplotlib
                     plot is produces
    parallel: Bool
              Whether to attempt to apply the metric to all models at
              once in paralle. Warning can be memory intensive, and
              may fail if underlying metrics are already parallelized.
    processes: int
               Number of processes to use if using parallel, defaults
               to the number of cores on the machine.
    Returns:
    --------
    None produces output files
    """
    plt.ion()
    plt_flag = False
    if parallel:
        if not comp_model is None:
            f = open('./results/'+ filename + 'metric_comparison.txt','a')
            f.write(metric_name+'\n')
        k = sorted(models.keys())
        if processes is None:
            #No sense in starting more processes than we need to
            processes = min(len(k),p_utils.multiprocessing.cpu_count())
        res = p_utils.pmap(metric_func,
                           [models[m] for m in k],
                           processes=processes)
        results = dict(zip(k,res))
        try:
            plt.clf()
        except:
            pass
        for m in sorted(results.keys()):
            try:
                plt_ret = metric_plot_func(results[m],label=str(m))
                plt_flag = plt_flag or not (plt_ret is None)
            except:
                pass
        if plt_flag:
            try:
                plt.xlabel(plot_xlabel)
                plt.ylabel(plot_ylabel)
                plt.title(plot_title)
                plt.legend(loc=plot_legend_loc)
                plt.savefig('./results/' + filename + metric_name + '.png')
            except:
                pass
        if not comp_model is None:
            for m in results:
                for cf in comp_funcs:
                    f.write('\t' +
                            cf + '\t' +
                            str(comp_model) + '\t' +
                            str(m) + '\t' +
                            str(comp_funcs[cf](results[comp_model],
                                               results[m])) + '\n')
    else:
        if not comp_model is None:
            f = open('./results/'+ filename + 'metric_comparison.txt','a')
            f.write(metric_name+'\n')
            comp_res = metric_func(models[comp_model])
        try:
            plt.clf()
        except:
            pass
        for m in sorted(models.keys()):
            print(m)
            if m == comp_model:
                results = comp_res
            else:
                results = metric_func(models[m])
            if not comp_model is None:
                for cf in comp_funcs:
                    f.write('\t' +
                            cf + '\t' +
                            str(comp_model) + '\t' +
                            str(m) + '\t' +
                            str(comp_funcs[cf](comp_res, results)) + '\n')
            try:
                plt_ret = metric_plot_func(results,label=str(m))
                #plt.show()
            except:
                plt_ret = None
        if not plt_ret is None:
            try:
                plt.xlabel(plot_xlabel)
                plt.ylabel(plot_ylabel)
                plt.title(plot_title)
                #plt.legend(loc=plot_legend_loc)
                plt.savefig('./results/' + filename + metric_name + '.png')
            except:
                pass
    if not comp_model is None:
        f.close()
    return

def make_comp_matrix(filename='',
                     comp_metrics=None,
                     inc_comp_model=False,
                     aggregate=False,
                     scale=False,
                     overall=False):
    """ Takes output files from the above analysis and creates
    an easy to ready plot comparing the output comparisons. 

    Parameters:
    -----------
    filename: string
              File which contains the output from compare_models above
              should be in the results folder
    comp_metrics: List of comparison metrics to use
                  List of which comparison metrics should be used to make
                  the plot. If None, will simply use those present in the file.
    inc_comp_model: Boolean
                    Whether to include the comparison model in the plot,
                    this will show whether the various models do well
                    relative to one another or the baseline comparison
    aggregate: Boolean
               Whether to aggregate the various comparison methods used on
               a given metric into a single value, simply takes an average
    scale: Boolean
           Whether to scale the results of the comparison methods on a 0,
           to 1 scale, makes the aggregate measure make more sense
    overall: Boolean
             Whether to include a new measurement which is the average
             performance overall metrics.

    Returns:
    --------
    None: Produces output graphs"""
    
    f = open(filename)
    lines = f.readlines()
    data = []
    for l in lines:
        data.append(l.strip('\t').lstrip(' ').rstrip('\n').split('\t'))

    if comp_metrics is None:
        comp_met_empt = True
        comp_metrics = []
    else:
        comp_met_empt = False
    metrics = []
    models = []
    comp_data = {}
    while (len(data) > 0):
        if (len(data[0]) == 1):
            metrics.append(data[0][0])
            curr_met = data[0][0]
            comp_data[curr_met] = {}
            data.pop(0)
        else:
            d = data.pop(0)
            if not d[0] in comp_data[curr_met]:
                comp_data[curr_met][d[0]] = {}
            if comp_met_empt and not d[0] in comp_metrics:
                comp_metrics.append(str(d[0]))
            if not d[2] in models:
                models.append(d[2])
            if d[1] != d[2] or inc_comp_model:
                comp_data[curr_met][d[0]][d[2]] = float(d[3])
    if not inc_comp_model:
        models.remove(d[1])

    if scale:
        for met in comp_data:
            for comp_met in comp_data[met]:
                comp_data[met][comp_met] = _scale_dict(comp_data[met][comp_met])
                
    if aggregate:
        comp_metrics.append('Aggregate')
        for met in comp_data:
            comp_mets = comp_data[met].keys()
            comp_data[met]['Aggregate'] = dict.fromkeys(models,0.0)
            num_comp_mets = 0
            for comp_met in comp_mets:
                num_comp_mets += 1
                for m in comp_data[met][comp_met]:
                    comp_data[met]['Aggregate'][m] += comp_data[met][comp_met][m]
            for m in comp_data[met]['Aggregate']:
                comp_data[met]['Aggregate'][m] /= float(num_comp_mets)
    if overall and aggregate:
        metrics.append('Overall')
        comp_data['Overall'] = {}
        comp_data['Overall']['Aggregate'] = dict.fromkeys(models,0.0)
        for met in comp_data:
            for m in comp_data[met]['Aggregate']:
                comp_data['Overall']['Aggregate'][m] += comp_data[met]['Aggregate'][m]/(float(len(metrics)))

    models.sort()
    metrics.sort()

    for comp_met in comp_metrics:
        met_mat = []
        used_metrics = []
        for met in metrics:
            if comp_met in comp_data[met]:
                used_metrics.append(met)
                met_mat.append([comp_data[met][comp_met][m] for m in models])
        _draw_comp(np.transpose(met_mat),models,used_metrics,comp_met=comp_met)
#        plt.show()
        
    return comp_data,models,metrics,comp_metrics

def _draw_comp(mat,models,metrics,comp_met=''):
    plt.imshow((mat),interpolation='nearest',cmap=cm.RdYlBu_r)
    fig = plt.gcf()
    fig.set_size_inches(16,16)
    ax = plt.gca()
    ax.set_yticks(np.arange(0,len(models),1.0))
    ax.set_xticks(np.arange(0,len(metrics),1.0))
    ax.set_yticklabels(models)
    ax.set_xticklabels(metrics)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(90)
    plt.ylim([len(models)-0.5, -0.5])
    plt.xlim([-0.5,len(metrics)-0.5])
    plt.colorbar(shrink=.55)
    plt.title(str(comp_met) + ' Model and Metric Comparison Chart')
    plt.savefig('./results/' + str(comp_met).replace(' ','') + 'CompChart.png')

def _scale_dict(d,xmin=None, xmax=None):
    if xmin is None:
        xmin = min(d.values())
    if xmax is None:
        xmax = max(d.values())
    norm_d = {}
    for k in d:
        norm_d[k] = float(d[k]-xmin)/float(xmax-xmin)
    return norm_d
