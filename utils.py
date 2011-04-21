import random
import bisect

def dict_list_to_tuples(d):
    l = []
    for k in d:
        l.extend(zip([k]*len(d[k]),d[k]))
    return l

def flatten_dict_dict(d):
    l_of_ls = [d[x].values() for x in d]
    return reduce(lambda x,y: x+y, l_of_ls, [])

def cumulative_sum(numbers):
    """Yield cumulative sum of numbers.
    
    >>> import networkx.utils as utils
    >>> list(utils.cumulative_sum([1,2,3,4]))
    [1, 3, 6, 10]
    """
    csum = 0
    for n in numbers:
        csum += n
        yield csum

def weighted_random_choice(w,cumulative=False):
    """ Select a weighted random choice from a dictionary of key weight
    pairs

    Parameters:
    -----------
    w: dict
       dictionary of item weight pairs

    Returns:
    --------
    key_ind[rand_ret] : Random key
    """
    if not cumulative:
        cs = list(cumulative_sum(w.values()))
    else:
        cs = [w[k] for k in sorted(w.values())]
    key_ind = dict(zip(range(len(w)),w.keys()))
    rnd = random.random()*cs[-1]
    rand_ret = bisect.bisect_left(cs,rnd)
    return key_ind[rand_ret]
