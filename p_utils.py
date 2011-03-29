import multiprocessing.pool
import cPickle as pickle
import itertools
import types
import __builtin__
from warnings import warn

RUN = 0

def mapstar(a):
    return map(*a)

def reducestar(a):
    return reduce(*a)

def filterstar(a):
    return filter(*a)

def filter_reduce(x,y):
    return x+y

def chunks(l,n):
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c,n))
        if not x:
            return
        yield x

class PickleableFunction():
    def __init__(self,f):
        self.argcount = f.__code__.co_argcount
        self.nlocals = f.__code__.co_nlocals
        self.stacksize = f.__code__.co_stacksize
        self.flags = f.__code__.co_flags
        self.codestring = f.__code__.co_code
        self.constants = f.__code__.co_consts
        self.names = f.__code__.co_names
        self.varnames = f.__code__.co_varnames
        self.filename = f.__code__.co_filename
        self.name = f.__code__.co_name
        self.firstlineno = f.__code__.co_firstlineno
        self.lnotab = f.__code__.co_lnotab
        self.freevars = f.__code__.co_freevars
        self.cellvars = f.__code__.co_cellvars
        self.globals = f.__globals__.copy()
        for k in f.__globals__:
            try:
                pickle.dumps(self.globals[k])
            except:
                self.globals.pop(k)
        self.argdefs = f.__defaults__
        self.closure = f.__closure__
    def __call__(self,*args,**kwargs):
        co = types.CodeType(self.argcount,
                            self.nlocals,
                            self.stacksize,
                            self.flags,
                            self.codestring,
                            self.constants,
                            self.names,
                            self.varnames,
                            self.filename,
                            self.name,
                            self.firstlineno,
                            self.lnotab,
                            self.freevars,
                            self.cellvars)
        gbls = self.globals.copy()
        gbls.update(__builtin__.__dict__)
        f = types.FunctionType(co,globals=gbls,argdefs=self.argdefs,closure=self.closure)
        return f(*args,**kwargs)

class ePool(multiprocessing.pool.Pool):
    """Provides an enhancment to the Pool class provided by
    the multiprocessing module. Provides a map function emap,
    which more accurately matches the builtin map function,
    as well as parallel versions of a number of builtin functions
    including, reduce, filter, all, any, max, min, sum, and zip."""
    @staticmethod
    def _eget_tasks(func, its, size):
        its_c = []
        for it in its:
            its_c.append(iter(it))
        while 1:
            x = []
            for it in its_c:
                x.append(tuple(itertools.islice(it, size)))
            if not x[0]:
                return
            x.insert(0,func)
            yield tuple(x)
            
    def emap(self, func, sequence, *sequences, **kw):
        '''
        Equivalent of `map()` builtin
        '''
        try:
            chunksize=kw['chunksize']
        except:
            chunksize = None
        assert self._state == RUN

        return self.emap_async(func, sequence, *sequences, chunksize=chunksize).get()

    def emap_async(self, func, sequence, *sequences, **kw):
        '''
        Asynchronous equivalent of `map()` builtin
        '''

        try:
            pickle.dumps(func)
        except:
            F = PickleableFunction(func)
            try:
                pickle.dumps(F)
                test_args = itertools.izip(sequence,*sequences).next()
                F(*test_args)
            except:
                raise Exception("Cannot pickle function, or create working pickable version, is it a lamda function that makes use of unpickleable globals?")
            func = F

        try:
            chunksize = kw['chunksize']
        except:
            chunksize = None

        try:
            callback = kw['callback']
        except:
            callback = None

        assert self._state == RUN

        if not hasattr(sequence, '__len__'):
            sequence = list(sequence)

        seqs = [sequence]
        for s in sequences:
            if not hasattr(s, '__len__'):
                seqs.append(list(s))
            else:
                seqs.append(s)

        max_len = max([len(l) for l in seqs])

        for i in range(len(seqs)):
            if not len(seqs[i]) == max_len:
                try:
                    seqs[i].extend([None]*(max_len - len(seqs[i])))
                except:
                    seqs[i] = list(seqs[i])
                    seqs[i].extend([None]*(max_len - len(seqs[i])))

        if chunksize is None:
            chunksize, extra = divmod(max_len, len(self._pool))
            if extra:
                chunksize += 1
        
        task_batches = ePool._eget_tasks(func, seqs, chunksize)
        
        result = multiprocessing.pool.MapResult(self._cache,
                                                chunksize,
                                                max_len,
                                                callback)
        try:
            self._taskqueue.put((((result._job, i, mapstar, (x,), {})
                                  for i, x in enumerate(task_batches)), None))
        except KeyboardInterrupt:
            self._terminate()
        return result

    def reduce(self, func, iterable, initial=None,chunksize=None,callback=None):
        '''
        Parallel equivalent of reduce
        '''
        try:
            pickle.dumps(func)
        except:
            F = PickleableFunction(func)
            try:
                pickle.dumps(F)
                test_args = iter(iterable).next()
                F(*test_args)
            except:
                raise Exception("Cannot pickle function, or create working pickable version, is it a lamda function that makes use of unpickleable globals? Using serial map")
            func = F

        assert self._state == RUN
        
        if not hasattr(iterable, '__len__'):
            iterable = list(iterable)
        if len(iterable) == 0:
            if not initial is None:
                return initial
            else:
                raise Exception("Must provide non-empty list")

        if len(iterable) == 1:
            if not initial is None:
                return func(initial,iterable[0])
            else:
                return iterable[0]

        if chunksize is None:
            chunksize, extra = divmod(len(iterable), len(self._pool) )
            if extra:
                chunksize += 1
                
        num_chunks = len(iterable)//chunksize
        if len(iterable) % chunksize:
            num_chunks += 1


        if num_chunks <= len(iterable):
            return reduce(func,iterable)
        
        reduce_chunks = chunks(iterable,chunksize)

        result = self.emap(reducestar,
                           zip([func]*num_chunks,
                               reduce_chunks))
        if not initial is None:
            return func(initial,
                        self.reduce(func,
                                    result,
                                    chunksize=chunksize,
                                    callback=callback))
        else:
            return self.reduce(func,
                               result,
                               chunksize=chunksize,
                               callback=callback)

    def filter(self, func, iterable,chunksize=None,callback=None):
        '''
        Parallel equivalent of filter
        '''
        try:
            pickle.dumps(func)
        except:
            F = PickleableFunction(func)
            try:
                pickle.dumps(F)
                test_args = iter(iterable).next()
                F(*test_args)
            except:
                raise Exception("Cannot pickle function, or create working pickable version, is it a lamda function that makes use of unpickleable globals? Using serial map")
            func = F
        
        assert self._state == RUN
        if not hasattr(iterable, '__len__'):
            iterable = list(iterable)
        if len(iterable) == 0:
            return []

        if chunksize is None:
            chunksize, extra = divmod(len(iterable), len(self._pool) )
            if extra:
                chunksize += 1
                
        num_chunks = len(iterable)//chunksize
        if len(iterable) % chunksize:
            num_chunks += 1

        if num_chunks <= len(iterable):
            return filter(func,iterable)
        
        filter_chunks = chunks(iterable, chunksize)

        result = self.emap(filterstar,
                           zip([func]*num_chunks,
                               filter_chunks))
        return self.reduce(filter_reduce,
                           result,
                           chunksize=chunksize,
                           callback=callback)

    def sum(self, iterable, start=None, chunksize=None,callback=None):
        '''
        Parallel equivalent of sum
        '''
        assert self._state == RUN
        if not hasattr(iterable, '__len__'):
            iterable = list(iterable)

        if len(iterable) == 0:
            if not start is None:
                return start
            else:
                raise Exception("Must provide non-empty list")

        if len(iterable) == 1:
            if not start is None:
                return start + iterable[0]
            else:
                return iterable[0]

        if chunksize is None:
            chunksize, extra = divmod(len(iterable), len(self._pool) )
            if extra:
                chunksize += 1

        num_chunks = len(iterable)//chunksize
        if len(iterable) % chunksize:
            num_chunks += 1

        if num_chunks <= len(iterable):
            return sum(iterable)

        sum_chunks = chunks(iterable,chunksize)

        result = self.emap(sum, sum_chunks)
        if not start is None:
            return start + self.sum(result,
                                    chunksize=chunksize,
                                    callback=callback)
        else:
            return self.sum(result,chunksize=chunksize,callback=callback)

    def all(self, iterable, chunksize=None, callback=None):
        '''
        Parallel equivalent of all
        '''
        assert self._state == RUN
        if not hasattr(iterable, '__len__'):
            iterable = list(iterable)

        if len(iterable) == 0:
            raise Exception("Must provide non-empty list")

        if len(iterable) == 1:
            return bool(iterable[0])
        
        if chunksize is None:
            chunksize, extra = divmod(len(iterable), len(self._pool) )
            if extra:
                chunksize += 1
                
        num_chunks = len(iterable)//chunksize
        if len(iterable) % chunksize:
            num_chunks += 1

        if num_chunks <= len(iterable):
            return all(iterable)

        all_chunks = chunks(iterable,chunksize)

        result = self.emap(all, all_chunks)
        return self.all(result,chunksize=chunksize,callback=callback)

    def any(self, iterable, chunksize=None, callback=None):
        '''
        Parallel equivalent of any
        '''
        assert self._state == RUN
        if not hasattr(iterable, '__len__'):
            iterable = list(iterable)

        if len(iterable) == 0:
            raise Exception("Must provide non-empty list")

        if len(iterable) == 1:
            return bool(iterable[0])
        
        if chunksize is None:
            chunksize, extra = divmod(len(iterable), len(self._pool) )
            if extra:
                chunksize += 1

        num_chunks = len(iterable)//chunksize
        if len(iterable) % chunksize:
            num_chunks += 1

        if num_chunks <= len(iterable):
            return any(iterable)

        any_chunks = chunks(iterable,chunksize)

        result = self.emap(any, any_chunks)
        return self.any(result,chunksize=chunksize,callback=callback)

    def max(self, *vals, **kw):
        '''
        Parallel equivalent of max
        '''
        if len(vals) == 1:
            iterable = vals[0]
        else:
            iterable = vals

        if kw.has_key('chunksize'):
            chunksize = kw['chunksize']
        else:
            chunksize = None
        if kw.has_key('callback'):
            callback = kw['callback']
        else:
            chunksize = None

        if kw.has_key('key'):
            iterable = self.emap(kw['key'],iterable)
        
        assert self._state == RUN
        if not hasattr(iterable, '__len__'):
            iterable = list(iterable)

        if len(iterable) == 0:
            raise Exception("Must provide non-empty list")

        if len(iterable) == 1:
            return iterable[0]
        
        if chunksize is None:
            chunksize, extra = divmod(len(iterable), len(self._pool) )
            if extra:
                chunksize += 1

        num_chunks = len(iterable)//chunksize
        if len(iterable) % chunksize:
            num_chunks += 1

        if num_chunks <= len(iterable):
            return max(iterable)

        max_chunks = chunks(iterable,chunksize)

        result = self.emap(max, max_chunks)
        return self.max(result,chunksize=chunksize,callback=callback)
    
    def min(self, *vals, **kw):
        '''
        Parallel equivalent of min
        '''
        if len(vals) == 1:
            iterable = vals[0]
        else:
            iterable = vals

        if kw.has_key('chunksize'):
            chunksize = kw['chunksize']
        else:
            chunksize = None
        if kw.has_key('callback'):
            callback = kw['callback']
        else:
            callback = None

        if kw.has_key('key'):
            iterable = self.emap(kw['key'],iterable)
        
        assert self._state == RUN
        if not hasattr(iterable, '__len__'):
            iterable = list(iterable)

        if len(iterable) == 0:
            raise Exception("Must provide non-empty list")

        if len(iterable) == 1:
            return iterable[0]
        
        if chunksize is None:
            chunksize, extra = divmod(len(iterable), len(self._pool) )
            if extra:
                chunksize += 1


        num_chunks = len(iterable)//chunksize
        if len(iterable) % chunksize:
            num_chunks += 1

        if num_chunks <= len(iterable):
            return min(iterable)
        
        min_chunks = chunks(iterable,chunksize)

        result = self.emap(min, min_chunks)
        
        return self.min(result,chunksize=chunksize,callback=callback)

    def zip(self, *seqs, **kw):
        '''Parallel equivalent of zip'''
        if kw.has_key('chunksize'):
            chunksize = kw['chunksize']
        else:
            chunksize = None
        if kw.has_key('callback'):
            callback = kw['callback']
        else:
            callback = None

        min_len = self.min([len(list(s)) for s in seqs])

        results = self.emap(None,*seqs,chunksize=chunksize,callback=callback)

        return results[:min_len]
        
    
def pmap(func,sequence,*sequences,**kw):
    """ Replacement for map, which runs in parallel. Does not work for
    inline functions or static methods, will use serial map if
    some of these functions is used.

    Also cannot use functions that themselves contain parallel higher
    order functions defined in this module."""

    try:
        processes = kw['processes']
    except:
        processes = None
    try:
        return ePool(processes=processes).emap(func,sequence,*sequences,**kw)
    except:
        warn("pmap failed, it is possible that the function references another parallel higher order function or uses inaccessable globals. Using serial map")
        return map(func,sequence,*sequences)

def preduce(func,iterable,initial=None,processes=None,chunksize=None):
    """ Replaces builtin reduce, works more exactly like builtin
    reduce, with the same caveats for pmap."""
    try:
        return ePool(processes=processes).reduce(func,iterable,initial,chunksize=chunksize)
    except:
        warn("preduce failed, it is possible that the function references another parallel higher order function or uses inaccessable globals. Using serial reduce")
        return reduce(func,iterable,initial)

def pfilter(func,iterable,processes=None,chunksize=None):
    """ Replaces builtin filter, works exactly like builtin filter
    with the same caveats as pmap."""
    try:
        return ePool(processes=processes).filter(func,iterable,chunksize=chunksize)
    except:
        warn("pfilter failed, it is possible that the function references another parallel higher order function or uses inaccessable globals. Using serial filter")
        return filter(func,iterable)

def pall(iterable,processes=None,chunksize=None):
    """Parallel replacement for all, may not be as fast if only
    running on a small number of processes."""
    return ePool(processes=processes).all(iterable,chunksize=chunksize)

def pany(iterable,processes=None,chunksize=None):
    """Parallel replacement for any, may not be as fast if only
    running on a small number of processes."""
    return ePool(processes=processes).any(iterable,chunksize=chunksize)

def pmax(*vals, **kw):
    """Parallel replacement for max, may not be as fast if only
    running on a small number of processes."""
    if kw.has_key['processes']:
        prcesses = kw['processes']
    return ePool(processes=processes).max(*vals, **kw)

def pmin(*vals,**kw):
    """Parallel replacement for min, may not be as fast if only
    running on a small number of processes."""
    if kw.has_key['processes']:
        prcesses = kw['processes']
    return ePool(processes=processes).min(*vals, **kw)

def psum(iterable,start=None,processes=None,chunksize=None):
    """Parallel replacement for sum, may not be as fast if only
    running on a small number of processes."""
    return ePool(processes=processes).sum(iterable,start,chunksize=chunksize)

def test_inline_func(n):
    def f(x):
        return x**2
    return pmap(f,range(n))

def _test_lambda_func(n):
    return pmap(lambda x: _test_long_f(x),range(n))

def _test_f(x):
    return x**2

def _test_long_f(x):
    for i in range(2000):
        x+=1
    return x

def _test_nest_f(x):
    return pmap(test_long_f, range(x))

def _test_named_func(n):
    return pmap(test_long_f,range(n))

def _test_nest(n):
    return pmap(test_nest_f, range(n))
