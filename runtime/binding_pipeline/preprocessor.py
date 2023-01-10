import signal
from multiprocessing import Pool

def process_data(pdb):
    assert(process_data.callback)    
    return process_data.callback(pdb, **process_data.params)

def initializer(init, callback, params, init_params):
    if init is not None:
        init(**init_params)
    process_data.callback = callback
    process_data.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PDBPreprocessor:
    def __init__(self, pdb_list):
        self.__data = pdb_list
        self.size = len(pdb_list)
        
    def count(self):
        return len(self.__data)

    def execute(self, callback, parallelism = 8, limit = None, params = None, init = None, init_params = None):
        if limit is None:
            data = self.__data
        else:
            data = self.__data[:limit]
        with Pool(initializer = initializer, processes=parallelism, 
                  initargs = (init, callback, params, init_params)) as pool:

            for res in pool.imap(process_data, data):
                if res:
                    yield res