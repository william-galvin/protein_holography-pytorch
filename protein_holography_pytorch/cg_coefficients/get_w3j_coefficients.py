import gzip, pickle
import e3nn
from e3nn import o3

print('e3nn version: ', e3nn.__version__)

def get_wigner_3j(lmax):
    w3j_matrices = {}
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            for l3 in range(abs(l2 - l1), min(l2 + l1, lmax) + 1):
                w3j_matrices[(l1, l2, l3)] = o3.wigner_3j(l1, l2, l3).numpy()
    return w3j_matrices

lmax = 14
with gzip.open('w3j_matrices-lmax=%d-version=%s.pkl' % (lmax, e3nn.__version__), 'wb') as f:
    pickle.dump(get_wigner_3j(lmax), f)

