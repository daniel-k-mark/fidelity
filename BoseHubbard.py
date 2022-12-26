import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from scipy.linalg import expm
from k_design import *
from Fc import *

def hil_sp_dim(M,N):
### Hilbert space dimension of N particles on M sites
    if M==0:
        return int(N<1)
    if M==1:
        return 1
    elif M>1:
        dim = 1
        for j in range(1,N+1):
            dim+=hil_sp_dim(M-1,j)
        return dim

def idx_to_boson(idx,M,N):
    occ = np.zeros(M,dtype=np.int16)
    num = idx
    for j in range(N,0,-1):
        for k in list(range(M,-1,-1)):
            if num>=hil_sp_dim(k,j):
                occ[M-k-1]+=1
                num += -hil_sp_dim(k,j)
                #print(hil_sp_dim(k,j))
                break
    assert(sum(occ)==N)
    return occ

def Fock_vec_to_occ(b):
    #M = len(b)
    #N = sum(b)
    #occ_vec = sum([[j]*bb[j-1] for j in range(len(bb),0,-1)],[]) ##list_comprehension that is slower than current version
    occ_vec = []
    for j in range(len(b),0,-1):
        occ_vec += [j]*b[j-1]
        #for k in range(b[j-1]): ##oldest version
        #    occ_vec.append(j) #which site the jth particle is at
    #assert(len(occ_vec)==N)
    return occ_vec

def boson_to_idx(b):
    M = len(b)
    N = sum(b)
    occ_vec = Fock_vec_to_occ(b)
    num = 0
    for j in range(1,N+1):
        num += hil_sp_dim(M-occ_vec[j-1],j)
        #num.append(hil_sp_dim(M-occ_vec[j-1],j))
    return num
    
def nearest_neighbors_1d(M):
    return [[j,j+1] for j in range(M-1)]

def nearest_neighbors_2d(Lx,Ly):
    ar = np.reshape(range(Lx*Ly),(Lx,Ly))
    lr_pairs = np.transpose(np.reshape([np.roll(ar,1,axis=1)[:,1:],ar[:,1:]],(2,-1)))
    ud_pairs = np.transpose(np.reshape([np.roll(ar,1,axis=0)[1:],ar[1:]],(2,-1)))
    return np.concatenate([lr_pairs, ud_pairs])

def hopping_ham(M,N):
    dim = hil_sp_dim(M,N)
    ham = lil_matrix((dim,dim))
    for j in range(dim):
        ar = idx_to_boson(j,M,N)
        for l in nearest_neighbors_1d(M):
            for ii in [0,1]:
                if ar[l[ii]]>0:
                    ar[l[ii]]-= 1
                    ar[l[1-ii]]+=1
                    idx = boson_to_idx(ar)
                    ham[idx,j] += -np.sqrt((ar[l[ii]]+1)*ar[l[1-ii]])
                    ar[l[ii]]+= 1
                    ar[l[1-ii]]-=1
    return ham.tocsr()

def hopping_ham_2d(Mx,My,N):
    M = Mx*My
    dim = hil_sp_dim(M,N)
    ham = lil_matrix((dim,dim))
    for j in range(dim):
        ar = idx_to_boson(j,M,N)
        for l in nearest_neighbors_2d(Mx,My):
            for ii in [0,1]:
                if ar[l[ii]]>0:
                    ar[l[ii]]-= 1
                    ar[l[1-ii]]+=1
                    idx = boson_to_idx(ar)
                    ham[j,idx] += -np.sqrt(ar[l[ii]]+1)*np.sqrt(ar[l[1-ii]])
                    ar[l[ii]]+= 1
                    ar[l[1-ii]]-=1
    return ham.tocsr()

def U_ham(M,N):
    dim = hil_sp_dim(M,N)
    ham = lil_matrix((dim,dim))
    for j in range(dim):
        ar = idx_to_boson(j,M,N)
        ham[j,j] = np.dot(ar,ar-1)
    return ham.tocsr()
    
## Error operators
def make_error_cs_boson(M,N,k):
    dim = hil_sp_dim(M,N)
    ham = np.zeros(dim)
    for j in range(dim):
        ar = idx_to_boson(j,M,N)
        ham[j] += ar[k]
    return ham

def make_error_cdaggercs_boson(M,N,k):
    dim = hil_sp_dim(M,N)
    ham = lil_matrix((dim,dim))
    for j in range(dim):
        ar = idx_to_boson(j,M,N)
        ham[j,j] += ar[k]**2
    return ham.tocsr()
    
## Single-site rotation
def hopping_ham_one_site(M,N):
    dim = hil_sp_dim(M,N)
    ham = lil_matrix((dim,dim))
    for j in range(dim):
        ar = idx_to_boson(j,M,N)
        for l in [[M//2,M//2+1]]:
            for ii in [0,1]:
                if ar[l[ii]]>0:
                    ar[l[ii]]-= 1
                    ar[l[1-ii]]+=1
                    idx = boson_to_idx(ar)
                    ham[idx,j] += -np.sqrt((ar[l[ii]]+1)*ar[l[1-ii]])
                    ar[l[ii]]+= 1
                    ar[l[1-ii]]-=1
    return ham.tocsr()
    
## Optimized functions
def readout_dic(M,N):
    dim = hil_sp_dim(M,N)
    lst = []
    for j in range(dim):
        lst.append(idx_to_boson(j,M,N))
    return np.array(lst)

def hil_sp_dim_square(M,N):
    square = []
    for m in range(M+1):
        square_row = []
        for n in range(N+1):
            square_row.append(hil_sp_dim(m,n))
        square.append(square_row)
    return np.array(square)

def idx_to_boson_fast(idx,M,N,hs_sq):
    occ = np.zeros(M,dtype=np.int16)
    num = idx
    for j in range(N,0,-1):
        k = M
        hh = hs_sq[k,j]
        while (num<hh and k>=0):
            k += -1
            hh = hs_sq[k,j]
        occ[M-k-1]+=1
        num += -hh
    assert(sum(occ)==N)
    return occ

def boson_to_idx_fast(b,hs_sq):
    M = len(b)
    N = sum(b)
    assert np.shape(hs_sq)==(M+1,N+1)
    occ_vec = Fock_vec_to_occ(b)
    num = 0
    for j in range(1,N+1):
        num += hs_sq[M-occ_vec[j-1],j]
    return num

def readout_dic_fast(M,N,hs_sq):
    dim = hil_sp_dim(M,N)
    lst = []
    for j in range(dim):
        lst.append(idx_to_boson_fast(j,M,N,hs_sq))
        if j%1000==0:
            print(j/dim, end="\r")
    return np.array(lst)

def hopping_ham_fast(M,N,readout,hs_sq):
    assert np.shape(hs_sq)==(M+1,N+1)
    dim = hs_sq[M,N]
    ham = lil_matrix((dim,dim))
    nn_l = nearest_neighbors_1d(M)
    for j in range(dim):
        ar = readout[j]
        ba=Fock_vec_to_occ(ar)
        #print(ba)
        for k in range(N):
            if M>ba[k] and (k==0 or ba[k]<ba[k-1]): #we want ba[k] to increment by 1 (i.e. jth particle to hop right)
                new_idx = j + hs_sq[M-ba[k]-1,k+1] - hs_sq[M-ba[k],k+1]
                ham[new_idx,j] += -np.sqrt((ar[ba[k]-1])*(ar[ba[k]]+1))
            if 1<ba[k] and (k==(N-1) or ba[k]>ba[k+1]): #we want ba[k] to decrement by 1 (i.e. jth particle to hop right)
                new_idx = j + hs_sq[M-ba[k]+1,k+1] - hs_sq[M-ba[k],k+1]
                ham[new_idx,j] += -np.sqrt((ar[ba[k]-1])*(ar[ba[k]-2]+1))
    return ham.tocsr()


def hopping_ham_fast_disorder(M,N,readout,hs_sq,disar):
    assert np.shape(hs_sq)==(M+1,N+1)
    dim = hs_sq[M,N]
    ham = lil_matrix((dim,dim))
    nn_l = nearest_neighbors_1d(M)
    for j in range(dim):
        ar = readout[j]
        ba=Fock_vec_to_occ(ar)
        #print(ba)
        for k in range(N):
            if M>ba[k] and (k==0 or ba[k]<ba[k-1]): #we want ba[k] to increment by 1 (i.e. jth particle to hop right)
                new_idx = j + hs_sq[M-ba[k]-1,k+1] - hs_sq[M-ba[k],k+1]
                ham[new_idx,j] += -disar[ba[k]-1]*np.sqrt((ar[ba[k]-1])*(ar[ba[k]]+1))
            if 1<ba[k] and (k==(N-1) or ba[k]>ba[k+1]): #we want ba[k] to decrement by 1 (i.e. jth particle to hop right)
                new_idx = j + hs_sq[M-ba[k]+1,k+1] - hs_sq[M-ba[k],k+1]
                ham[new_idx,j] += -disar[ba[k]-2]*np.sqrt((ar[ba[k]-1])*(ar[ba[k]-2]+1))
    return ham.tocsr()

def U_ham_fast(M,N,readout):
    dim = hil_sp_dim(M,N)
    ham = lil_matrix((dim,dim))
    for j in range(dim):
        ar = readout[j]#idx_to_boson(j,M,N)
        ham[j,j] = np.dot(ar,ar-1)
    return ham.tocsr()   
    
def make_error_cs_boson_fast(M,N,k,readout):
    return readout[:,k]

def make_error_cdaggercs_boson_fast(M,N,k,readout):
    dim = hil_sp_dim(M,N)
    ham = lil_matrix((dim,dim))
    ham.setdiag(readout[:,k]**2)
    return ham.tocsr()
    
##Preparation error    
def hopping_ham_one_site_fast(M,N,readout,hs_sq):
    assert np.shape(hs_sq)==(M+1,N+1)
    dim = hs_sq[M,N]
    ham = lil_matrix((dim,dim))
    nn_l = nearest_neighbors_1d(M)
    for j in range(dim):
        ar = readout[j]
        ba=Fock_vec_to_occ(ar)
        #print(ba)
        for k in [M//2]:
            if M>ba[k] and (k==0 or ba[k]<ba[k-1]): #we want ba[k] to increment by 1 (i.e. jth particle to hop right)
                new_idx = j + hs_sq[M-ba[k]-1,k+1] - hs_sq[M-ba[k],k+1]
                ham[new_idx,j] += -np.sqrt((ar[ba[k]-1])*(ar[ba[k]]+1))
            if 1<ba[k] and (k==(N-1) or ba[k]>ba[k+1]): #we want ba[k] to decrement by 1 (i.e. jth particle to hop right)
                new_idx = j + hs_sq[M-ba[k]+1,k+1] - hs_sq[M-ba[k],k+1]
                ham[new_idx,j] += -np.sqrt((ar[ba[k]-1])*(ar[ba[k]-2]+1))
    return ham.tocsr()
    
##Initial state
def init_st_BH(M,N,hs_sq = []):
    if len(hs_sq)==0:
        hs_sq = hil_sp_dim_square(M,N)
    wfk0 = np.zeros(hs_sq[M,N])
    wfk0[boson_to_idx_fast(np.full(N,1),hs_sq)] = 1
    return wfk0
