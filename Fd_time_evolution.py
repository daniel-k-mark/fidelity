import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, linalg
import matplotlib.colors as colors
from scipy import optimize

from k_design import *
from spin_half_time_evolution import *
from Fc import *
from MFIM import *
from basis_change import *
from colors import *
from Hubbard import *
from spin_half_ED import *

def Frel(p,p0,pd):
    return (np.dot(p,np.array(p0)/pd)-1)/(np.dot(p0,np.array(p0)/pd)-1)

#Noiseless time evolution
def time_E(timear,ham,st0,interval=2):
    states = [st0]
    st = np.copy(st0)
    jjj = 0
    dt = timear[1]-timear[0]
    for t in timear[1:]:
        st = time_evolve(ham,st,dt,err=1e-10)
        jjj+=1
        if jjj%interval==0:
            states.append(st)        
    return states
    
def time_E2(timear,ham,st0,dt):
    states = [st0]
    st = np.copy(st0)
    jjj = 0
    interval_num = (timear[1]-timear[0])/dt
    assert np.isclose(np.round(interval_num,0),interval_num)
    while jjj < timear[-1]/dt:
        st = time_evolve(ham,st,dt,err=1e-10)
        jjj+=1
        if np.isclose(np.product(np.array(timear)-jjj*dt),0):
            states.append(st)        
    return states
    
def time_evolve_exact(time,cs,vals,vecs):
    return np.matmul(vecs,np.multiply(cs,np.exp(-1j*time*vals)))
    
def time_E_exact(timear,vals,vecs,wfk0):
    cs = np.conj(np.matmul(np.conj(wfk0),vecs))
    states0 = np.einsum('zE,tE->tz',vecs,np.multiply(np.exp(-1j*np.outer(timear,vals)),cs))
    return states0
    
    
def pdiag_exact(state0,vecs):
    p_en = np.abs(np.dot(np.transpose(np.conj(state0)),vecs))**2
    pdiag = np.dot(np.abs(vecs)**2,p_en)
    return pdiag
    
def effective_dimension_exact(st0,vecs,pdiag):
    cs=np.einsum('z,zE->E',np.conj(st0),vecs)
    return np.matmul(np.abs(vecs)**4,np.abs(cs)**4)/pdiag**2
    
def time_E_Fd(timear,ham,st0,states0,pdiag,interval=2):
    Fds = [Fcd(make_p(st0),make_p(states0[0]),pdiag)]
    st = np.copy(st0)
    jjj = 0
    dt = timear[1]-timear[0]
    for t in timear[1:]:
        st = time_evolve(ham,st,dt,err=1e-10)
        jjj+=1
        if jjj%interval==0:
            p = make_p(st)
            Fds.append(Fcd(p,make_p(states0[jjj]),pdiag))        
    return Fds
        
def time_E_Fd_Frel(timear,ham,st0,states0,pdiag,interval=2):
    Fds = [Fcd(make_p(st0),make_p(states0[0]),pdiag)]
    Frels = [Frel(make_p(st0),make_p(states0[0]),pdiag)]
    st = np.copy(st0)
    jjj = 0
    dt = timear[1]-timear[0]
    for t in timear[1:]:
        st = time_evolve(ham,st,dt,err=1e-10)
        jjj+=1
        if jjj%interval==0:
            p = make_p(st)
            Fds.append(Fcd(p,make_p(states0[jjj]),pdiag))
            Frels.append(Frel(p,make_p(states0[jjj]),pdiag))        
    return Fds, Frels

def time_E_Fd_Frel(timear,ham,st0,states0,pdiag,interval=1):
    Fds = [Fcd(make_p(st0),make_p(states0[0]),pdiag)]
    Frels = [Frel(make_p(st0),make_p(states0[0]),pdiag)]
    st = np.copy(st0)
    jjj = 0
    dt = timear[1]-timear[0]
    for t in timear[1:]:
        st = time_evolve(ham,st,dt,err=1e-10)
        jjj+=1
        if jjj%interval==0:
            p = make_p(st)
            Fds.append(Fcd(p,make_p(states0[jjj]),pdiag))
            Frels.append(Frel(p,make_p(states0[jjj]),pdiag))        
    return Fds, Frels

def time_E_FXEB_varFXEB(timear,ham,st0,states0,pdiag,interval=1):
    FXEBs = [np.dot(make_p(st0),make_p(states0[0])/pdiag)]
    varFXEBs = [np.dot(make_p(st0),(make_p(states0[0])/pdiag)**2)-FXEBs[-1]**2]
    st = np.copy(st0)
    jjj = 0
    dt = timear[1]-timear[0]
    for t in timear[1:]:
        st = time_evolve(ham,st,dt,err=1e-10)
        jjj+=1
        if jjj%interval==0:
            pz = make_p(st)
            p0z_tilde = make_p(states0[jjj])/pdiag
            FXEBs.append(np.dot(pz,p0z_tilde))
            varFXEBs.append(np.dot(pz,p0z_tilde**2)-FXEBs[-1]**2)        
    return FXEBs, varFXEBs

def make_pdiag(timear,ham,st0):
    p_d = np.abs(st0)**2
    st = np.copy(st0)
    jjj = 0
    dt = timear[1]-timear[0]
    for t in timear[1:]:
        st = time_evolve(ham,st,dt,err=1e-10)
        jjj+=1
        p_d += np.abs(st)**2        
    assert np.isclose(sum(p_d),jjj+1)
    return p_d/(jjj+1)

def pdiag_est(states):
    return np.mean(np.abs(states)**2,axis=0)
    
def make_Fideal_d(states0,pdiag):
    return [Fideal_d(np.abs(st)**2,pdiag) for st in states0]
    
#Single error
def overlap(th,ov,st0,fn):
    #stt = linalg.expm_multiply(-1j*th*ham_rot,st0)
    stt = fn(st0,th)
    return np.abs(np.vdot(stt,st0))**2-ov

#Noisy time evolution
rng = np.random.default_rng(12345)
def update_step(dt,heff,st0,r_res,error_ops=[],L=0):
    ##Time evolve with non-Hermitian Hamiltoninan
    st_t = linalg.expm_multiply(-1j*dt*heff,st0)
    perr = 1-np.linalg.norm(st_t)**2
    ##If random_number is less than probability of error
    if r_res< perr:
        error_states = [e.dot(st_t) for e in error_ops]
        #Calculate which error occured
        error_probs = [np.linalg.norm(es)**2 for es in error_states]
        r2 = rng.uniform(low=0, high=1)        
        pp = np.cumsum(error_probs)/np.sum(error_probs)        
        error_num = sum(r2>pp)        
        #pp = perr*np.cumsum(error_probs)/np.sum(error_probs)
        #error_num = sum(r_res>pp)
        #Return the new state
        new_state = error_states[error_num]/np.linalg.norm(error_states[error_num])
        return new_state, rng.uniform(low=0, high=1)
    else:
        return st_t, r_res
        
def update_step_diagonal(dt,heff,st0,r_res,error_ops=[],L=0):
    ##Time evolve with non-Hermitian Hamiltoninan
    st_t = linalg.expm_multiply(-1j*dt*heff,st0)
    perr = 1-np.linalg.norm(st_t)**2
    ##If random_number is less than probability of error
    if r_res< perr:
        error_states = [np.multiply(e,st_t) for e in error_ops]
        #Calculate which error occured
        error_probs = [np.linalg.norm(es)**2 for es in error_states]
        r2 = rng.uniform(low=0, high=1)        
        pp = np.cumsum(error_probs)/np.sum(error_probs)        
        #pp = perr*np.cumsum(error_probs)/np.sum(error_probs)
        error_num = sum(r2>pp)
        #Return the new state
        new_state = error_states[error_num]/np.linalg.norm(error_states[error_num])
        return new_state, rng.uniform(low=0, high=1)
    else:
        return st_t, r_res

### Specific to spin-1/2 Pauli errors
def update_step_Pauli(dt,heff,st0,r_res,error_ops=[],L=0):
    sigZ = np.array([[1,0],[0,-1]])
    sigX = np.array([[0,1],[1,0]])
    st_t = linalg.expm_multiply(-1j*dt*heff,st0)
    perr = 1-np.linalg.norm(st_t)**2
    if r_res< perr:
        error_states = [np.reshape(np.einsum('ij,kjl',sigZ,np.reshape(st_t,(2**k,2,-1))),-1) for k in range(L)]
        error_states += [np.reshape(np.einsum('ij,kjl',sigX,np.reshape(st_t,(2**k,2,-1))),-1) for k in range(L)]
        error_probs = [np.linalg.norm(e)**2 for e in error_states]
        pp = perr*np.cumsum(error_probs)/np.sum(error_probs)
        error_num = sum(r_res>pp)
        new_state = error_states[error_num]/np.linalg.norm(error_states[error_num])
        return new_state, rng.uniform(low=0, high=1)
    else:
        return st_t, r_res


def make_dm(heff,wfk0,timear,states0,pdiag,update_fn,nums=100,interval=1,error_ops=[]):
    dt = timear[1]-timear[0]
    L = int(np.log2(len(wfk0)))
    p0 = [make_p(st) for st in states0]
    ps = [nums*np.abs(wfk0)**2]
    fid = [nums]
    for kk in range(nums):
        jj = 0
        st = np.copy(wfk0)
        r_res = rng.uniform(low=0, high=1)
        for t in timear:
            st, r_res = update_fn(dt,heff,st,r_res,error_ops=error_ops,L=L)
            jj+=1
            if jj%interval==0:
                stn = st/np.linalg.norm(st)
                if kk==0:
                    ps.append(np.abs(stn)**2)
                    fid.append(np.abs(np.vdot(stn,states0[jj//interval]))**2)
                else:
                    ps[jj//interval] += np.abs(stn)**2
                    fid[jj//interval] += np.abs(np.vdot(stn,states0[jj//interval]))**2
        print(kk,end="\r")
    ps = np.array(ps)/nums
    Fcs = []
    Fds = []
    Fd_ls = []
    for j in range(len(ps)):
        Fcs.append(Fc(ps[j],p0[j]))
        Fds.append(Fcd(ps[j],p0[j],pdiag))
        Fd_ls.append(np.dot(ps[j],p0[j]/pdiag)-1)
    fid = np.array(fid)/nums
    return Fcs,Fds,Fd_ls,fid
    
def make_dm_new(dt,heff,wfk0,timear,states0,pdiag,update_fn,nums=100,error_ops=[]):
    dt_factor = (timear[1]-timear[0])/dt
    assert(np.isclose(int(dt_factor),dt_factor))
    dt_factor = int(dt_factor)
    L = int(np.log2(len(wfk0))) #This can be a meaningless variable (just for spin-half update fn)
    #Bitstring speckle patterns
    p0 = [make_p(st) for st in states0]
    ptilde = np.einsum('tz,z->tz',p0,1/pdiag)
    FXEBsp1 = [[] for kk in range(nums)]
    fids = [[] for kk in range(nums)]
    #For each trajectory, compute FXEB, fid
    ll = len(timear)
    for kk in range(nums):
        jj = 0
        time = 0
        st = np.copy(wfk0)
        r_res = rng.uniform(low=0, high=1)
        while jj < len(timear)*dt_factor-1:
            st, r_res = update_fn(dt,heff,st,r_res,error_ops=error_ops,L=L)
            jj+=1
            #time += dt
            if jj%dt_factor==0:
                stn = st/np.linalg.norm(st)
                FXEBsp1[kk].append(np.dot(np.abs(stn)**2,ptilde[jj//dt_factor]))
                fids[kk].append(np.abs(np.vdot(stn,states0[jj//dt_factor]))**2)
        #print(kk,end="\r")
    return FXEBsp1, fids
    
def make_dm_qz(dt,heff,wfk0,timear,states0,update_fn,nums=100,error_ops=[]):
    dt_factor = (timear[1]-timear[0])/dt
    assert(np.isclose(int(dt_factor),dt_factor))
    dt_factor = int(dt_factor)
    L = int(np.log2(len(wfk0)))
    #p0 = [make_p(st) for st in states0]
    ps = [nums*np.abs(wfk0)**2]
    fid = [nums]
    for kk in range(nums):
        jj = 0
        st = np.copy(wfk0)
        r_res = rng.uniform(low=0, high=1)
        while jj < len(timear)*dt_factor-1:
            st, r_res = update_fn(dt,heff,st,r_res,error_ops=error_ops,L=L)
            jj+=1
            if jj%dt_factor==0:
                stn = st/np.linalg.norm(st)
                if kk==0:
                    ps.append(np.abs(stn)**2)
                    fid.append(np.abs(np.vdot(stn,states0[jj//dt_factor]))**2)
                else:
                    ps[jj//dt_factor] += np.abs(stn)**2
                    fid[jj//dt_factor] += np.abs(np.vdot(stn,states0[jj//dt_factor]))**2
        print(kk,end="\r")
    ps = np.array(ps)/nums
    fid = np.array(fid)/nums
    return ps, fid
    
##########
def Fcd(p,p0,pd):
    #assert np.isclose(sum(p),1) and np.isclose(sum(p0),1)
    return 2*(np.dot(p,np.array(p0)/pd)/np.dot(p0,np.array(p0)/pd)) -1

def Fd_var(p,p0,pd):
    return (2/np.dot(p0,np.array(p0)/pd))*np.sqrt(np.dot(p,(np.array(p0)/pd)**2)-np.dot(p,(np.array(p0)/pd))**2)
    
##Estimating the Renyi 2-entropy of the diagonal ensemble
def S2_est(states0,ll=50):
    return np.mean([np.abs(np.vdot(states0[0],s))**2 for s in states0[ll:]])
