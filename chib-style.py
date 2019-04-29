import math
import numpy as np

def getLogSumExp(log_weights):
    log_weights_max = np.max(log_weights)
    log_weights_z = log_weights - log_weights_max
    logSumExp = log_weights_max + math.log(sum(np.exp(log_weights)))
    return(logSumExp)

def getLogPzDoc(i_sample,n_topics,topic_prior,topic_alpha):
    Nd=len(i_sample)
    Nk,cls=np.histogram(i_sample, range(n_topics+1))
    log_pz=\
    + sum(map(math.lgamma, Nk+ topic_prior))\
    + math.lgamma(topic_alpha)\
    - sum(map(math.lgamma, topic_prior))\
    - math.lgamma(Nd + topic_alpha)
    return(log_pz)


def log_Tprob_base(zto, zfrom, Nz_, doc, phi, topic_prior):
    Nd = len(doc)
    lp = 0
    for t in range(Nd):
        Nz_[zfrom[t]] = Nz_[zfrom[t]]  - 1
        pz = phi[:, doc[t]]* (Nz_+topic_prior)
        pz = pz/pz.sum()
        lp = lp + math.log(pz[zto[t]])
        Nz_[zto[t]] = Nz_[zto[t]] + 1
    return(lp)

def getDiscreteSample(pz):
    if pz.sum()!=1:
        pz=pz/pz.sum()
    z_sample=np.argmax(np.random.multinomial(1, pz, 1), axis=1).tolist()[0]
    return(z_sample)

def getCSLogLikelihood(phi, doc, alpha=0.01,ms_iters=100):

    ms_iters=ms_iters
    b_iters=3
    c_iters=12
    Nd = len(doc);
    n_topics=phi.shape[0]
    if isinstance(alpha,float):
        topic_alpha = alpha*n_topics
        topic_prior = np.repeat(alpha,n_topics)
    if isinstance(alpha,list):
        topic_alpha = sum(alpha)
        topic_prior = alpha

    #% Assign latents to words in isolation as a simple initialization

    Nz = np.zeros((1, n_topics))[0]
    zz=np.zeros((1, Nd))[0].astype(int)

    for t in range(Nd):   
        pz=phi[:, doc[t]]* topic_prior
        zz_t= getDiscreteSample(pz)
        zz[t]=zz_t
        Nz[zz_t]=Nz[zz_t]+1

    #% Run some sweeps of Gibbs sampling
    for sweeps in range(b_iters):
        for t in range(Nd):
            Nz[zz[t]]=Nz[zz[t]]-1
            pz=pz = phi[:, doc[t]]* (Nz+topic_prior)
            zz_t=getDiscreteSample(pz)
            zz[t]=zz_t
            Nz[zz_t]=Nz[zz_t]+1

    #% Find local optimim to use as z^*, "iterative conditional modes"
    #% But don't spend forever on this, bail out if necessary
    for i in range(c_iters):
        old_zz =np.copy(zz)
        for t in range(Nd):
            Nz[zz[t]] = Nz[zz[t]]  - 1
            pz = phi[:, doc[t]]* (Nz+topic_prior)
            zz_t=np.argmax(pz)
            zz[t]=zz_t
            Nz[zz_t] = Nz[zz_t]  + 1

        if np.array_equal(old_zz,zz):
            break

    #% Run Murray & Salakhutdinov algorithm
    zstar = np.copy(zz) 
    log_Tvals = np.zeros((ms_iters, 1))

    #% draw starting position
    ss = np.random.choice(ms_iters, 1)

    #% Draw z^(s)
    for t in np.sort(range(Nd))[::-1].tolist():
        Nz[zz[t]]=Nz[zz[t]]-1
        pz = phi[:, doc[t]]* (Nz+topic_prior)
        zz_t= getDiscreteSample(pz)
        zz[t]=zz_t
        Nz[zz_t]=Nz[zz_t]+1

    zs = zz
    Ns = Nz

    Nz_aux=np.copy(Nz)
    log_Tvals[ss] = log_Tprob_base(zstar, zz, Nz_aux, doc, phi, topic_prior) 

    for sprime in range(ss+1,ms_iters):
        for t in range(Nd):
            Nz[zz[t]] = Nz[zz[t]]  - 1
            pz = phi[:, doc[t]]* (Nz+topic_prior)
            zz_t=getDiscreteSample(pz)
            zz[t]=zz_t
            Nz[zz_t]=Nz[zz_t]+1

        Nz_aux=np.copy(Nz)
        log_Tvals[sprime]=log_Tprob_base(zstar, zz, Nz_aux, doc, phi, topic_prior)

    #% Go back to middle
    zz = np.copy(zs)
    Nz = np.copy(Ns)

    #% Draw backward stuff
    for sprime in np.sort(range(ss))[::-1].tolist():
        for t  in np.sort(range(Nd))[::-1].tolist():
            Nz[zz[t]] = Nz[zz[t]]  - 1
            pz = phi[:, doc[t]]* (Nz+topic_prior)
            zz_t=getDiscreteSample(pz)
            zz[t]=zz_t
            Nz[zz_t]=Nz[zz_t]+1

        Nz_aux=np.copy(Nz)
        log_Tvals[sprime] = log_Tprob_base(zstar, zz, Nz_aux, doc, phi, topic_prior)

    #% Final estimate
    log_pz = getLogPzDoc(zstar,n_topics,topic_prior,topic_alpha)  

    log_w_given_z = 0;
    for t in range(Nd):
        log_w_given_z +=  math.log(phi[zstar[t] ,doc[t]])

    log_joint = log_pz + log_w_given_z
    log_evidence=log_joint- (getLogSumExp(log_Tvals)- math.log(ms_iters))
    return(log_evidence)
