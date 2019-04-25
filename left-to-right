import math
import numpy as np


def getDiscreteSample(pz):
    pz=pz/pz.sum()
    z_sample=np.argmax(np.random.multinomial(1, pz, 1), axis=1).tolist()[0]
    return(z_sample)

def getLtoR(phi, doc, alpha):
    Nd = len(doc);
    n_topics=phi.shape[0]
    if isinstance(alpha,float):
        topic_alpha = alpha*n_topics
        topic_prior = np.repeat(alpha,n_topics)
    if isinstance(alpha,list):
        topic_alpha = sum(alpha)
        topic_prior = alpha

    #% Run left-to-right
    pn=[]
    Ns = np.zeros((1, n_topics))[0]
    ss = np.zeros((1, Nd))[0].astype(int)
    for n in range(Nd):       
        #lines 5-7
        for n_ in range(n):
            #print('n ',str(n),' r ',str(r),' n_ ',str(n_))
            Ns[ss[n_]] = Ns[ss[n_]]  - 1
            N=sum(Ns)
            ps = phi[:, doc[n_]]* (Ns+topic_prior)/(N +sum(topic_prior))
            ss_t = getDiscreteSample(ps)#sample new topic
            #print('n ',str(n),' r ',str(r),' n_ ',str(n_), 'ss_t ',str(ss_t))
            ss[n_]=ss_t
            Ns[ss_t] = Ns[ss_t]  + 1
        N=sum(Ns)
        ps = phi[:, doc[n]]* (Ns+topic_prior)/(N +sum(topic_prior))

        pn.append(sum(ps))#line 8

        ss_t = getDiscreteSample(ps)#line 9 : sample new topic
        ss[n] = ss_t
        Ns[ss_t] = Ns[ss_t]  + 1

    return(pn)

def getLtoRLogLikelihood(phi, doc, alpha, particles):
    pn=[]
    for i in range(particles):
        pn.append(getLtoR(phi, doc, alpha))
    pn=np.mean(pn,axis=0)
    llk=sum(map(math.log,pn))
    return(llk)
