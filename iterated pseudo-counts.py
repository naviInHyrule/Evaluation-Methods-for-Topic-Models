def getQq(phi, doc, alpha, max_iter):
    n_topics=len(phi)
    topic_alpha=alpha*n_topics
    topic_prior=np.repeat(alpha,n_topics)
    w_t= phi[:, doc].T
    qstar = topic_prior  * w_t
    qq = qstar / qstar.sum(axis=1)[:,None]                         
    for iteration in range(max_iter):       
        pseudo_counts = topic_prior + qq.sum(axis=0)-qq
        qstar = pseudo_counts*w_t
        qq = qstar / qstar.sum(axis=1)[:,None]  
    return(qq)
 
def getQqSamples(qq,num_samples):
    Nd=qq.shape[0]
    samples = np.zeros((Nd, num_samples))
    for n in range(Nd):
        samples[n, :]=np.argmax(np.random.multinomial(1, qq[n,:], size=num_samples), axis=1)
    return(samples)

def getLogPzDoc(i_sample,n_topics,alpha):
    Nd=len(i_sample)
    topic_alpha=alpha*n_topics
    topic_prior=np.repeat(alpha,n_topics)
    Nk,cls=np.histogram(i_sample, range(n_topics+1))
    log_pz=\
    + sum(map(math.lgamma, Nk+ topic_prior))\
    + math.lgamma(topic_alpha)\
    - sum(map(math.lgamma, topic_prior))\
    - math.lgamma(Nd + topic_alpha)
    return(log_pz)

def getLogSumExp(log_weights):
    log_weights_max = np.max(log_weights)
    log_weights_z = log_weights - log_weights_max
    logSumExp = log_weights_max + log(sum(exp(log_weights_z)))
    return(logSumExp)
                            
def getISLogLikelihood(phi, doc, alpha, max_iter,num_samples):
    n_topics=TERtopic.shape[0]
    Nd=len(doc)
    qq=getQq(TERtopic, doc, alpha, max_iter)
    samples=getQqSamples(qq,num_samples)
    log_pz=[]
    for i in range(num_samples):
        log_pz.append(getLogPzDoc(samples.T[i],n_topics,alpha))
    log_w_given_z = np.zeros((1, num_samples))[0]
    for n in range(Nd):
        log_w_given_z +=  log(phi[map(int,samples[n,:]) ,doc[n]]).tolist()
    log_joint=log_w_given_z+log_pz
    log_qq = np.zeros((1, num_samples))[0]
    for n  in range(Nd): 
        log_qq += log(qq[n, map(int,samples[n,:]) ]).tolist()
    log_weights=log_joint-log_qq
    log_evidence=getLogSumExp(log_weights)- log(len(log_weights))
    return(log_evidence)