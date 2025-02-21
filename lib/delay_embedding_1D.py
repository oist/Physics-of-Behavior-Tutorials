import numpy as np
import numpy.ma as ma


def trajectory_matrix(X,K):
    '''
    Build a trajectory matrix
    X: N x dim data
    K: the number of delays
    out: (N-K)x(dim*K) dimensional
    '''
    if ma.count_masked(X)>0:
        traj_matrix = ma.zeros((X.shape[0],X.shape[1]*(K+1)))
        traj_matrix[int(np.floor(K/2)):-int(np.ceil(K/2)+1)] = ma.vstack([ma.hstack(np.flip(X[t:t+K+1],axis=0)) for t in range(len(X)-K-1)])
        traj_matrix[np.any(traj_matrix.mask,axis=1)]=ma.masked
        traj_matrix[traj_matrix==0]=ma.masked
        return traj_matrix
    else:
        return np.vstack([np.hstack(np.flip(X[t:t+K+1],axis=0)) for t in range(len(X)-K-1)])

def whitening(X):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_mean = np.mean(X,axis=0)
    X_centered = X - X_mean
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    U, Lambda, V = np.linalg.svd(Sigma)
    W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
    return U,X_mean,W,np.dot(X_centered, W.T)

def embed_tseries(X,K,m=0,whiten=False,return_modes=False):
    '''
    Get delay embedding with K delays and m SVD dimensions
    K: number of delays
    m: number of SVD modes (default = 0 return the full trajectory matrix)
    out: if m>0 modes and projections, otherwise the full trajectory matrix
    '''
    traj_matrix = trajectory_matrix(X,K)
    if m>0:
        if whiten:
            if ma.count_masked(traj_matrix)>0:
                modes,mean_X,transformation,whitened_X = whitening(ma.compress_rows(traj_matrix))
                whitened = ma.zeros(traj_matrix.shape)
                whitened[~np.any(traj_matrix.mask,axis=1)] = whitened_X
                whitened[np.any(traj_matrix.mask,axis=1)] = ma.masked
                return modes,mean_X,transformation,whitened[:,:m]
            else:
                modes,mean_X,transformation,whitened_X = whitening(traj_matrix)
                return modes,mean_X,transformation,whitened_X[:,:m]
        else:
            if ma.count_masked(traj_matrix)>0:
                u,s,v=np.linalg.svd(ma.compress_rows(traj_matrix),full_matrices=False)
                phspace = ma.zeros((traj_matrix.shape[0],m))
                phspace[~np.any(traj_matrix.mask,axis=1),:]=u[:,:m]
                phspace[phspace==0]=ma.masked
            else:
                u,s,v=np.linalg.svd(traj_matrix,full_matrices=False)
                phspace = u[:,:m]
        if return_modes:
            return v,s,phspace
        else:
            return phspace
    else:
        phspace = traj_matrix
        return phspace
