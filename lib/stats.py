import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma
from scipy import ndimage as ndi


def bootstrap(l,n_times,n_samples=10000,confidence_interval=95,median=False,maximum=False,log=False):
    if n_samples>len(l):
        n_samples=len(l)*2
    if median:
        per=(100-confidence_interval)/2
        new_means=[]
        for i in range(n_times):
            indices=np.random.choice(range(0,len(l)),n_samples)#len(l))
            new_list=[l[idx] for idx in indices]
            new_means.append(ma.median(new_list,axis=0))
        new_means=ma.vstack(new_means)
        cil=np.zeros(new_means.shape[1])
        ciu=np.zeros(new_means.shape[1])
        for i in range(new_means.shape[1]):
            cil[i]=np.nanpercentile(new_means[:,i].filled(np.nan),per)
            ciu[i]=np.nanpercentile(new_means[:,i].filled(np.nan),100-per)
        cil = ma.masked_array(cil, np.isnan(cil))
        ciu = ma.masked_array(ciu, np.isnan(ciu))
        return ma.median(l,axis=0),cil,ciu
    elif maximum:
        per=(100-confidence_interval)/2
        new_means=[]
        for i in range(n_times):
            indices=np.random.choice(range(0,len(l)),n_samples)#len(l))
            new_list=[l[idx] for idx in indices]
            new_means.append(ma.max(new_list,axis=0))
        new_means=ma.vstack(new_means)
        cil=np.zeros(new_means.shape[1])
        ciu=np.zeros(new_means.shape[1])
        for i in range(new_means.shape[1]):
            cil[i]=np.nanpercentile(new_means[:,i].filled(np.nan),per)
            ciu[i]=np.nanpercentile(new_means[:,i].filled(np.nan),100-per)
        cil = ma.masked_array(cil, np.isnan(cil))
        ciu = ma.masked_array(ciu, np.isnan(ciu))
        return ma.max(l,axis=0),cil,ciu
    elif log:
        per=(100-confidence_interval)/2
        new_means=[]
        for i in range(n_times):
            indices=np.random.choice(range(0,len(l)),n_samples)#len(l))
            new_list=[np.log10(l[idx]) for idx in indices]
            new_means.append((ma.mean(new_list,axis=0)))
        new_means=10**(ma.vstack(new_means))
        cil=np.zeros(new_means.shape[1])
        ciu=np.zeros(new_means.shape[1])
        for i in range(new_means.shape[1]):
            cil[i]=np.nanpercentile(new_means[:,i].filled(np.nan),per)
            ciu[i]=np.nanpercentile(new_means[:,i].filled(np.nan),100-per)
        cil = ma.masked_array(cil, np.isnan(cil))
        ciu = ma.masked_array(ciu, np.isnan(ciu))
        return 10**(ma.mean(np.log10(np.array(l)),axis=0)),cil,ciu
    else:
        per=(100-confidence_interval)/2
        new_means=[]
        for i in range(n_times):
            indices=np.random.choice(range(0,len(l)),n_samples)#len(l))
            new_list=[l[idx] for idx in indices]
            new_means.append(ma.mean(new_list,axis=0))
        new_means=ma.vstack(new_means)
        cil=np.zeros(new_means.shape[1])
        ciu=np.zeros(new_means.shape[1])
        for i in range(new_means.shape[1]):
            cil[i]=np.nanpercentile(new_means[:,i].filled(np.nan),per)
            ciu[i]=np.nanpercentile(new_means[:,i].filled(np.nan),100-per)
        cil = ma.masked_array(cil, np.isnan(cil))
        ciu = ma.masked_array(ciu, np.isnan(ciu))
        return ma.mean(l,axis=0),cil,ciu

def histcount(array,binlims):
    binmin,binmax=binlims
    c=0
    for x in array:
        if binmin<x<binmax:
            c+=1
    return c

def gaussian(x,a,b,c):
    return a*np.exp(-((x-b)**2)/(2*c**2))


def histogram(array,n_bins,xmin,xmax,norm_amp=False,norm_var=False,plot=False,fit_gaussian=False):
    '''
    Returns the bincenters, the frequencies and the binsizes of an histogram.
    In case fit_gaussian=True, returns also the gaussian fitted to the histogram (r,g)
    xmin and xmax are the points where we want to start or end an histogram. In case norm_var=True,
    this is just the distance from the mean, in either side
    norm_var normalizes by the variance
    norm_amp normalizes by the amplitude
    plot plots the histogram
    fit_gaussian fits a gaussian distribution to the histogram
    '''
    binsize=(xmax-xmin)/float(n_bins)
    bincenters=[]
    x=xmin
    mean=array.mean()
    while x<xmax:
        bincenters.append(x+binsize/2.)
        x+=binsize
    if norm_var:
        array=array-array.mean()
        array=array/array.std()
        histogram=[]
        for b in bincenters:
            binlims=(b-binsize/2.,b+binsize/2.)
            histogram.append(histcount(array,binlims))
    else:
        histogram=[]
        for b in bincenters:
            binlims=(b-binsize/2.,b+binsize/2.)
            histogram.append(histcount(array,binlims))
    hist=np.array(histogram)
    if norm_amp:
        hist=hist/np.float(np.sum(hist))
    if fit_gaussian:
        sigma=array.std()
        a=hist.max()
        if norm_var:
            sigma=1
        r=np.linspace(xmin-binsize/2.,xmax+binsize/2.,n_bins*10)
        g=gaussian(r,a,array.mean(),sigma)
        print(sigma,a,array.mean())
        if norm_var:
            plt.plot(r+mean,g,c='k')
        else:
            plt.plot(r,g,c='k')
    if plot:
        if norm_var:
            plt.bar(np.array(bincenters)+mean,hist,binsize)
        else:
            plt.bar(np.array(bincenters),hist,binsize)
        plt.show()
    if norm_var and fit_gaussian:
        return bincenters,hist,binsize,r+mean,g
    if norm_var==False and fit_gaussian:
        return bincenters,hist,binsize,r,g
    else:
        return bincenters,hist,binsize


def acf(x, lags=500, exclude=None):
    if exclude is None:
        exclude = np.zeros(x.shape)
    exclude = np.cumsum(exclude.astype(int))

    # from stackexchange
    x = x - x.mean()  # remove mean
    if type(lags) is int:
        lags = range(lags)

    C = ma.zeros((len(lags),))
    sigma2 = x.var()
    for i, l in enumerate(lags):
        if l == 0:
            C[i] = 1
        elif l >= x.shape[0]:
            C[i] = ma.masked
        else:
            x0 = x[:-l].copy()
            x1 = x[l:].copy()
            reject = (exclude[l:]-exclude[:-l])>0
            x0[reject] = ma.masked
            x1[reject] = ma.masked
            C[i] = (x0*x1).mean()/sigma2
    return C


def ccf(x, y, lags, exclude=None):
    if exclude is None:
        exclude = np.zeros(x.shape)
    exclude = np.cumsum(exclude.astype(int))

    x = x - x.mean()  # remove mean
    y = y - y.mean()
    if type(lags) is int:
        lags = np.arange(-lags,lags)
    C = ma.zeros((len(lags),))
    sigma2 = x.std()*y.std()
    for i, l in enumerate(lags):
        if l == 0:
            C[i] = (x*y).mean()/sigma2
        else:
            if l > 0:
                x0 = x[:-l].copy()
                y1 = y[l:].copy()
            else:
                x0 = y[:l].copy()
                y1 = x[-l:].copy()
            reject = (exclude[l:]-exclude[:-l])>0
            x0[reject] = ma.masked
            y1[reject] = ma.masked

            C[i] = (x0*y1).mean()/sigma2
    return C

def dotacf(x, lags=500, exclude=None):
    if exclude is None:
        exclude = np.zeros(x.shape)
    exclude = np.cumsum(exclude.astype(int))

    if type(lags) is int:
        lags = range(lags)
    C = ma.zeros((len(lags),))
    size=len(x)
    for i, l in enumerate(lags):
        if l == 0:
            C[i] = (x*x).sum(axis=1).mean()
        else:
            x0 = x[:-l, :].copy()
            x1 = x[l:, :].copy()
            reject = (exclude[l:]-exclude[:-l])>0
            x0[reject, :] = ma.masked
            x1[reject, :] = ma.masked
            C[i] = (x0*x1).sum(axis=1).mean()
    return C


def density_plot(X,Y,xrange,yrange,n_grid_x,n_grid_y,border=5,smooth=False,log=False):
    # view area range
    view_xmin,view_xmax=xrange
    view_ymin,view_ymax=yrange

    # get data
    xl = X
    yl = Y

    # get visible data points
    xlvis = []
    ylvis = []
    for i in range(0,len(xl)):
        if view_xmin < xl[i] < view_xmax and view_ymin < yl[i] < view_ymax:
            xlvis.append(xl[i])
            ylvis.append(yl[i])

    kx = (n_grid_x - 1) / (view_xmax - view_xmin)
    ky = (n_grid_y - 1) / (view_ymax - view_ymin)
    imgw = (n_grid_x + 2 * border)
    imgh = (n_grid_y + 2 * border)
    img = np.zeros((imgh,imgw))
    for x, y in zip(xl,yl):
        ix = int((x - view_xmin) * kx) + border
        iy = int((y - view_ymin) * ky) + border
        if 0 <= ix < imgw and 0 <= iy < imgh:
            img[iy][ix] += 1
    if smooth:
        if log:
            log_img=np.log10(img)
            sel=~np.isfinite(log_img)
            log_img[sel]=0.
            filtered_img=ndi.gaussian_filter(log_img, (border,border))
            return filtered_img
        else:
            return ndi.gaussian_filter(img, (border,border))  ## gaussian convolution
    else:
        if log:
            return np.log10(img)
        else:
            return img

def KLDiv(P, Q):
    if P.shape[0] != Q.shape[0]:
        raise Exception()
    if np.any(~np.isfinite(P)) or np.any(~np.isfinite(Q)):
        raise Exception()
    Q = Q / Q.sum()
    P = P / P.sum(axis=0)
    dist = np.sum(P*np.log2(P/Q), axis=0)
    if np.isnan(dist):
        dist = 0
    return dist


def JSDiv(P, Q):
    if P.shape[0] != Q.shape[0]:
        raise Exception()
    Q = Q / Q.sum(axis=0)
    P = P / P.sum(axis=0)
    M = 0.5*(P+Q)
    dist = 0.5*KLDiv(P, M) + 0.5*KLDiv(Q, M)
    return dist
