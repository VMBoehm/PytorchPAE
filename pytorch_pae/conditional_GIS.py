import sys
import numpy as np
import sinf
from sinf import *
from sinf.GIS import *


def _conditional_transform_batch_model(model, data, label, logj, index, batchsize, start_index=0, end_index=None, start=0, end=None, param=None, nocuda=False):

    if torch.cuda.is_available() and not nocuda:
        gpu    = index % torch.cuda.device_count()
        device = torch.device('cuda:%d'%gpu)
    else:
        device = torch.device('cpu')

    model = model.to(device)

    if end_index is None:
        end_index = len(data)

    i = 0
    while i * batchsize < end_index-start_index:
        start_index0 = start_index + i * batchsize 
        end_index0 = min(start_index + (i+1) * batchsize, end_index) 
        if param is None:
            data1, logj1 = model.transform(data[start_index0:end_index0].to(device), label[start_index0:end_index0].to(device), start=start, end=end, param=param)
        else:
            data1, logj1 = model.transform(data[start_index0:end_index0].to(device), label[start_index0:end_index0].to(device), start=start, end=end, param=param[start_index0:end_index0].to(device))
        data[start_index0:end_index0] = data1.to(data.device)
        logj[start_index0:end_index0] = logj[start_index0:end_index0] + logj1.to(logj.device)
        i += 1

    del data1, logj1, model 
    if torch.cuda.is_available() and not nocuda:
        torch.cuda.empty_cache()

    return


def conditional_transform_batch_model(model, data, label, batchsize, logj=None, start=0, end=None, param=None, pool=None, nocuda=False):

    if logj is None:
        logj = torch.zeros(len(data), device=data.device)

    if pool is None: 
        _transform_batch_model(model, data, label, logj, 0, batchsize, start=start, end=end, param=param, nocuda=nocuda) 
    else:
        if torch.cuda.is_available() and not nocuda:
            nprocess = torch.cuda.device_count()
        else:
            nprocess = mp.cpu_count()
        param0 = [(model, data, label, logj, i, batchsize, len(data)*i//nprocess, len(data)*(i+1)//nprocess, start, end, param, nocuda) for i in range(nprocess)]
        pool.starmap(_conditional_transform_batch_model, param0)

    return data, logj


def _conditional_transform_batch_layer(layer, data, label, logj, index, batchsize, start_index=0, end_index=None, direction='forward', param=None, nocuda=False):

    if torch.cuda.is_available() and not nocuda:
        gpu = index % torch.cuda.device_count()
        device = torch.device('cuda:%d'%gpu)
    else:
        device = torch.device('cpu')
    
    layer = layer.to(device)

    if end_index is None:
        end_index = len(data)

    i = 0
    while i * batchsize < end_index-start_index:
        start_index0 = start_index + i * batchsize 
        end_index0 = min(start_index + (i+1) * batchsize, end_index) 
        if direction == 'forward': 
            if param is None:
                data1, logj1 = layer.forward(data[start_index0:end_index0].to(device),label[start_index0:end_index0].to(device), param=param)
            else:
                data1, logj1 = layer.forward(data[start_index0:end_index0].to(device), label[start_index0:end_index0].to(device), param=param[start_index0:end_index0].to(device))
        else: 
            if param is None:
                data1, logj1 = layer.inverse(data[start_index0:end_index0].to(device),label[start_index0:end_index0].to(device), param=param)
            else:
                data1, logj1 = layer.inverse(data[start_index0:end_index0].to(device),label[start_index0:end_index0].to(device), param=param[start_index0:end_index0].to(device))
        data[start_index0:end_index0] = data1.to(data.device)
        logj[start_index0:end_index0] = logj[start_index0:end_index0] + logj1.to(logj.device)
        i += 1

    del data1, logj1, layer 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return


def conditional_transform_batch_layer(layer, data, label, batchsize, logj=None, direction='forward', param=None, pool=None, nocuda=False):
    assert direction in ['forward', 'inverse']
    
    if logj is None:
        logj = torch.zeros(len(data), device=data.device)
    
    if pool is None: 
        _conditional_transform_batch_layer(layer, data, label, logj, 0, batchsize, direction=direction, param=param, nocuda=nocuda) 
    else:
        if torch.cuda.is_available() and not nocuda:
            nprocess = torch.cuda.device_count()
        else:
            nprocess = mp.cpu_count()
        param0 = [(layer, data, label, logj, i, batchsize, len(data)*i//nprocess, len(data)*(i+1)//nprocess, direction, param, nocuda) for i in range(nprocess)]
        pool.starmap(_conditional_transform_batch_layer, param0)
    
    return data, logj


class ConditionalSINF(nn.Module):

    #Sliced Iterative Normalizing Flow model
    
    def __init__(self, ndim, n_class):
        
        super().__init__()
        
        self.layer = nn.ModuleList([])
        self.ndim = ndim
        self.n_class = n_class
    
    def forward(self, data, label, start=0, end=None):
        
        if data.ndim == 1:
            data = data.view(1,-1)
        if end is None:
            end = len(self.layer)
        elif end < 0:
            end += len(self.layer)
        if start < 0:
            start += len(self.layer)
        
        assert start >= 0 and end >= 0 and end >= start

        logj = torch.zeros(data.shape[0], device=data.device)
        
        for i in range(start, end):
            data, log_j = self.layer[i](data, param=label)
            logj += log_j

        return data, logj
    
    
    def inverse(self, data, label, start=None, end=0, d_dz=None):

        if data.ndim == 1:
            data = data.view(1,-1)
        if end < 0:
            end += len(self.layer)
        if start is None:
            start = len(self.layer)
        elif start < 0:
            start += len(self.layer)
        
        assert start >= 0 and end >= 0 and end <= start

        logj = torch.zeros(data.shape[0], device=data.device)
        
        for i in reversed(range(end, start)):
            if d_dz is None:
                data, log_j = self.layer[i].inverse(data, param=label)
            else:
                data, log_j, d_dz = self.layer[i].inverse(data,d_dz=d_dz, param=label)
            logj += log_j

        if d_dz is None:
            return data, logj
        else:
            return data, logj, d_dz


    def transform(self, data, label, start, end,):

        if start is None:
            return self.inverse(data=data, start=start, end=end, param=label) 
        elif end is None:
            return self.forward(data=data, start=start, end=end, param=label) 
        elif start < 0:
            start += len(self.layer)
        elif end < 0:
            end += len(self.layer)
        
        if start < 0:
            start = 0
        elif start > len(self.layer):
            start = len(self.layer)
        if end < 0:
            end = 0
        elif end > len(self.layer):
            end = len(self.layer)

        if start <= end:
            return self.forward(data=data, start=start, end=end, param=label) 
        else:
            return self.inverse(data=data, start=start, end=end, param=label) 
    
    
    def add_layer(self, layer, position=None):
        
        if position is None or position == len(self.layer):
            self.layer.append(layer)
        else:
            if position < 0:
                position += len(self.layer)
            assert position >= 0 and position < len(self.layer)
            self.layer.insert(position, layer)
        
        return self
    
    
    def delete_layer(self, position=-1):
        
        if position == -1 or position == len(self.layer)-1:
            self.layer = self.layer[:-1]
        else:
            if position < 0:
                position += len(self.layer)
            assert position >= 0 and position < len(self.layer)-1
            
            for i in range(position, len(self.layer)-1):
                self.layer._modules[str(i)] = self.layer._modules[str(i + 1)]
            self.layer = self.layer[:-1]
        
        return self
    
    
    def evaluate_density(self, data, label, start=0, end=None):
        
        data, logj = self.forward(data, label, start=start, end=end)
        logq = -self.ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(data.reshape(len(data), self.ndim)**2,  dim=1)/2
        logp = logj + logq
        
        return logp


    def loss(self, data, start=0, end=None, param=None):
        return -torch.mean(self.evaluate_density(data, label, start=start, end=end, param=param))
    
    
    def sample(self, nsample, start=None, end=0, device=torch.device('cuda'), param=None):

        #device must be the same as the device of the model
        
        x       = torch.randn(nsample, self.ndim, device=device)
        logq    = -self.ndim/2.*torch.log(torch.tensor(2.*math.pi)) - torch.sum(x**2,  dim=1)/2
        x, logj = self.inverse(x, start=start, end=end, param=param)
        logp    = logj + logq

        return x, logp


    def score(self, data, label, start=0, end=None, param=None):

        #returns score = dlogp / dx

        data.requires_grad_(True)
        logp  = torch.sum(self.evaluate_density(data, label, start, end, param))
        score = torch.autograd.grad(logp, data)[0]
        data.requires_grad_(False)

        return score
    
    
def train_ConditionalGIS(data_train, label_train, data_valid, label_valid, iteration=None, K=None, M=None, KDE=True, b_factor=1, alpha=None, bounds=None,max_iter=400,
        edge_bins=None, ndata_A=None, MSWD_max_iter=None, NBfirstlayer=False, Whiten=True, batchsize=None, nocuda=False, patch=False, shape=None, model=None, verbose=True):
    
    '''
    data_train: (ndata_train, ndim).
    data_valid: (ndata_valid, ndim), optional. If provided, its logp will be used to determine the number of iterations.
    iteration: integer, optional. The maximum number of GIS iterations. Required if data_valid is not provided.
    K: integer, optional. The number of slices for each iteration. See max K-SWD in the SINF paper. 1 <= K <= ndim.
    M: integer, optional. The number of spline knots for rational quadratic splines.
    KDE: bool. Whether to use KDE for estimating 1D PDF. Recommended True.
    b_factor: positive float number, optional. The multiplicative factor for KDE kernel width.
    alpha: two non-negative float number in the format of (alpha1, alpha2), optional. Regularization parameter. See Equation 13 of SINF paper. alpha1 for interpolation, alpha2 for extrapolation slope. 0 <= alpha1,2 < 1. If not given, very heavy regularization will be used, which could result in slow training and a large number of iterations.
    bounds: sequence, optional. In the format of [[x1_min, x1_max], [x2_min, x2_max], ..., [xd_min, xd_max]]. Represent infinity and negative infinity with None.
    edge_bins: non-negative integer, optional. The number of spline knots at the boundary.
    ndata_A: positive integer, optional. The number of training data used for fitting A (slice axes).
    MSWD_max_iter: positive integer, optional. The maximum number of iterations for optimizing A (slice axes). See Algorithm 1 of SINF paper. Called L_iter in the paper.
    NBfirstlayer: bool, optional. Whether to use Naive Bayes (no rotation) at the first layer.
    Whiten: bool, optional. Whether to whiten the data before applying GIS.
    batchsize: positive integer, optional. The batch size for transforming the data. Does not change the performance. Only saves the memory. 
    Useful when the data is too large and can't fit in the memory.
    nocuda: bool, optional. Whether to use gpu.
    patch: bool, optional. Whether to use patch-based modeling. Only useful for image datasets.
    shape: sequence, optional. The shape of the image datasets, if patch is enabled.
    model: GIS model, optional. Trained GIS model. If provided, new iterations will be added in the model.
    verbose: bool, optional. Whether to print training information.
    '''

    assert data_valid is not None or iteration is not None
 
    #hyperparameters
    ndim    = data_train.shape[1]
    nclass  = len(np.unique(label_train.cpu().numpy()))
    print(ndim,nclass)

    ndata = len(data_train)

    if M is None:
        M = max(min(200, int(ndata**0.5)), 50)
    if alpha is None:
        alpha = (1-0.02*math.log10(ndata), 1-0.001*math.log10(ndata))#).to(device)
    if bounds is not None:
        assert len(bounds) == ndim
        for i in range(ndim):
            assert len(bounds[i]) == 2
    if edge_bins is None:
        edge_bins = max(int(math.log10(ndata))-1, 0)
    if batchsize is None:
        batchsize = len(data_train)
    if not patch:
        if K is None:
            if ndim <= 8 or ndata / float(ndim) < 20:
                K = ndim
            else:
                K = 8
        if ndata_A is None:
            ndata_A = min(len(data_train), int(math.log10(ndim)*1e5))
        if MSWD_max_iter is None:
            MSWD_max_iter = min(round(ndata) // ndim, 200)
    else:
        assert shape[0] > 4 and shape[1] > 4
        K0 = K
        ndata_A0 = ndata_A
        MSWD_max_iter0 = MSWD_max_iter

        
    best_accuracy = 0
    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not nocuda else "cpu")

    
    ### taking out logit transform and replacing log jacobian with zeros 
    logj_train    = torch.zeros(data_train.shape[0],device=device)
    logj_valid    = torch.zeros(data_valid.shape[0],device=device)

    #define the model
    if model is None:
        model = ConditionalSINF(ndim=ndim,n_class=nclass).requires_grad_(False).to(device)
        if data_valid is not None:
            best_logp_valid = -1e10
            best_Nlayer     = 0
            wait            = 0
            maxwait         = 5 

    
    #whiten
    if Whiten:
        layer = whiten(ndim_data=ndim, scale=True, ndim_latent=ndim).requires_grad_(False).to(device)
        layer.fit(data_train)

        data_train, logj_train0 = layer(data_train)
        logj_train += logj_train0

        logp_train = (torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item()
        
        if data_valid is not None:
            data_valid, logj_valid0 = layer(data_valid)
            logj_valid += logj_valid0
            logp_valid = (torch.mean(logj_valid) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_valid**2,  dim=1)/2)).item()
            if logp_valid > best_logp_valid:
                best_logp_valid = logp_valid
                best_Nlayer = len(model.layer)

        model.add_layer(layer)
        if verbose:
            if data_valid is not None:
                print('After whiten logp:', logp_train, logp_valid)
            else:
                print('After whiten logp:', logp_train)

                
                
    data_train    = data_train[None,:].repeat_interleave(nclass, axis=0)
    data_valid    = data_valid[None,:].repeat_interleave(nclass, axis=0)

    logj_train    = logj_train[None,:].repeat_interleave(nclass, axis=0)
    logj_valid    = logj_valid[None,:].repeat_interleave(nclass, axis=0)

    #GIS iterations
    
    ii=0
    while True:
        t = time.time()

        if NBfirstlayer:
            layer = ConditionalSlicedTransport_discrete(ndim=ndim, n_class = nclass, K=ndim, M=M).requires_grad_(False).to(device)
        else:
            layer = ConditionalSlicedTransport_discrete(ndim=ndim, n_class = nclass, K=K, M=M).requires_grad_(False).to(device)

        #fit the layer
        if NBfirstlayer:
            layer.A[:] = torch.eye(ndim).to(device)
            NBfirstlayer = False
        elif ndim > 1:
            layer.fit_A(data_train[label_train, torch.arange(data_train.shape[1]).to(device)], MSWD_max_iter=MSWD_max_iter, verbose=verbose)

        layer.fit_spline(data_train[label_train, torch.arange(data_train.shape[1]).to(device)], label_train, edge_bins=edge_bins, derivclip=1, alpha=alpha, KDE=KDE, b_factor=b_factor, verbose=False)
        
        for label in range(nclass):
            data_train[label], logj_train1 = layer(data_train[label], torch.ones(data_train.shape[1], dtype=torch.int, device=data_train.device)*label)
            logj_train[label] = logj_train[label] + logj_train1

            data_valid[label], logj_valid1 = layer(data_valid[label], torch.ones(data_valid.shape[1], dtype=torch.int, device=data_valid.device)*label)
            logj_valid[label] = logj_valid[label] + logj_valid1

        model.add_layer(layer)
        
        logp_train = (torch.mean(logj_train[label_train, torch.arange(data_train.shape[1]).to(device)]) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train[label_train, torch.arange(data_train.shape[1]).to(device)]**2,  dim=1)/2)).item()
        logp_valid = (torch.mean(logj_valid[label_valid, torch.arange(data_valid.shape[1]).to(device)]) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_valid[label_valid, torch.arange(data_valid.shape[1]).to(device)]**2,  dim=1)/2)).item()

            
        if logp_valid > best_logp_valid:
            best_logp_valid = logp_valid
            best_Nlayer = len(model.layer)
            wait = 0
        else:
            wait += 1
        if wait == maxwait:
            model.layer = model.layer[:best_Nlayer]
            break
        if ii>max_iter:
            break

        if verbose:
            if data_valid is not None: 
                print ('logp:', logp_train, logp_valid, 'time:', time.time()-t, 'iteration:', len(model.layer), 'best:', best_Nlayer)
            else:
                print ('logp:', logp_train, 'time:', time.time()-t, 'iteration:', len(model.layer))

        if iteration is not None and len(model.layer) >= iteration:
            if data_valid is not None:
                model.layer = model.layer[:best_Nlayer]
            break
        ii+=1

    return model