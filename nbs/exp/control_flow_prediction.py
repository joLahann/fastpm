
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/12_fastpm_control_flow_prediction.ipynb

from exp.eventlog import *
from exp.dl_utils import *
import editdistance as ed

class ListContainer():
    def __init__(self, items): self.items = listify(items)
    def __getitem__(self, idx):
        if isinstance(idx, (int,slice)): return self.items[idx]
        if isinstance(idx[0],bool):
            assert len(idx)==len(self) # bool mask
            return [o for m,o in zip(idx,self.items) if m]
        return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res

class ControlFlowList(ListContainer):

    def __init__(self,items,path='.'):
        super().__init__(items)
        self.path=Path(path)

    @classmethod
    def from_df(cls,df,col='concept:name',trace_id='trace_id'):
        items=[]
        for n,g in df.groupby('trace_id'):
            items.append(list(g[col]))

        return cls(items)
    def new(self, items, cls=None):
        if cls is None: cls=self.__class__
        return cls(items, self.path)

    def get(self,o): return "->".join([str(x) for x in listify(self.items[o])])

    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{[self.get(i) for i in range(2)]}'
        if len(self)>2: res = res[:-1]+ '...]'
        return res

import random


def random_splitter(fn, p_valid): return random.random() < p_valid
def split_by_func(items, f):
    mask = [f(o) for o in items]
    # `None` values will be filtered out
    f = [o for o,m in zip(items,mask) if m==False]
    t = [o for o,m in zip(items,mask) if m==True ]
    return f,t

class SplitData():
    def __init__(self, train, valid, test): self.train,self.valid,self.test = train,valid,test

    @classmethod
    def split_by_func(cls, il, f, Test=True):
        train, valid = map(il.new, split_by_func(il.items, f))
        test=None
        if Test:
            train, test = map(train.new, split_by_func(train.items, f))

        return cls(train,valid,test)

    def __repr__(self): return f'{self.__class__.__name__}\nTrain: {self.train}\nValid: {self.valid}\nTest: {self.test}\n'

UNK, PAD, BOT, EOT = "xxunk xxpad xxbot xxeot ".split()

def add_eot_bot(x): return  x + [EOT]
default_spec_tok = [UNK, PAD, BOT, EOT]
default_pre_rules = []
default_post_rules = [add_eot_bot]

from concurrent.futures import ProcessPoolExecutor

def parallel(func, arr, max_workers=4):
    if max_workers<2: results = list(progress_bar(map(func, enumerate(arr)), total=len(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(progress_bar(ex.map(func, enumerate(arr)), total=len(arr)))
    if any([o is not None for o in results]): return results


class TokenizeProcessor():
    def __init__(self, lang="en", chunksize=2000, pre_rules=None, post_rules=None, max_workers=4):
        self.chunksize,self.max_workers = chunksize,max_workers
        self.pre_rules  = default_pre_rules  if pre_rules  is None else pre_rules
        self.post_rules = default_post_rules if post_rules is None else post_rules

    def proc_chunk(self, args):
        i,chunk = args
        docs = [compose(t, self.pre_rules) for t in chunk]
        docs = [compose(t, self.post_rules) for t in docs]
        return docs

    def __call__(self, items):
        toks = []
        if isinstance(items[0], Path): items = [read_file(i) for i in items]
        chunks = [items[i: i+self.chunksize] for i in (range(0, len(items), self.chunksize))]
        toks = parallel(self.proc_chunk, chunks, max_workers=self.max_workers)
        return sum(toks, [])

    def proc1(self, item): return self.proc_chunk([item])[0]


import collections

class NumericalizeProcessor():
    def __init__(self, vocab=None, max_vocab=60000, min_freq=0,spec_token=None,unk_token=None):
        self.vocab,self.max_vocab,self.min_freq = vocab,max_vocab,min_freq
        self.spec_token = default_spec_tok if spec_token is None else spec_token
        self.unk_token = UNK if unk_token is None else unk_token


    def __call__(self, items):
        #The vocab is defined on the first use.
        if self.vocab is None:
            freq = Counter(p for o in items for p in o)
            self.vocab = [o for o,c in freq.most_common(self.max_vocab) if c >= self.min_freq]
            for o in reversed(default_spec_tok):
                if o in self.vocab: self.vocab.remove(o)
                self.vocab.insert(0, o)
        if getattr(self, 'otoi', None) is None:
            self.otoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.vocab)})
        return [self.proc1(o) for o in items]
    def proc1(self, item):  return [self.otoi.get(o, self.unk_token) for o in item]

    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]
    def deproc1(self, idx): return [self.vocab[i] for i in idx]


def _label_by_func(ds, f, cls=ControlFlowList): return cls([f(o) for o in ds.items], path=ds.path)

#This is a slightly different from what was seen during the lesson,
#   we'll discuss the changes in lesson 11
class LabeledData():
    def process(self, il, proc): return il.new(compose(il.items, proc))

    def __init__(self, x, y, proc_x=None, proc_y=None):
        self.x,self.y = self.process(x, proc_x),self.process(y, proc_y)
        self.proc_x,self.proc_y = proc_x,proc_y

    def __repr__(self): return f'{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n'
    def __getitem__(self,idx): return self.x[idx],self.y[idx]
    def __len__(self): return len(self.x)

    def x_obj(self, idx): return self.obj(self.x, idx, self.proc_x)
    def y_obj(self, idx): return self.obj(self.y, idx, self.proc_y)

    def obj(self, items, idx, procs):
        isint = isinstance(idx, int) or (isinstance(idx,torch.LongTensor) and not idx.ndim)
        item = items[idx]
        for proc in reversed(listify(procs)):
            item = proc.deproc1(item) if isint else proc.deprocess(item)
        return item

    @classmethod
    def label_by_func(cls, il, f, proc_x=None, proc_y=None):
        return cls(il, _label_by_func(il, f), proc_x=proc_x, proc_y=proc_y)

def label_by_func(sd, f, proc_x=None, proc_y=None):
    train = LabeledData.label_by_func(sd.train, f, proc_x=proc_x, proc_y=proc_y)
    valid = LabeledData.label_by_func(sd.valid, f, proc_x=proc_x, proc_y=proc_y)
    test =None if sd.test is None else LabeledData.label_by_func(sd.test, f, proc_x=proc_x, proc_y=proc_y)

    return SplitData(train,valid,test)

class CF_LM_Dataset():
    def __init__(self, data, bs=64, bptt=70, shuffle=False):
        self.data,self.bs,self.bptt,self.shuffle = data,bs,bptt,shuffle
        total_len = sum([len(t) for t in data.x])
        self.n_batch = total_len // bs
        self.batchify()

    def __len__(self): return ((self.n_batch-1) // self.bptt) * self.bs

    def __getitem__(self, idx):
        source = self.batched_data[idx % self.bs]
        seq_idx = (idx // self.bs) * self.bptt
        return source[seq_idx:seq_idx+self.bptt],source[seq_idx+1:seq_idx+self.bptt+1]

    def batchify(self):
        texts = self.data.x
        if self.shuffle: texts = texts[torch.randperm(len(texts))]
        stream = torch.cat([tensor(t) for t in texts])
        self.batched_data = stream[:self.n_batch * self.bs].view(self.bs, self.n_batch)

def get_lm_dls(train_ds, valid_ds, bs, bptt, **kwargs):
    return (DataLoader(CF_LM_Dataset(train_ds, bs, bptt, shuffle=True), batch_size=bs, **kwargs),
            DataLoader(CF_LM_Dataset(valid_ds, bs, bptt, shuffle=False), batch_size=2*bs, **kwargs))

def lm_databunchify(sd, bs, bptt, **kwargs):
    return DataBunch(*get_lm_dls(sd.train, sd.valid, bs, bptt, **kwargs))

def dropout_mask(x, sz, p):
    return x.new(*sz).bernoulli_(1-p).div_(1-p)

class RNNDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p=p

    def forward(self, x):
        if not self.training or self.p == 0.: return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m

import warnings

WEIGHT_HH = 'weight_hh_l0'

class WeightDropout(nn.Module):
    def __init__(self, module, weight_p=[0.], layer_names=[WEIGHT_HH]):
        super().__init__()
        self.module,self.weight_p,self.layer_names = module,weight_p,layer_names
        for layer in self.layer_names:
            #Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            #To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

class EmbeddingDropout(nn.Module):
    "Applies dropout in the embedding layer by zeroing out some elements of the embedding vector."
    def __init__(self, emb, embed_p):
        super().__init__()
        self.emb,self.embed_p = emb,embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None: self.pad_idx = -1

    def forward(self, words, scale=None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0),1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else: masked_embed = self.emb.weight
        if scale: masked_embed.mul_(scale)
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)

def to_detach(h):
    "Detaches `h` from its history."
    return h.detach() if type(h) == torch.Tensor else tuple(to_detach(v) for v in h)

class AWD_LSTM(nn.Module):
    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182."
    initrange=0.1

    def __init__(self, vocab_sz, emb_sz, n_hid, n_layers, pad_token,
                 hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5):
        super().__init__()
        self.bs,self.emb_sz,self.n_hid,self.n_layers = 1,emb_sz,n_hid,n_layers
        self.emb = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.emb_dp = EmbeddingDropout(self.emb, embed_p)
        self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz), 1,
                             batch_first=True) for l in range(n_layers)]
        self.rnns = nn.ModuleList([WeightDropout(rnn, weight_p) for rnn in self.rnns])
        self.emb.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input):
        bs,sl = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        raw_output = self.input_dp(self.emb_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden)
        return raw_outputs, outputs

    def _one_hidden(self, l):
        "Return one hidden state."
        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz
        return next(self.parameters()).new(1, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]

class LinearDecoder(nn.Module):
    def __init__(self, n_out, n_hid, output_p, tie_encoder=None, bias=True):
        super().__init__()
        self.output_dp = RNNDropout(output_p)
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        if bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight
        else: init.kaiming_uniform_(self.decoder.weight)

    def forward(self, input):
        raw_outputs, outputs = input
        output = self.output_dp(outputs[-1]).contiguous()
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded, raw_outputs, outputs

class SequentialRNN(nn.Sequential):
    "A sequential module that passes the reset call to its children."
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

def get_language_model(vocab_sz, emb_sz, n_hid, n_layers, pad_token, output_p=0.4, hidden_p=0.2, input_p=0.6,
                       embed_p=0.1, weight_p=0.5, tie_weights=True, bias=True):
    rnn_enc = AWD_LSTM(vocab_sz, emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token,
                       hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    enc = rnn_enc.emb if tie_weights else None
    return SequentialRNN(rnn_enc, LinearDecoder(vocab_sz, emb_sz, output_p, tie_encoder=enc, bias=bias))

def cross_entropy_flat(input, target):
    bs,sl = target.size()
    return F.cross_entropy(input.view(bs * sl, -1), target.view(bs * sl))

def accuracy_flat(input, target):
    bs,sl = target.size()
    return accuracy(input.view(bs * sl, -1), target.view(bs * sl))

class GradientClipping(Callback):
    def __init__(self, clip=None): self.clip = clip
    def after_backward(self):
        if self.clip:  nn.utils.clip_grad_norm_(self.run.model.parameters(), self.clip)

class RNNTrainer(Callback):
    def __init__(self, α, β): self.α,self.β = α,β

    def after_pred(self):
        #Save the extra outputs for later and only returns the true output.
        self.raw_out,self.out = self.pred[1],self.pred[2]
        self.run.pred = self.pred[0]

    def after_loss(self):
        #AR and TAR
        if self.α != 0.:  self.run.loss += self.α * self.out[-1].float().pow(2).mean()
        if self.β != 0.:
            h = self.raw_out[-1]
            if len(h)>1: self.run.loss += self.β * (h[:,1:] - h[:,:-1]).float().pow(2).mean()

    def begin_epoch(self):
        pass
        #Shuffle the texts at the beginning of the epoch
        #if hasattr(self.dl.dataset, "batchify"): self.dl.dataset.batchify()

class CudaCallback(Callback):
    def begin_fit(self): self.model.cuda()
    def begin_batch(self): self.run.xb,self.run.yb = self.xb.cuda(),self.yb.cuda()

from torch import LongTensor,FloatTensor

def process_data_for_next_step_prediction(test,startIndex=1):
    xs,ys=[],[]
    for trace in progress_bar(test.items):
        for i in range(startIndex,len(listify(trace))-1):
            x,y=[],[]
            xs.append(tensor(trace[:i]))
            ys.append(tensor(trace[i]))
    return xs,tensor(ys)

def func(t):
    return len(pd_data[int(t)][0])

from torch.utils.data import Sampler

class SortSampler(Sampler):
    def __init__(self, data_source, key): self.data_source,self.key = data_source,key
    def __len__(self): return len(self.data_source)
    def __iter__(self):
        return iter(sorted(list(range(len(self.data_source))), key=self.key, reverse=True))

class SortishSampler(Sampler):
    def __init__(self, data_source, key, bs):
        self.data_source,self.key,self.bs = data_source,key,bs

    def __len__(self) -> int: return len(self.data_source)

    def __iter__(self):
        idxs = torch.randperm(len(self.data_source))
        megabatches = [idxs[i:i+self.bs*50] for i in range(0, len(idxs), self.bs*50)]
        sorted_idx = torch.cat([tensor(sorted(s, key=self.key, reverse=True)) for s in megabatches])
        batches = [sorted_idx[i:i+self.bs] for i in range(0, len(sorted_idx), self.bs)]
        max_idx = torch.argmax(tensor([self.key(ck[0]) for ck in batches]))  # find the chunk with the largest key,
        batches[0],batches[max_idx] = batches[max_idx],batches[0]            # then make sure it goes first.
        batch_idxs = torch.randperm(len(batches)-2)
        sorted_idx = torch.cat([batches[i+1] for i in batch_idxs]) if len(batches) > 1 else LongTensor([])
        sorted_idx = torch.cat([batches[0], sorted_idx, batches[-1]])
        return iter(sorted_idx)

def pad_collate(samples, pad_idx=1, pad_first=False):
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx
    for i,s in enumerate(samples):
        if pad_first: res[i, -len(s[0]):] = LongTensor(s[0])
        else:         res[i, :len(s[0]) ] = LongTensor(s[0])
    return res, tensor([s[1] for s in samples])

def predict_next_step(learner, test_dl):
    iter_dl = iter(test_dl)
    learner.model.cuda()
    learner.model.eval()
    acc4batch = []
    for x, y in progress_bar(iter_dl):
        x, y = x.cuda(), y.cuda()
        learner.model.reset()
        pred = learner.model(x)
        acc4batch.append((torch.argmax(learner.model(x)[0].view(*x.shape,-1)[:,-1,:], dim=1)==y).float().mean().cpu())
    return np.mean(acc4batch)

def process_data_for_suffix_prediction(test,startIndex=1):
    xs,ys=[],[]
    for trace in progress_bar(test.items):
        for i in range(startIndex,len(listify(trace))-1):
            x,y=[],[]
            xs.append(tensor(trace[:i]))
            ys.append(tensor(trace[i:]))
    return xs,ys

def pad_collate_sp(samples, pad_idx=1):
    max_len = max([len(s[0]) for s in samples])
    res_x = torch.zeros(len(samples), max_len).long() + pad_idx
    for i,s in enumerate(samples): res_x[i, -len(s[0]):] = LongTensor(s[0])
    max_len = max([len(s[1]) for s in samples])
    res_y = torch.zeros(len(samples), max_len).long() + pad_idx
    for i,s in enumerate(samples): res_y[i, :len(s[1]) ] = LongTensor(s[1])
    return res_x,res_y

from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance

def suffix_measure(x, y, x2, res):
    for k in range(len(y)):
        b = y[k]
        h = b!=1
        b = b[h].clone()
        a = x2[:, x.shape[1]:][k]
        a = a[h].clone()
        a_res = a.cpu().numpy()
        b_res = b.cpu().numpy()

        res.append(1-normalized_damerau_levenshtein_distance(list(a_res), list(b_res)))

def predict_suffix(learner, test_dl):
    iter_dl = iter(test_dl)
    res = []
    learner.model.cuda()
    learner.model.eval()
    # preds4batch = []

    for x, y in progress_bar(iter_dl):
        learner.model.reset()
        x2 = x.clone().cuda()
        y = y.cuda()
        for i in range(y.shape[1]):
            x.shape, y.shape
            pred = torch.argmax(learner.model(x2)[0].view(*x2.shape,-1)[:,-1,:], dim=1)
            x2 = torch.cat([x2, pred[:, None]], dim=1)
        #preds4batch.append(x2[:,x.shape[1]:])
        suffix_measure(x, y, x2, res)
    return np.mean(res)