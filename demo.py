# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:25:05 2017

@author: lcr
"""
import torch
import torch.autograd as autogard
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import utilize_data
import sys

torch.manual_seed(1)

TRAIN_PATH = './squad/train-v1.1.json'
DEV_PATH = './squad/dev-v1.1.json'
train_s,train_q,train_a,dev_s,dev_q,dev_a = utilize_data.load_task(TRAIN_PATH,DEV_PATH)

#train_s = ["Jack is a pig , and Jack is in the bedroom".split(),"Jack is a pig , and Jack love playing football".split()]
#train_q = ["What Jack is".split(),"What Jack like to do".split()]
#train_a = ["Jack is a pig".split(),"Jack love playing football".split()]


word_to_ix = {}
word_to_ix["_PAD"] = 0
ix_to_word = {}
ix_to_word[0] = "_PAD"
for s in train_s:
    for w in s:
        if w not in word_to_ix:
            ix_to_word[len(word_to_ix)] = w
            word_to_ix[w] = len(word_to_ix)
for q in train_q:
    for w in q:
        if w not in word_to_ix:
            ix_to_word[len(word_to_ix)] = w
            word_to_ix[w] = len(word_to_ix)
for a in train_a:
    for w in a:
        if w not in word_to_ix:
            ix_to_word[len(word_to_ix)] = w
            word_to_ix[w] = len(word_to_ix)
for s in dev_s:
    for w in s:
        if w not in word_to_ix:
            ix_to_word[len(word_to_ix)] = w
            word_to_ix[w] = len(word_to_ix)
for q in dev_q:
    for w in q:
        if w not in word_to_ix:
            ix_to_word[len(word_to_ix)] = w
            word_to_ix[w] = len(word_to_ix)
for a_list in dev_a:
    for a in a_list:
        for w in a:
            if w not in word_to_ix:
                ix_to_word[len(word_to_ix)] = w
                word_to_ix[w] = len(word_to_ix)

print('the size of vocab:',len(word_to_ix))
print('the size of ix_to_word',len(ix_to_word))

EMBEDDING_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 32
TRAIN_SIZE = len(train_s)
DEV_SIZE = len(dev_s)
EPOCHS = 200
EVALUATION_INTERVAL = 1
GLOVE_PATH = './glove/glove.6B.100d.txt'
DROPOUT = 0.2
LAYER_NUM = 3

def load_embeddings(vocab,filepath,embedding_size):
    embeddings = np.random.uniform(-0.25,0.25,[len(vocab),embedding_size])
    with open(filepath,"r",encoding="utf8") as f:
        for line in f:
            word = line.strip().split(" ")[0]
            if(word in vocab):
                vec = line.strip().split(" ")[1:]
                vec = np.array(vec)
                embeddings[vocab[word]] = vec
    return embeddings

def prepare_sentence(batch_size_sen,to_ix):
    max_len = max([len(e) for e in batch_size_sen])    
    for i in range(len(batch_size_sen)):
        batch_size_sen[i] += ["_PAD"] * (max_len - len(batch_size_sen[i]))
    idxs = []
    for s in batch_size_sen:
        idxs.append([to_ix[w] for w in s])
    tensor = torch.LongTensor(idxs)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return autogard.Variable(tensor,requires_grad=False)
    
def prepare_mask(batch_size_sen):
    mask_idxs = []
    for s in batch_size_sen:
        sub_idxs = []
        for w in s:
            if(w == "_PAD"):
                sub_idxs.append(0)
            else:
                sub_idxs.append(1)
        mask_idxs.append(sub_idxs)
    tensor = torch.FloatTensor(mask_idxs)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return autogard.Variable(tensor,requires_grad=False)
    
    
def prepare_target(batch_size_s,batch_size_a):
    target_start = []
    target_end = []
    for i in range(len(batch_size_s)):
        flag = True
        k = 0
        while flag and k <= len(batch_size_a[i]):
            for j in range(len(batch_size_s[i])):
                if(batch_size_s[i][j:j+len(batch_size_a[i])-k] == batch_size_a[i][0:len(batch_size_a[i])-k]):
                    flag = False
                    target_start.append(j)
                    target_end.append(j+len(batch_size_a[i]) - 1)
                    break
            k = k + 1
        if flag:
            target_start.append(0)
            target_end.append(len(batch_size_a[i]) - 1)
    tensor_start = torch.LongTensor(target_start)
    tensor_end = torch.LongTensor(target_end)
    if torch.cuda.is_available():
        tensor_start = tensor_start.cuda()
        tensor_end = tensor_end.cuda()
    return autogard.Variable(tensor_start,requires_grad=False),autogard.Variable(tensor_end,requires_grad=False)
        
#    for i in range(len(paragraph)):
#        if(paragraph[i:i+len(answer)]==answer):
#            return (autogard.Variable(torch.LongTensor([i])),
#                    autogard.Variable(torch.LongTensor([i+len(answer)-1])))
                        
    
class ASReader(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 vocab_size,
                 init_embeddings,
                 word_to_ix,
                 ix_to_word,
                 dropout=0.5):
        super(ASReader,self).__init__()
        self.Ws = torch.randn(hidden_dim,hidden_dim)
        self.We = torch.randn(hidden_dim,hidden_dim)
        self.Wq = torch.randn(hidden_dim,1)
        if torch.cuda.is_available():
            self.Ws = self.Ws.cuda()
            self.We = self.We.cuda()
            self.Wq = self.Wq.cuda()
        self.Ws = autogard.Variable(self.Ws, requires_grad=True)
        self.We = autogard.Variable(self.We, requires_grad=True)
        self.Wq = autogard.Variable(self.Wq, requires_grad=True)
        self.hidden_dim = HIDDEN_DIM
        self.word_embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.word_embeddings.weight = nn.Parameter(init_embeddings) 
        self.word_embeddings.weight.requires_grad = False
        self.plstm = nn.LSTM(embedding_dim*2+2,
                             hidden_dim//2,
                             num_layers=self.layer_num,
                             bidirectional=True,
                             dropout=dropout)
        self.qlstm = nn.LSTM(embedding_dim,
                             hidden_dim//2,
                             num_layers=self.layer_num,
                             bidirectional=True,
                             dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word
#        self.phidden = self.init_hidden1()
#        self.qhidden = self.init_hidden2()
        
    def init_hidden1(self,batch_size):
        return (autogard.Variable(torch.randn(2*3, batch_size, self.hidden_dim//2), requires_grad=True),
                autogard.Variable(torch.randn(2*3, batch_size, self.hidden_dim//2), requires_grad=True))
        
    def init_hidden2(self,batch_size):
        return (autogard.Variable(torch.randn(2,batch_size,self.hidden_dim//2), requires_grad=True),
                autogard.Variable(torch.randn(2,batch_size,self.hidden_dim//2), requires_grad=True))
    
    def forward(self,paragraph,query,s_mask,q_mask):
        para_embeddings = self.word_embeddings(paragraph)
        query_embeddings = self.word_embeddings(query)
        para_encoding = self.paragraph_encoding(para_embeddings)
        para_encoding = para_encoding * s_mask.view(len(s_mask),len(s_mask[0]),1).expand_as(para_encoding)
        que_encoding = self.query_encoding(query_embeddings,q_mask)
        start_predict_score = self.start_prediction(para_encoding,que_encoding)
        end_predict_score = self.end_prediction(para_encoding,que_encoding)
#        start_predict_score = start_predict_score * mask
#        end_predict_score = end_predict_score * mask
        return start_predict_score,end_predict_score
                        
    def start_prediction(self,para_encoding,que_encoding):
        result = []
        for i in range(len(que_encoding)):
            single_para_encoding = para_encoding[i]
            single_que_encoding = que_encoding[i]
            single_result = single_para_encoding.mm(self.Ws).mm(single_que_encoding.view(self.hidden_dim,1))
            result.append(single_result)
        result = torch.cat(result,1)
        result = result.transpose(0,1)
        predict_score = F.log_softmax(result)
        return predict_score
        
    def end_prediction(self,para_encoding,que_encoding):
        result = []        
        for i in range(len(que_encoding)):
            single_para_encoding = para_encoding[i]
            single_que_encoding = que_encoding[i]
            single_result = single_para_encoding.mm(self.We).mm(single_que_encoding.view(self.hidden_dim,1))
            result.append(single_result)
        result = torch.cat(result,1)
        result = result.transpose(0,1)
        predict_score = F.log_softmax(result)
        return predict_score
    
#    def dropout_layer(self,inp,training):
#        return F.dropout(inp,self.dropout,training)
        
    def paragraph_encoding(self,embeddings):
        embeddings = self.dropout(embeddings)
        #print(embeddings)
#        self.phidden = self.init_hidden1(len(paragraph))
        lstm_out,self.phidden = self.plstm(embeddings.view(len(embeddings[0]),len(embeddings),EMBEDDING_DIM))
        
        #embeds = self.word_embeddings(paragraph).view(len(paragraph), 1, -1)
        #lstm_out, self.phidden = self.plstm(embeds)

        return lstm_out.transpose(0,1).contiguous()
        
    def query_encoding(self,embeddings,mask):
        embeddings = self.dropout(embeddings)
#        self.qhidden = self.init_hidden2(len(query))
        lstm_out,self.qhidden = self.qlstm(embeddings.view(len(embeddings[0]),len(embeddings),EMBEDDING_DIM))
        lstm_out = torch.transpose(lstm_out,0,1)
        Wq = self.Wq.expand(len(lstm_out),len(self.Wq),len(self.Wq[0]))
        attention = torch.bmm(lstm_out,Wq)
        attention = attention.transpose(1,2)
        mask = mask.view(len(mask),1,len(mask[0]))
        attention = attention * mask
        result_out = torch.bmm(attention,lstm_out)
        return result_out 
        
def eva_prediction(start_score,end_score):
    batch_output = []
    for k in range(len(start_score)):    
        start = 0
        end = 0
        min_value = start_score[k][0] + end_score[k][0]
        for i in range(len(start_score[k])):
            for j in range(i,i+15):
                if(j >= len(end_score[k])): break
                if(start_score[k][i]+end_score[k][j] > min_value):
                    start = i
                    end = j
                    min_value = start_score[k][i]+end_score[k][j]
        batch_output.append([start,end])
    return batch_output
    
def evaluation_result(pre_result,target_result):
    _sum = len(pre_result)
    _em_sum = 0
    overlap = 0
    pre_width = 0
    target_width = 0
    for i in range(len(pre_result)):
        if(pre_result[i][0] == target_result[i][0] and pre_result[i][1] == target_result[i][1]):
            _em_sum += 1
        pre_set = set(list(range(pre_result[i][0],pre_result[i][1]+1)))
        tar_set = set(list(range(target_result[i][0],target_result[i][1]+1)))
        overlap += len(pre_set & tar_set)
        pre_width += len(pre_set)
        target_width += len(tar_set)
    P = overlap/pre_width
    R = overlap/target_width
    F = 0
    if P+R != 0:
        F = (2*P*R)/(P+R)
    em = _em_sum/_sum
    return em,F

def evaluation_devresult(pre_result,target_result):
    _sum = len(pre_result)
    _em_sum = 0
    overlap_sum = 0
    pre_width = 0
    target_width = 0
    for i in range(len(pre_result)):
        flag = True
        pre_set = set(list(range(pre_result[i][0],pre_result[i][1]+1)))
        local_R = 0
        local_overlap = 0
        local_target_width = 0
        for j in range(len(target_result[i])):
            if(flag and pre_result[i][0] == target_result[i][j][0] and pre_result[i][1] == target_result[i][j][1]):
                _em_sum += 1
                flag = False
            tar_set = set(list(range(target_result[i][j][0],target_result[i][j][1]+1)))
            overlap = len(pre_set & tar_set)
            if(overlap/len(tar_set) > local_R):
                local_R = overlap/len(tar_set)
                local_overlap = overlap
                local_target_width = len(tar_set)
            if(local_R == 0):
                local_target_width = len(tar_set)
        overlap_sum += local_overlap
        pre_width += len(pre_set)
        target_width += local_target_width
    P =overlap_sum/pre_width
    R = overlap_sum/target_width
    F = 0    
    if P+R != 0:
        F = (2*P*R)/(P+R)
    em = _em_sum/_sum
    return em,F
            
            
def eva_prepare_target(batch_size_s,batch_size_a):
    target_batch_output = []
    for i in range(len(batch_size_s)):
        flag = True
        k = 0
        while flag and k <= len(batch_size_a[i]):
            for j in range(len(batch_size_s[i])):
                if(batch_size_s[i][j:j+len(batch_size_a[i])-k] == batch_size_a[i][0:len(batch_size_a[i])-k]):
                    flag = False
                    target_batch_output.append([j,j+len(batch_size_a[i]) - 1])
                    break
            k = k + 1
        if flag:
            target_batch_output.append([0,len(batch_size_a[i]) - 1])
    return target_batch_output

def eva_prepare_multarget(batch_size_s,batch_size_a):
    target_batch_output = []
    for i in range(len(batch_size_s)):
        a_list = []
        for a in batch_size_a[i]:
            flag = True
            k = 0
            while flag and k <= len(a):
                for j in range(len(batch_size_s[i])):
                    if(batch_size_s[i][j:j+len(a)-k] == a[0:len(a)-k]):
                        flag = False
                        a_list.append([j,j+len(a) - 1])
                        break
                k = k + 1
            if flag:
                a_list.append([0,len(a) - 1])
        target_batch_output.append(a_list)
    return target_batch_output        
#start_index = [-0.1,-0.5,-0.4,-0.01]
#end_index = [-0.3,-0.4,-0.01,-1]
#start,end = test_prediction(start_index,end_index)
#print(start,end)
    
init_embeddings = load_embeddings(word_to_ix,GLOVE_PATH,EMBEDDING_DIM)        
model = ASReader(EMBEDDING_DIM,HIDDEN_DIM,len(word_to_ix),torch.FloatTensor(init_embeddings),word_to_ix,ix_to_word,LAYER_NUM,DROPOUT)
loss_function = nn.NLLLoss()
#optimizer = optim.SGD(model.parameters(),lr=0.01)
#optimizer = optim.Adam(model.parameters(),lr=0.01)
parameters = [e for e in model.parameters() if e.requires_grad]
optimizer = optim.Adamax(parameters,lr=0.005)

#batches = zip(range(0,TRAIN_SIZE - BATCH_SIZE + 1, BATCH_SIZE),range(BATCH_SIZE, TRAIN_SIZE + 1, BATCH_SIZE))
#batches = [(start,end) for start,end in batches]
batches = [(start,start + BATCH_SIZE)for start in range(0,TRAIN_SIZE,BATCH_SIZE)]
#print(batches)

if torch.cuda.is_available():
    model.cuda(0)

for t in range(1,EPOCHS+1):
    np.random.shuffle(batches)
    loss_sum = 0  
    model.train()
    for start,end in batches:
        if(end > len(train_s)): end = len(train_s)
        model.zero_grad()
        s = train_s[start:end]
        q = train_q[start:end]
        a = train_a[start:end]
        paragraph_in = prepare_sentence(s,word_to_ix)
        query_in = prepare_sentence(q,word_to_ix)
        s_mask_in = prepare_mask(s)
        q_mask_in = prepare_mask(q)
        start_score,end_score = model(paragraph_in,query_in,s_mask_in,q_mask_in)
        start_target,end_target = prepare_target(s,a)
        loss = loss_function(start_score,start_target)+loss_function(end_score,end_target)
        loss_sum += loss.cpu().data.numpy()
        loss.backward()
        optimizer.step()
    
    if t % EVALUATION_INTERVAL == 0:
        model.eval()
        pre_result = []
        target_result = []
        for start in range(0,TRAIN_SIZE,BATCH_SIZE):
            end = start + BATCH_SIZE
            s = train_s[start:end]
            q = train_q[start:end]
            a = train_a[start:end]
            s_in = prepare_sentence(s,word_to_ix)
            q_in = prepare_sentence(q,word_to_ix)
            s_mask_in = prepare_mask(s)
            q_mask_in = prepare_mask(q)
            start_score,end_score = model(s_in,q_in,s_mask_in,q_mask_in)
            pre_batch_result = eva_prediction(start_score.cpu().data.numpy(),end_score.cpu().data.numpy())
            tar_batch_result =eva_prepare_target(s,a)
            pre_result.extend(pre_batch_result)
            target_result.extend(tar_batch_result)
#        print(pre_result)
#        print(target_result)
        em,f1 = evaluation_result(pre_result,target_result)
        print('iteration:',t)
        print('train dataset:')
        print('the exact match value is:',em)
        print('the F1 value is:',f1)
        print('the total loss is:',loss_sum[0]) 
        print('------------------------------')
        pre_result = []
        target_result = []
        for start in range(0,DEV_SIZE,BATCH_SIZE):
            end = start + BATCH_SIZE
            s = dev_s[start:end]
            q = dev_q[start:end]
            a = dev_a[start:end]
            s_in = prepare_sentence(s,word_to_ix)
            q_in = prepare_sentence(q,word_to_ix)
            s_mask_in = prepare_mask(s)
            q_mask_in = prepare_mask(q)
            start_score,end_score = model(s_in,q_in,s_mask_in,q_mask_in)
            pre_batch_result = eva_prediction(start_score.cpu().data.numpy(),end_score.cpu().data.numpy())
            tar_batch_result = eva_prepare_multarget(s,a)
            pre_result.extend(pre_batch_result)
            target_result.extend(tar_batch_result)
#        print(target_result)
        em,f1 = evaluation_devresult(pre_result,target_result)
        print('dev dataset:')
        print('the exact match value is:',em)
        print('the F1 value is:',f1)
        print('------------------------------')