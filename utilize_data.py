# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:16:47 2017

@author: lcr
"""

import json
import re
import nltk

def load_task(train_filepath,dev_filepath):
    print('loading dataset...')
    train_data = load_traindata(train_filepath)
    trainS,trainQ,trainA = vectorize_data(train_data)
    dev_data = load_devdata(dev_filepath)
    devS,devQ,devA = vectorize_devdata(dev_data)
    print('loading over...')
    return trainS,trainQ,trainA,devS,devQ,devA

def load_traindata(filepath):
    data = []
    with open(filepath,"r") as f:  
        json_str = json.load(f)
        
        title_list = json_str["data"]
        for title in title_list:
            title = json.dumps(title)
            title = json.loads(title)
            paragraph_list = title["paragraphs"]
            for paragraph in paragraph_list:
                paragraph = json.dumps(paragraph)
                passage = json.loads(paragraph)
                context = passage["context"]
                qa_list = passage["qas"]
                for qa in qa_list:
                    qa = json.dumps(qa)
                    qa = json.loads(qa)
                    question = qa["question"]
                    a_list = qa["answers"]
                    for a in a_list:
                        a = json.dumps(a)
                        a = json.loads(a)
                        answer = a["text"]
                        data.append([context,question,answer])
    return data[0:10]
    
def load_devdata(filepath):
    data = []
    with open(filepath,"r") as f:  
        json_str = json.load(f)
        
        title_list = json_str["data"]
        for title in title_list:
            title = json.dumps(title)
            title = json.loads(title)
            paragraph_list = title["paragraphs"]
            for paragraph in paragraph_list:
                paragraph = json.dumps(paragraph)
                passage = json.loads(paragraph)
                context = passage["context"]
                qa_list = passage["qas"]
                for qa in qa_list:
                    qa = json.dumps(qa)
                    qa = json.loads(qa)
                    question = qa["question"]
                    a_list = qa["answers"]
                    answer_list = []
                    for a in a_list:
                        a = json.dumps(a)
                        a = json.loads(a)
                        answer = a["text"]
                        answer_list.append(answer)
                    data.append([context,question,answer_list])
    
    return data[0:5]
                        
#def tokenize(sent):
#    '''Return the tokens of a sentence including punctuation.
#    >>> tokenize('Bob dropped the apple. Where is the apple?')
#    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
#    '''
#    sym_set = set([':','.','``','`','\'\'','(',')',',','!','?',';','"','\''])
#    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip() and x.strip() not in sym_set]
                        
def tokenize(sent):
    sens = nltk.sent_tokenize(sent)
    words = []
    for sen in sens:
        words.extend(nltk.word_tokenize(sen))
    return words
    
def vectorize_devdata(data):
    S = []
    Q = []
    A = []
    for s,q,a_list in data:
        s = tokenize(s)
        q = tokenize(q)
        a = []
        for answer in a_list:
            answer = tokenize(answer)
            if(len(answer) != 0):
                a.append(answer)
        if(len(a) != 0):
            S.append(s)
            Q.append(q)
            A.append(a)
    return S,Q,A    

def vectorize_data(data):
    S = []
    Q = []
    A = []
    for s,q,a in data:
        s = tokenize(s)
        q = tokenize(q)
        a = tokenize(a)
        if(len(a) != 0):
            S.append(s)
            Q.append(q)
            A.append(a)
    return S,Q,A

        