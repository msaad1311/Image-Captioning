import pandas as pd
import numpy as np 
import os
import torch
import torchvision.transforms as transforms
import spacy 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import cv2

nlp = spacy.load("en_core_web_lg")

class Vocabulary():
    def __init__(self,freqThreshold):
        self.freqThreshold = freqThreshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    
    def buildVocab(self,sentenceList):
        frequencies = {}
        idx=4
        for sentence in sentenceList:
            for word in nlp(sentence):
                if word not in frequencies:
                    frequencies[word]=1
                else:
                    frequencies[word]+=1
                if frequencies[word]==self.freqThreshold:
                    self.stoi[idx]=word
                    self.itos[word]=idx
                    idx+=1
    def str2numeric(self,text):
        words = nlp(text)
        return [self.stoi[w] if w in self.stoi else self.stoi['<UNK>']
                for w in words]

class flickrDataset(Dataset):
    def __init__(self,imgFolder,captionFile,freqThreshold,transform):
        self.imgFolder = imgFolder
        self.captionFile = self.fileReader(captionFile)['caption']
        self.img = self.fileReader(captionFile)['image']
        self.freqThreshold = freqThreshold
        self.transform = transform
        
        self.vocab = Vocabulary(self.freqThreshold)
        self.vocab.buildVocab(captionFile.toList())
        
    def __getitem__(self, index):
        caption = self.captionFile[index]
        imageID = self.img[index]
        image = cv2.imread(os.path.join(self.imgFolder,imageID))
        cap2num = [self.vocab.stoi['<SOS>']]
        cap2num+=self.vocab.str2numeric(caption)
        cap2num.append(self.vocab.stoi['<EOS>'])
        
        return image,torch.tensor(cap2num)
    
    def fileReader(self,txtFile):
        df = pd.read_csv(txtFile)
        return df
    
def myCollate():
    def __init__(self,pad_idx):
        self.pad_index = pad_index
    def __call__(self,batch):
        img1 = [item[0] for item in batch]
        print(img1.shape)
        img = [item[0].unsqueeze(0) for item in batch]
        target = [item[1] for item in batch]
        target = pad_sequence(target,padding_value=self.pad_idx)
        return img,target
    
    
