import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import spacy 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import cv2

nlp = spacy.load("en_core_web_sm")

class Vocabulary():
    def __init__(self,freqThreshold):
        self.freqThreshold = freqThreshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    
    @staticmethod
    def tokenize(text):
        return [tok.text.lower() for tok in nlp.tokenizer(text)]
    
    def buildVocab(self,sentenceList):
        frequencies = {}
        idx=4
        for sentence in sentenceList:
            for word in self.tokenize(sentence):
                if word not in frequencies:
                    frequencies[word]=1
                else:
                    frequencies[word]+=1
                if frequencies[word]==self.freqThreshold:
                    self.itos[idx]=word
                    self.stoi[word]=idx
                    idx+=1
    def str2numeric(self,text):
        words = self.tokenize(text)
        return [self.stoi[w] if w in self.stoi else self.stoi['<UNK>']
                for w in words]
    def __len__(self):
        return len(self.itos)

class flickrDataset(Dataset):
    def __init__(self,imgFolder,captionFile,freqThreshold,transform):
        self.imgFolder = imgFolder
        self.dataframe = pd.read_csv(captionFile)
        self.caption = self.dataframe['caption']
        self.img = self.dataframe['image']
        self.freqThreshold = freqThreshold
        self.transform = transform
        self.vocab = Vocabulary(self.freqThreshold)
        self.vocab.buildVocab(self.caption.tolist())
        
    def __getitem__(self, index):
        caption = self.caption[index]
        imageID = self.img[index]
        image = cv2.imread(os.path.join(self.imgFolder,imageID))
        if self.transform is not None:
            image = self.transform(image)
        cap2num = [self.vocab.stoi['<SOS>']]
        cap2num+=self.vocab.str2numeric(caption)
        cap2num.append(self.vocab.stoi['<EOS>'])
        return image,torch.tensor(cap2num)

    def __len__(self):
        return len(self.dataframe)
    
class myCollate():
    def __init__(self,pad_idx):
        self.pad_idx = pad_idx
    def __call__(self,batch):
        img = [item[0].unsqueeze(0) for item in batch]
        img = torch.cat(img,dim=0)
        target = [item[1] for item in batch]
        target = pad_sequence(target,padding_value=self.pad_idx)
        return img,target

def getLoader(rootFolder,captionFile,transform,batchSize=32,shuffle=True,pin_memory=True):
    dataset = flickrDataset(rootFolder,captionFile,5,transform)
    pad_idx = dataset.vocab.stoi['<PAD>']
    loader = DataLoader(dataset,batchSize,shuffle,collate_fn=myCollate(pad_idx=pad_idx))
    
    return loader,dataset
    
if __name__ =='__main__':
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    rootFolder = r'../Data/flickr8k/images/'
    captionFile = r'../Data/flickr8k/captions.txt'
    
    loader,dataset = getLoader(rootFolder,captionFile,transform,32)
    
    for img,caption in loader:
        print(img.shape)
        print(caption.shape)