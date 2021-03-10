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
    
    def buildVocab(self,sentenceList):
        frequencies = {}
        idx=4
        for sentence in sentenceList:
            for word in nlp(sentence.lower()):
                if word not in frequencies:
                    print('if it is not in',frequencies)
                    frequencies[word]=1
                else:
                    print('if it is in',frequencies)
                    frequencies[word]+=1
                if frequencies[word]==self.freqThreshold:
                    self.stoi[idx]=word
                    self.itos[word]=idx
                    print(self.itos)
                    idx+=1
    def str2numeric(self,text):
        words = nlp(text)
        return [self.stoi[w] if w in self.stoi else self.stoi['<UNK>']
                for w in words]
    def __len__(self):
        return len(self.itos)

class flickrDataset(Dataset):
    def __init__(self,imgFolder,captionFile,freqThreshold,transform):
        self.imgFolder = imgFolder
        self.dataframe = self.fileReader(captionFile)
        self.captionFile = self.dataframe['caption']
        self.img = self.dataframe['image']
        self.freqThreshold = freqThreshold
        self.transform = transform
        # print(self.captionFile)
        self.vocab = Vocabulary(self.freqThreshold)
        self.vocab.buildVocab(self.captionFile.tolist())
        
    def __getitem__(self, index):
        caption = self.captionFile[index]
        imageID = self.img[index]
        image = cv2.imread(os.path.join(self.imgFolder,imageID))
        if self.transform is not None:
            image = self.transform(image)
        cap2num = [self.vocab.stoi['<SOS>']]
        cap2num+=self.vocab.str2numeric(caption)
        cap2num.append(self.vocab.stoi['<EOS>'])
        return image,torch.tensor(cap2num)
    
    def fileReader(self,txtFile):
        df = pd.read_csv(txtFile)
        return df
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
    
    
if __name__ =='__main__':
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    print('loading the data')
    dataset = flickrDataset('../Data/flickr8k/images',
                            '../Data/flickr8k/captions.txt',
                            5,transform)
    print('going into the dataloader')
    pad_idx = dataset.vocab.stoi['<PAD>']
    loader = DataLoader(dataset,batch_size=32,shuffle=True,collate_fn=myCollate(pad_idx=pad_idx))
    print('completed the dataloader')
    for idx,(img,captions) in enumerate(loader):
        print(img.shape)
        print(captions.shape)