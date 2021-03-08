import torch
from torch.nn.modules.sparse import Embedding
from torchvision import models
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,embed_size):
        super(Encoder,self).__init__()
        self.inceptionNet = models.inception_v3(pretrained=True, aux_logits=False)
        self.inceptionNet.fc = nn.Linear(self.inceptionNet.fc.in_features,embed_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self,image):
        x = self.inceptionNet(image)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Decoder(nn.Module):
    def __init__(self,vocab,embed_size,hidden_size,num_layers):
        super(Decoder,self).__init__()
        self.embed = nn.Embedding(vocab,embed_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers)
    def forward(self,features,caption):
        embeddings = self.dropout(self.embed(caption))
        embeddings = torch.cat((features.unsqueeze(0),embeddings),dim=0)
        output,_ = self.lstm(embeddings)
        output = self.linear(output)
        return output
    
class E2D(nn.Module):
    def __init__(self,vocab,embed_size,hidden_size,num_layers):
        super(E2D,self).__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(vocab,embed_size,hidden_size,num_layers)
        
    def forward(self,img,captions):
        encoderOutput = self.encoder(img)
        decoderOutput = self.decoder(encoderOutput,captions)
        return decoderOutput
        
        