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
        features = self.inceptionNet(image)
        return self.dropout(self.relu(features))

class Decoder(nn.Module):
    def __init__(self,vocab,embed_size,hidden_size,num_layers):
        super(Decoder,self).__init__()
        self.embed = nn.Embedding(vocab,embed_size)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size,vocab)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers)
    def forward(self,features,caption):
        embeddings = self.dropout(self.embed(caption))
        embeddings = torch.cat((features.unsqueeze(0),embeddings),dim=0)
        hiddens,_ = self.lstm(embeddings)
        output = self.linear(hiddens)
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
    
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
        
        