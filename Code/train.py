import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import getData
import model
from torch.utils.data import DataLoader
from statistics import mean

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((356,356)),
     transforms.RandomCrop((299,299)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

dataset = getData.flickrDataset('../Data/flickr8k/images',
                                '../Data/flickr8k/captions.txt',
                                5,transform)
pad_idx = dataset.vocab.stoi['<PAD>']
loader = DataLoader(dataset,batch_size=32,shuffle=True,collate_fn=getData.myCollate(pad_idx=pad_idx))

vocab = len(dataset.vocab)
hidden_size = 256
embed_size = 256
num_layers = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.E2D(vocab,embed_size,hidden_size,num_layers)
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>'])

#fine tuning the encoder

for name,params in model.encoder.inceptionNet.named_parameters():
    if 'fc.weights' in name or 'fc.bias' in name:
        params.requires_grad = True
    else:
        params.requires_grad = False
        
model.train()

print(len(loader))

for i in range(100):
    overaLoss = []
    for idx,(img,caption) in enumerate(loader):
        img = img.to(device)
        caption = caption.to(device)
        checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
        save_checkpoint(checkpoint)
        output = model(img,caption[:-1])
        loss = criterion(
            output.reshape(-1,output.shape[2]),caption.reshape(-1)
        )
        overaLoss.append(loss)
        optimizer.zero_grad()
        loss.backward(loss)
        optimizer.step()
        print(f'{idx} done')
        
    print(f'the loss for {i} epoch is {mean(loss)}')

    


