import torch
import torch.optim as optim
import torchvision.transforms as transforms
import model
import getData
from torch.utils.data import DataLoader

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

checkpoint = r'my_checkpoint.pth.tar'
model.load_state_dict(torch.load(checkpoint)['state_dict'])
optimizer.load_state_dict(torch.load(checkpoint)['optimizer'])
