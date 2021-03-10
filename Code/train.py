from numpy import mod
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms
import getData
import model
from statistics import mean
from torch.utils.data import DataLoader
import cv2
# import inference

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    return

def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(cv2.imread("../Data/flickr8k/test_examples/dog.jpg")).unsqueeze(0)
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    # test_img2 = transform(
    #     cv2.imread("../Data/flickr8k/test_examples/child.jpg")).unsqueeze(0)
    # print("Example 2 CORRECT: Child holding red frisbee outdoors")
    # print(
    #     "Example 2 OUTPUT: "
    #     + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    # )
    # test_img3 = transform(cv2.imread("../Data/flickr8k/test_examples/bus.png")).unsqueeze(0)
    # print("Example 3 CORRECT: Bus driving by parked cars")
    # print(
    #     "Example 3 OUTPUT: "
    #     + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    # )
    # test_img4 = transform(
    #     cv2.imread("../Data/flickr8k/test_examples/boat.png")).unsqueeze(0)
    # print("Example 4 CORRECT: A small boat in the ocean")
    # print(
    #     "Example 4 OUTPUT: "
    #     + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    # )
    # test_img5 = transform(
    #     cv2.imread("../Data/flickr8k/test_examples/horse.png")).unsqueeze(0)
    # print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    # print(
    #     "Example 5 OUTPUT: "
    #     + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    # )
    model.train()

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((356,356)),
     transforms.RandomCrop((299,299)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

loader,dataset = getData.getLoader('../Data/flickr8k/images/','../Data/flickr8k/captions.txt',transform)

torch.backends.cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab = len(dataset.vocab)
hidden_size = 256
embed_size = 256
num_layers = 1
learningRate = 3e-4
numEpochs = 100

model = model.E2D(vocab,embed_size,hidden_size,num_layers)
model = model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>'])
optimizer = optim.Adam(model.parameters(),lr=learningRate)

#fine tuning the encoder
for name,params in model.encoder.inceptionNet.named_parameters():
    if 'fc.weights' in name or 'fc.bias' in name:
        params.requires_grad = True
    else:
        params.requires_grad = False
        
model.train()

for epoch in range(2):
    overaLoss = []
    for idx,(img,caption) in enumerate(loader):
        img = img.to(device)
        caption = caption.to(device)
        output = model(img,caption[:-1])
        loss = criterion(
            output.reshape(-1,output.shape[2]),
            caption.reshape(-1)
            )
        overaLoss.append(loss.item())
        optimizer.zero_grad()
        loss.backward(loss)
        optimizer.step()
        if idx%100 ==0:
            print(f'============================{idx} done==================')
            print_examples(model,device,dataset)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    save_checkpoint(checkpoint)  
    print(f'the loss for {epoch} epoch is {mean(overaLoss)}')

    


