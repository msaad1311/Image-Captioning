from numpy import mod
from spacy import load
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import model
import getData
import cv2

def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    model.to(device)
    model.eval()
    test_img1 = transform(cv2.imread("../Data/flickr8k/test_examples/dog.jpg")).unsqueeze(0)
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    print('='*50)
    test_img2 = transform(
        cv2.imread("../Data/flickr8k/test_examples/child.jpg")).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )
    print('='*50)
    test_img3 = transform(cv2.imread("../Data/flickr8k/test_examples/bus.png")).unsqueeze(0)
    print("Example 3 CORRECT: Bus driving by parked cars")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )
    print('='*50)
    test_img4 = transform(
        cv2.imread("../Data/flickr8k/test_examples/boat.png")).unsqueeze(0)
    print("Example 4 CORRECT: A small boat in the ocean")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    )
    print('='*50)
    test_img5 = transform(
        cv2.imread("../Data/flickr8k/test_examples/horse.png")).unsqueeze(0)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    )
    print('='*50)
    model.train()

if __name__ =='__main__':
    checkpoint = torch.load('my_checkpoint.pth.tar')
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
    
    models = model.E2D(vocab,embed_size,hidden_size,num_layers)
    models.load_state_dict(checkpoint['state_dict'])
    
    print_examples(models,device,dataset)
    