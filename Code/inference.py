import torch
import torch.optim as optim
import torchvision.transforms as transforms
import model
import getData
import cv2

def print_examples(model, device, dataset,transform):
    model.eval()
    test_img1 = transform(cv2.imread("../Data/flickr8k/test_examples/dog.jpg"))
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    test_img2 = transform(
        cv2.imread("../Data/flickr8k/test_examples/child.jpg"))
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )
    test_img3 = transform(cv2.imread("../Data/flickr8k/test_examples/bus.png"))
    print("Example 3 CORRECT: Bus driving by parked cars")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )
    test_img4 = transform(
        cv2.imread("../Data/flickr8k/test_examples/boat.png"))
    print("Example 4 CORRECT: A small boat in the ocean")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    )
    test_img5 = transform(
        cv2.imread("../Data/flickr8k/test_examples/horse.png"))
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    )
    model.train()

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

print_examples(model,device,dataset,transform)
