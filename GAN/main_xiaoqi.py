import os 
import numpy as np
import torch 
import torchvision 
from torch import nn 
from torchvision import transforms 
from torchvision.utils import save_image 

np.random.seed(7)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print("Using {} device".format(device))

# Hyper parameters 
latent_size = 64 
hidden_size = 256 
image_size = 784 
num_epochs = 400
batch_size = 100 
sample_dir = 'samples' 

# Create a directory if not exists 
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# image processing 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# MNIST dataset 
mnist = torchvision.datasets.MNIST(
    root="./data",
    train = True,
    transform=transform, 
    download=True
)



# Data loader 
data_loader = torch.utils.data.DataLoader(
    dataset=mnist, 
    batch_size=batch_size,
    shuffle=True
)

# Discriminator 
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
)

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
)

# Device setting 
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer 
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def denorm(x):
    out = (x+1)/2 
    return out.clamp(0, 1) 

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

total_step = len(data_loader)
for epoch in range(num_epochs): 
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)

        # create the labels which are later used as input for the BCE loss 
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ====================================== #
        # train the discriminator #
        # ====================================== # 

        # compute BCE loss using real images where BCE_loss
        # second term of the loss is always zero since real labels == 1 
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs 

        # compute bceloss using fake images 
        # first term of the loss is always zero since fake labels == 0 
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images) 
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # backdrop and optimize 
        d_loss = d_loss_real + d_loss_fake 
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================ #
        #  Train the generator #
        # ================================ # 

        # compute loss with fake images 
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images) 

        # we train G to maximize log(D(G(z))) instead of minimizing log(1-D(G(z)))
        g_loss = criterion(outputs, real_labels)

        # backprop and optimize 
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1)% 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()))
    
    # save real images 
    if (epoch+1) == num_epochs:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    # save sampled images 
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

# save the model checkpoints
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')