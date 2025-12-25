import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
from model import Discriminator, Generator
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 100
image_dim = 64
batch_size = 256
num_epochs = 5
disc_features = 200
gen_features = 200

disc_model = Discriminator(disc_features).to(device)
gen_model = Generator(z_dim, gen_features).to(device)


if os.path.exists("Generator_Weights.pth"):
    gen_model.load_state_dict(torch.load("Generator_Weights.pth", map_location=device))
    gen_model.to(device)

else:
    #initialize_weights(gen_model)
    pass

if os.path.exists("Discriminator_Weights.pth"):
    disc_model.load_state_dict(torch.load("Discriminator_Weights.pth", map_location=device))
    disc_model.to(device)

else:
    #initialize_weights(disc_model)
    pass

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


dataset = ImageFolder(root='dataset/', transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

image, label = dataset[3543]
image = image.permute(1, 2, 0)
image = image * 0.5 + 0.5

#print(image.shape)
#plt.imshow(image)
#plt.show()

opt_disc = optim.Adam(disc_model.parameters(), lr=3e-5, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen_model.parameters(), lr=3e-4, betas=(0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)

disc_model.train()
gen_model.train()
sample_iter = 0

os.makedirs('samples', exist_ok=True)

for epoch in range(num_epochs):
    for i, (real, _) in enumerate(loader):
        
        real_image = real.to(device)
        current_batch_size,_,_,_ = real_image.shape

        z_noise = torch.randn(current_batch_size, z_dim, 1, 1).to(device)
        fake_image = gen_model(z_noise)
        
        #Back Prob for Discriminator
        fake_prediction = disc_model(fake_image.detach()).view(-1)
        real_prediction = disc_model(real_image).view(-1)

        disc_loss_fake = criterion(fake_prediction, torch.zeros_like(fake_prediction))
        disc_loss_real = criterion(real_prediction, torch.ones_like(real_prediction))

        disc_loss = (disc_loss_fake + disc_loss_real) / 2
        disc_model.zero_grad()
        disc_loss.backward()
        opt_disc.step()


        #Back Prob for Generator Model  
        gen_prediction = disc_model(fake_image).view(-1)
        gen_loss = criterion(gen_prediction, torch.ones_like(gen_prediction))
        gen_model.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        if i % 25 == 0:

            print("saved model for epoch :", epoch+1)
            torch.save(gen_model.state_dict(), "Generator.pth")
            torch.save(disc_model.state_dict(), "Discriminator.pth")

        if i % 1 == 0:
            print(gen_loss.item(), disc_loss.item())

        if i % 10 == 0:
            gen_model.eval()
            with torch.no_grad():
                sample_iter += 1
                print(f"saved sample {sample_iter}")
                fake = gen_model(fixed_noise).detach().cpu()
                fake = fake[0, :, :, :].permute(1, 2, 0) * 0.5 + 0.5
                plt.imsave(f"samples/fake_images_epoch_{sample_iter}.png", fake.numpy())
                #plt.imshow(fake)
                #plt.show()
            gen_model.train()
       
