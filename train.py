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
disc_features = 128
gen_features = 128
critic_iterations = 5
weight_clip = 0.01

critic = Discriminator(disc_features).to(device)
generator = Generator(z_dim, gen_features).to(device)


if os.path.exists("Generator_Weights.pth"):
    generator.load_state_dict(torch.load("Generator_Weights.pth", map_location=device))
    generator.to(device)

else:
    #initialize_weights(generator)
    pass

if os.path.exists("Discriminator_Weights.pth"):
    critic.load_state_dict(torch.load("Discriminator_Weights.pth", map_location=device))
    critic.to(device)

else:
    #initialize_weights(critic)
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

opt_critic = optim.RMSprop(critic.parameters(), lr=5e-5)
opt_gen = optim.RMSprop(generator.parameters(), lr=5e-5)

fixed_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)

critic.train()
generator.train()
sample_iter = 0

os.makedirs('samples', exist_ok=True)

for epoch in range(num_epochs):
    for i, (real, _) in enumerate(loader):
        
        real_image = real.to(device)
        current_batch_size,_,_,_ = real_image.shape

        for _ in range(critic_iterations):
            z_noise = torch.randn(current_batch_size, z_dim, 1, 1).to(device)
            fake_image = generator(z_noise)      

            critic_real = critic(real_image).reshape(-1)
            critic_fake = critic(fake_image).reshape(-1) 
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-weight_clip, weight_clip)

        ### Train Generator: min -E[critic(gen_fake)] ###
        output = critic(fake_image).reshape(-1)
        loss_gen = -torch.mean(output)
        generator.zero_grad()
        loss_gen.backward() 
        opt_gen.step()
        
        '''#Back Prob for Discriminator
        fake_prediction = critic(fake_image.detach()).view(-1)
        real_prediction = critic(real_image).view(-1)

        disc_loss_fake = criterion(fake_prediction, torch.zeros_like(fake_prediction))
        disc_loss_real = criterion(real_prediction, torch.ones_like(real_prediction))

        disc_loss = (disc_loss_fake + disc_loss_real) / 2
        critic.zero_grad()
        disc_loss.backward()
        opt_disc.step()


        #Back Prob for Generator Model  
        gen_prediction = critic(fake_image).view(-1)
        gen_loss = criterion(gen_prediction, torch.ones_like(gen_prediction))
        generator.zero_grad()
        gen_loss.backward()
        opt_gen.step()'''

        if i % 25 == 0:

            print("saved model for epoch :", epoch+1)
            torch.save(generator.state_dict(), "Generator.pth")
            torch.save(critic.state_dict(), "Discriminator.pth")

        if i % 1 == 0:
            print(loss_gen.item(), loss_critic.item())

        if i % 10 == 0:
            generator.eval()
            with torch.no_grad():
                sample_iter += 1
                print(f"saved sample {sample_iter}")
                fake = generator(fixed_noise).detach().cpu()
                fake = fake[0, :, :, :].permute(1, 2, 0) * 0.5 + 0.5
                plt.imsave(f"samples/fake_images_epoch_{sample_iter}.png", fake.numpy())
                #plt.imshow(fake)
                #plt.show()
            generator.train()
       
