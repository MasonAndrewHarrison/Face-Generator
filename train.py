import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
from model import Critic, Generator, initialize_weights
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 100
image_dim = 64
batch_size = 256
num_epochs = 5
disc_features = 32
gen_features = 32
critic_iterations = 5
weight_clip = 0.01

critic = Critic(disc_features).to(device)
generator = Generator(z_dim, gen_features).to(device)


if os.path.exists("Generator.pth"):
    generator.load_state_dict(torch.load("Generator.pth", map_location=device))
    generator.to(device)

else:
    initialize_weights(generator)

if os.path.exists("Critic.pth"):
    critic.load_state_dict(torch.load("Critic.pth", map_location=device))
    critic.to(device)

else:
    initialize_weights(critic)

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
            critic_fake = critic(fake_image.detach()).reshape(-1) 

            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-weight_clip, weight_clip)

        ### Train Generator: min -E[critic(gen_fake)] ###
        z_noise = torch.randn(current_batch_size, z_dim, 1, 1).to(device)
        fake_image = generator(z_noise)    
        output = critic(fake_image).reshape(-1)
        loss_gen = -torch.mean(output)
        generator.zero_grad()
        loss_gen.backward() 
        opt_gen.step()
        
        if i == 25:

            print("saved model for epoch :", epoch+1)
            torch.save(generator.state_dict(), "Generator.pth")
            torch.save(critic.state_dict(), "Critic.pth")

        if i % 1 == 0:
            print(f" Generator Loss: {loss_gen.item()}, Critic Loss: {loss_critic.item()}")

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
       
