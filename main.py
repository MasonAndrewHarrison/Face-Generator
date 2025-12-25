import torch
from model import Generator
import os
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 100
gen_features = 200

gen_model = Generator(z_dim, gen_features).to(device)

if os.path.exists("Generator_Weights.pth"):
    gen_model.load_state_dict(torch.load("Generator_Weights.pth", map_location=device))
    gen_model.to(device)

z_noise = torch.randn(1, z_dim, 1, 1).to(device)



gen_model.eval()
with torch.no_grad():
    generated_image = gen_model(z_noise).detach().cpu()
    image = generated_image[0, :, :, :].permute(1, 2, 0) * 0.5 + 0.5
    print(image)
    plt.imshow(image)
    plt.show()

