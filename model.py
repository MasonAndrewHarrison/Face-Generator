import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_dim, features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            
            self._Depthwise_Separable_Conv2d(3, features, 5, 2, 2),
            self._Depthwise_Separable_Conv2d(features, features*2, 3, 2, 1),
            nn.Dropout(0.25),

            self._Depthwise_Separable_Conv2d(features*2, features*4, 3, 2, 1),
            self._Depthwise_Separable_Conv2d(features*4, features*8, 3, 2, 1),
            nn.Dropout(0.25),

            self._Depthwise_Separable_Conv2d(features*8, features*16, 3, 2, 1),
            
            self._Depthwise_Separable_Conv2d(features*16, 1, 2, 1, 0, activation=False),
            nn.Sigmoid()
        )
    
    def _Depthwise_Separable_Conv2d(self, in_channels, out_channels, kernel_size, stride, padding, activation=True):
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not activation,
                groups=in_channels
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=not activation,               
            ),
        ]

        if activation:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.disc(x).view(-1, 1)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim, features):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, z_dim, 3, 1, 0, bias=True),
            nn.ReLU(inplace=True)

        )

    def _Depthwise_Separable_ConvTranspose2d(self, in_channels, out_channels, kernel_size, stride, padding, activation=True):
        layers = [
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not activation,
                groups=in_channels
            ),
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not activation,
                groups=in_channels
            )
        ]

        if activation:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(0.2, inplace=True))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.gen(x)



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    z_noise = torch.randn(1, 100, 1, 1).to(device)
    gen = Generator(100, 64, 1000).to(device)
    disc = Discriminator(64, 64).to(device)
    print(z_noise.shape)
    image = gen(z_noise)

    print(image.shape)


    #new_image = disc(image)
    #print(new_image.shape)


