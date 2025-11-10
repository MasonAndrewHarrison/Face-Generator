import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_dim, features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, features, 5, 1, 2),
            nn.ReLU(inplace=True),

            self._block(features, features*2, 3, 2, 1),
            self._block(features*2, features*4, 3, 2, 1),
            nn.Dropout(0.25),

            self._block(features*4, features*8, 3, 2, 1),
            self._block(features*8, features*16, 3, 2, 1),
            nn.Dropout(0.25),

            self._block(features*16, features*32, 3, 2, 1),

            nn.Conv2d(features*32, 1, 2, 1, 0),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.disc(x)



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = torch.randn(1, 1, 64, 64).to(device)
    disc = Discriminator(64, 64).to(device)

    print(image.shape)
    new_image = disc(image)
    print(new_image.shape)


