import torch
import torch.nn as nn

# Implementing the Critic (same as DC-GAN Discriminator with small changes)...
class Critic(nn.Module):
    def __init__(self, img_channels, disc_feat):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            # input: N x img_channels x 64 x 64
            nn.Conv2d(img_channels, disc_feat, kernel_size=4, stride=2, padding=1, bias=False,),
            nn.LeakyReLU(0.2),
            self.create_block(disc_feat, disc_feat * 2, 4, 2, 1),
            self.create_block(disc_feat * 2, disc_feat * 4, 4, 2, 1),
            self.create_block(disc_feat * 4, disc_feat * 8, 4, 2, 1),
            nn.Conv2d(disc_feat * 8, 1, kernel_size=4, stride=1, padding=0),
            # Note there is no Sigmoid layer --> Output only raw score --> Hence the name Critic
        )

    def create_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False,),
            nn.InstanceNorm2d(out_channels, affine=True), # No batchnorm in the critic!
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.critic(x)

# Implementing the Generator (same as DC-GAN Generator)...
class Generator(nn.Module):
    def __init__(self, channels_noise, img_channels, gen_feat):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self.create_block(channels_noise, gen_feat * 8, 4, 1, 0),  
            self.create_block(gen_feat * 8, gen_feat * 4, 4, 2, 1),  
            self.create_block(gen_feat * 4, gen_feat * 2, 4, 2, 1),  
            self.create_block(gen_feat * 2, gen_feat, 4, 2, 1), 
            nn.ConvTranspose2d(gen_feat, img_channels, kernel_size=4, stride=2, padding=1, bias=False,),
            # Output: N x img_channels x 64 x 64
            nn.Tanh(),
        )

    def create_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(m):
    # Initializes weights according to the recommendations in DCGAN paper
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: # for conv layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def sanity_check():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Critic(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Check Critic implementation"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Check Generator implementation"
    print("Bravo!")


if __name__ == "__main__":
    sanity_check()