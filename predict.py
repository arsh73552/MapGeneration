import torch
from Unet import UNet
from torch import nn
from discriminator import Discriminator
from torchvision import transforms
from PIL import Image
from helper import show_tensor_images
from helper import *
input_dim = 3
real_dim = 3
lr = 0.0003
target_shape = 256
device = 'cpu'

img = Image.open("2.jpg")
convert_tensor = transforms.ToTensor()
image = convert_tensor(img)
image = torch.unsqueeze(image, dim = 0)
gen = UNet(input_dim, real_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(input_dim + real_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

loaded_state = torch.load("myModel5.pth")
gen.load_state_dict(loaded_state["gen"])
gen_opt.load_state_dict(loaded_state["gen_opt"])
disc.load_state_dict(loaded_state["disc"])
disc_opt.load_state_dict(loaded_state["disc_opt"])

image_width = image.shape[3]
condition = image[:, :, :, :image_width // 2]
real = image[:, :, :, image_width // 2:]
condition = nn.functional.interpolate(condition, size=target_shape)
real = nn.functional.interpolate(real, size=target_shape)
condition = condition.to(device)
real = real.to(device)
with torch.no_grad():
    fake = gen(condition)
a = show_tensor_images(fake, size=(real_dim, target_shape, target_shape))
b = show_tensor_images(real, size=(real_dim, target_shape, target_shape))
c = show_tensor_images(condition, size=(real_dim, target_shape, target_shape))
b = torch.cat([a, c, b], axis = 1)
plt.imshow(b)
plt.show()