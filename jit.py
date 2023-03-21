import torch
from PIL import Image
from torchvision import transforms

model = torch.load('myModel5.pth')
img = Image.open("2.jpg")
convert_tensor = transforms.ToTensor()
image = convert_tensor(img)
image = torch.unsqueeze(image, dim = 0)
traced_model = torch.jit.trace(model, image)
traced_model.save('traced_model.pt')