from octave_unet.unet.model import OctaveUnet
import torch

unet = OctaveUnet()
unet = unet.cuda()
image = torch.Tensor(1, 3, 1024, 1024)
image = image.cuda()
output = unet(image)

print(output.size())
pass