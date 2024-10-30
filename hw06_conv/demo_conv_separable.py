import torch
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Separable translation kernel
H_row = torch.zeros(1,200).float()
H_col = torch.zeros(200,1).float()

# In your HW submission, print out your code here
# We want to move Su7 to 200 pixels up and left
# TODO: assign H_row, H_col values
# TODO: repeat channels or resize both H_row and H_col to make the following code work,
#  since you want translate pixel on each of the RGB channels


###############################################
## Warning: Do not modify any lines below  ###
###############################################
# In your HW assignment, do not need to print any lines below
# Since you should not modify anything
# Read the image
image = cv2.imread('su7_ultra.jpg')
# Convert BGR image to RGB image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Define a transform to convert the image to torch tensor
transform = transforms.Compose([transforms.ToTensor()])
# Convert the image to Torch tensor, and make it [B=1, C=3, H, W]
I = transform(image)
I = I.unsqueeze(0)
print(I.size())
# check what is groups parameter in F.conv2d
# https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
# apply transform on horizontal axis
out1 = F.conv2d(I, H_row, groups=3, stride=[1,1], padding=[0,100])
print(out1.size())
# apply transform on vertical axis
out2 = F.conv2d(out1, H_col, groups=3,  stride=[1,1], padding=[100,0])
print(out2.size())
# save to image, you got a transformed XiaoMi Su7 Ultra, print your result in hw submission
save_image(out2, 'su7_ultra_move.png')