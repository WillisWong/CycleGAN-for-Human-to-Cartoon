import dataset
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch

transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
 ])
train_A = dataset.ImageData(['H:/cycleGAN/dataA/dataA'],'train', transform=transf)
test_A = dataset.ImageData(['H:/cycleGAN/dataA/dataA'], 'test', transform=transf)

train_loader = DataLoader(train_A, batch_size=1, shuffle=True)
test_loader = DataLoader(test_A, batch_size=1, shuffle=False)
# tttt=next(iter(train_loader))
# ltttt=list(tttt.keys())
# tar=tttt[ltttt[0]].numpy()
# tar = torch.Tensor(tar/255.0)
# tar = tar.unsqueeze(0)

fig, axes = plt.subplots(2, 2)
axes = np.reshape(axes, (4, ))
for i in range(4):
    # example = next(iter(train_loader))['0'].numpy().transpose((1, 2, 0))
    example = next(iter(train_loader))['A'].numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    example = std * example + mean
    axes[i].imshow(example)
    axes[i].axis('off')
plt.show()