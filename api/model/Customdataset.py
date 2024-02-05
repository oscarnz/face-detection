import glob
import torch
import PIL 
import os 

#Custom dataset class
class FacemaskDataset(object):
  def __init__(self, path, transform=None):
    paths_masked = glob.glob(os.path.join(path, "maskon/*.jpg"))
    paths_no_masked = glob.glob(os.path.join(path, "maskoff/*.jpg"))

    labels = []
    self.l_imgs_path = paths_masked + paths_no_masked
    self.img_labels = [1] * len(paths_masked) + [0] * len(paths_no_masked)
    self.img_labels = torch.Tensor(self.img_labels)
    self.transform = transform

  def __len__(self):
    return len(self.l_imgs_path)

  def __getitem__(self, idx):
    image = PIL.Image.open(self.l_imgs_path[idx])
    y_label = self.img_labels[idx]
    if self.transform:
      image = self.transform(image)
    return image, y_label