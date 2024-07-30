
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from mlp import BaseClassifier

in_dim, feature_dim, out_dim = 784, 256, 10
loss_fn = nn.CrossEntropyLoss()
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
classifier.load_state_dict(torch.load('mnist.pt'))
test_dataset = MNIST(".", train=False, 
                     download=True, transform=ToTensor())
test_loader = DataLoader(test_dataset, 
                         batch_size=64, shuffle=False)

def test(classifier=classifier, 
          loss_fn = loss_fn):
  classifier.eval()
  accuracy = 0.0
  computed_loss = 0.0

  with torch.no_grad():
      for data, target in test_loader:
          data = data.flatten(start_dim=1)
          out = classifier(data)
          _, preds = out.max(dim=1)

          # Get loss and accuracy
          computed_loss += loss_fn(out, target)
          accuracy += torch.sum(preds==target)
          
      print("Test loss: {}, test accuracy: {}".format(
          computed_loss.item()/(len(test_loader)*64), accuracy*100.0/(len(test_loader)*64)))
      
test()