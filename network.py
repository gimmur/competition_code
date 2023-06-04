import torchvision
import torch

class ResNet(torch.nn.Module):

    """Definition of a pretrained ResNet to compute features.
    This won't have a final layer sending to classes: the
    final layer will be an identity layer outputting the
    features of the next-to-last layer. Its configuration
    won't have num_classes"""

    def __init__(self, config):
        super(ResNet, self).__init__()
        # Load a resnet18 from torchvision, either use pretrained weights or not
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = torchvision.models.resnet18(weights=weights)
        # remove the last FC layer
        #num_output_feats = self.net.fc.in_features
        self.net.fc = torch.nn.Identity()

    def forward(self, x):
        features = self.net(x)
        return features
