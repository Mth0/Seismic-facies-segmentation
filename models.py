import torch
import torch.nn as nn
from torchvision import models

def init_mobile(IN_CHANNELS=1, NUM_CLASSES=6, device="cuda"):
    """
    Given input and output number, Initialize a pre-trained
    DeepLabV3 with a MobileNetV3-Large backbone based on these specs.
    Args:
        IN_CHANNELS (int): Model's number of channels
        NUM_CLASSES (int): Model's Number of classes
        device (str): The device where the computations are going to be made
    Returns:
        model (nn.Module): The final pre-trained model with a classification layer.
    """
    
    # Loads a pre-trained DeepLabV3 with a MobileNetV3-Large backbone
    model = models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    )
    
    # Freezes the backbone (optional, but good for fine-tuning)
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    original_conv1 = model.backbone['0'][0]
    
    # Create a new (IN_CHANNELS)-channel conv layer
    model.backbone['0'][0] = nn.Conv2d(
        IN_CHANNELS,
        original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=False
    )
    # Copies the weights from the 'red' channel
    model.backbone['0'][0].weight.data = original_conv1.weight.data[:, 0:1, :, :]
    
    # Finds the last conv layer in the classifier head
    model.classifier[4] = nn.Conv2d(
        256,  # Input features to this layer
        NUM_CLASSES,
        kernel_size=1
    )
    
    model.aux_classifier[4] = nn.Conv2d(
        10,  # Input features to this layer
        NUM_CLASSES,
        kernel_size=1
    )
    
    model = model.to(device)
    print("MobileNetV3-Large model built successfully!")

    return model


def load_mobile_net(IN_CHANNELS, NUM_CLASSES, state_dict, device="cuda"):
    """
    Loads a mobile-net-like neuralnet from its state_dict.
    Args:
        IN_CHANNELS (int): Model's number of channels
        NUM_CLASSES (int): Model's Number of classes
        state_dict (dict): The state_dict of the original model containing the weights
        and other model's informations.
        device (str): The device where the computations are going to be made
    Returns:
        model (nn.Module): The final pre-trained model with a classification layer.
    """
    # Loads the default pre-trained model
    model = models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    )
    
    original_conv1 = model.backbone['0'][0]
    new_conv1 = nn.Conv2d(
        IN_CHANNELS,
        original_conv1.out_channels, # This is 16
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=False
    )
    # Replaces the old 3-channel layer with your new 1-channel layer
    model.backbone['0'][0] = new_conv1
    
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(10, NUM_CLASSES, kernel_size=1)
    
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print("Model successfully built and weights loaded!")

    return model




def init_resnet50(IN_CHANNELS=1, NUM_CLASSES=6, device="cuda"):
    """
    Given input and output number, Initialize a pre-trained
    FCN with a ResNet50 backbone based on these specs.
    Args:
        IN_CHANNELS (int): Model's number of channels
        NUM_CLASSES (int): Model's Number of classes
        device (str): The device where the computations are going to be made
    Returns:
        model (nn.Module): The final pre-trained model with a classification layer.
    """
    
    # Loads a pre-trained FCN with a ResNet50 backbone
    model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)
    
    # Freezes the backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    original_conv1 = model.backbone.conv1
    original_weights = original_conv1.weight
    
    model.backbone.conv1 = nn.Conv2d(
        IN_CHANNELS, 
        original_conv1.out_channels, 
        kernel_size=original_conv1.kernel_size, 
        stride=original_conv1.stride, 
        padding=original_conv1.padding, 
        bias=False
    )
    # Copies weights from the first (red) channel
    model.backbone.conv1.weight.data = original_weights.data[:, 0:1, :, :]
    
    model.classifier[4] = nn.Conv2d(
        512,  # Number of input features to this layer
        NUM_CLASSES, 
        kernel_size=1
    )
    
    model = model.to(device)
    
    print("Model built successfully!")
    
    return model