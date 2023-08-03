from model.classifier import ImageClassifier
from torchvision import models 

model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')