import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def conv_1_3(in_dim, mid_dim, out_dim):
    #1x1 convolution followed by 3x3 convolution
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(),
        nn.Conv2d(mid_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

def max_3_1(in_dim, out_dim):
    #3x3 max pooling followed by 1x1 convolution
    model = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_dim),
    )
    return model

def conv_1(in_dim, out_dim):
    #1x1 convolution
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_dim),
    )
    return model

def conv_1_5(in_dim, mid_dim, out_dim):
    #1x1 convolution followed by 5x5 convolution
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(),
        nn.Conv2d(mid_dim, out_dim, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(out_dim),
    )
    return model

class inception_module(nn.Module):
    def __init__(self, in_dim, mid_dim1, out_dim3, out_dim1, out_dim5):
        super(inception_module, self).__init__()
        
        self.conv_1_3 = conv_1_3(in_dim, mid_dim1, out_dim3)
        self.max_3_1 = max_3_1(in_dim, out_dim1)
        self.conv_1 = conv_1(in_dim, out_dim1)
        self.conv_1_5 = conv_1_5(in_dim, mid_dim1, out_dim5)
        
    def forward(self, x):
        out1 = self.conv_1_3(x)
        out2 = self.max_3_1(x)
        out3 = self.conv_1(x)
        out4 = self.conv_1_5(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        
        return out

class ResNetBlock(BaseFeaturesExtractor):
    #residual Block 6 layers
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(ResNetBlock, self).__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.relu = nn.ReLU()
        #layers definition
        self.layer_1 = inception_module(n_input_channels, 16, 18, 18, 18)
        self.layer_2 = inception_module(72, 16, 18, 18, 18)
        self.layer_3 = inception_module(72, 16, 18, 18, 18)
        self.layer_4 = inception_module(72, 16, 18, 18, 18)
        self.layer_5 = inception_module(72, 16, 18, 18, 18)
        self.layer_6 = inception_module(72, 16, 18, 18, 18)
        #normalization layer
        self.norm = nn.BatchNorm2d(72)
        #flatten layer
        self.flatten = nn.Flatten()
       
        with torch.no_grad():
            dummy_input = torch.as_tensor(observation_space.sample()[None]).float()
            dummy_output = self.skip_connection(dummy_input)
            flatten_size = self.flatten(dummy_output).shape[1]
            
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, features_dim),
            nn.ReLU()
        )
            

    def skip_connection(self, observations):
        #first residual connection
        out1 = self.relu(self.layer_1(observations))
        out2 = self.relu(self.layer_2(out1))
        out3 = self.layer_3(out2)
        residual_1 = self.relu(out1 + out3)
    
        #second residual connection
        out4 = self.relu(residual_1)
        out5 = self.relu(self.layer_4(out4))
        residual_2 = self.relu(residual_1 + out5)
    
        #final layer and normalization
        out_6 = self.layer_6(residual_2)
        final_features = self.relu(self.norm(out_6))
        
        return final_features

    def forward(self, observations):
        features = self.skip_connection(observations)
    
        return self.linear(features)
        
           
           
        
        