# cnn_feature_extractor.py (버그가 완전히 수정된 최종 버전)

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNN_feature(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(CNN_feature, self).__init__(observation_space, features_dim)
        
        # 입력 채널 수를 observation_space.shape에서 정수로 올바르게 가져옵니다. (예: 6)
        n_input_channels = observation_space.shape[0]
        
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
      
        self.flatten = nn.Flatten()
        

    
        with torch.no_grad():
            dummy_input = torch.as_tensor(observation_space.sample()[None]).float()
            dummy_output = self.pool2(
                            self.relu2(
                                self.conv2(
                                    self.pool1(
                                        self.relu1(
                                            self.conv1(dummy_input)
                                        )
                                    )
                                )
                            )
                        )
                    
            flatten_size = self.flatten(dummy_output).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(flatten_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        '''print(f'input shape : {observations.shape}')'''
        x = self.pool1(
                self.relu1(
                    self.conv1(observations)
            )
        )
        '''print(f'After pool1 : {x.shape}')'''
        
        x = self.pool2(
                self.relu2(
                    self.conv2(x)
                )
        )
        '''print(f'After pool2 : {x.shape}')'''

        
        x = self.flatten(x)
        '''print(f'After flatten : {x.shape}')'''
        
        return self.linear(x)