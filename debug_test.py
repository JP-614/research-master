# debug_test.py

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.envs import FakeImageEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 1. 테스트를 위한 최소한의 특징 추출기 정의
class MinimalExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 16):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# 2. 테스트를 위한 가상 환경 생성 (이미지 환경)
# screen_height, screen_width, n_channels
env = FakeImageEnv(screen_height=40, screen_width=40, n_channels=3, discrete=True)

# 3. 문제가 되는 policy_kwargs 구조를 그대로 사용
policy_kwargs = dict(
    features_extractor_class=MinimalExtractor,
    features_extractor_kwargs=dict(features_dim=16),
)

# 4. DQN 모델 생성 시도
try:
    print("--- 디버그 테스트 시작 ---")
    model = DQN(
        policy='CnnPolicy',
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    print("\n✅✅✅ 테스트 성공! Stable-Baselines3와 사용자 정의 특징 추출기 연동에 문제가 없습니다.")
    print("      원인은 emg_env.py 또는 cnn_feature_extractor.py 파일 자체의 미세한 문제입니다.")

except TypeError as e:
    print("\n❌❌❌ 테스트 실패! 여전히 동일한 에러가 발생했습니다.")
    print("      이것은 코드 문제가 아니라, 'py3103' Conda 환경 자체가 손상되었을 가능성이 매우 높습니다.")
    print(f"      발생한 에러: {e}")

except Exception as e:
    print(f"\n다른 에러 발생: {e}")