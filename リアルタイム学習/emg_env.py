import gymnasium as gym # Stable Baselines3가 사용하는 강화학습 환경 표준
from gymnasium import spaces    # 행동/관측 공간을 정의하기 위해 필요
import numpy as np
import os   # 파일 경로를 다루기 위한 라이브러리
import glob # 특정 패턴의 파일 목록을 쉽게 찾기 위한 라이브러리
import librosa
from collections import Counter
import random   # 데이터 순서를 섞기 위해 필요
class emg_env(gym.Env):
    def __init__(self, data_dir, train=True, window_points=300, stride_points=40):
        super(emg_env, self).__init__()
        
        self.fs = 1600
        self.window_points = window_points
        self.stride_points = stride_points
        self.stft_nperseg = 256
        self.stft_noverlap = 128
        self.stft_win = 'hamming'
        
        file_list = glob.glob(os.path.join(data_dir, '*.npz'))
        if not file_list:
            raise ValueError(f"'{data_dir}' 경로에 NPZ 파일이 없습니다.")
        # 파일 순서가 OS마다 다를 수 있으므로, 먼저 이름순으로 정렬하여 재현성을 확보합니다.
        sorted_list = sorted(file_list)
        
        # 만약 학습(is_train) 모드라면, 데이터 순서를 무작위로 섞어줍니다.
        # 이는 모델이 데이터 순서에 과적합되는 것을 방지해 학습 성능을 높여줍니다.

        if train:
            random.shuffle(sorted_list) # 학습용 데이터는 섞어줍니다.
        # 최종적으로 사용할 파일 리스트를 self 변수에 저장합니다
        self.npz_files = sorted_list
        
        # --- 3. 환경의 상태를 기억하기 위한 변수 초기화 ---
        self.current_file_idx = 0   #다음 에피소드에 쓸 파일 번호
        self.current_window_idx = 0 #현재 에피소드 내의 윈도우 번호
        self.episode_windows = []   #현재 에피소드의 모든 윈도우 데이터
        self.episode_predictions = []   #현재 에피소드의 모든 예측 기록
        self.episode_label = None   #현재 에피소드의 정답
        
        # --- 4. 행동 공간(Action Space) 정의 ---
        # 에이전트에게 주어지는 관측(데이터)의 형태(shape)와 범위를 정의합니다.
        # 관측은 '하나의 윈도우'로 만든 스펙트로그램 이미지입니다.
        # 먼저 스펙트로그램의 shape을 계산해야 합니다.
        actions = 7
        # spaces.Discrete(6)은 0부터 5까지의 정수 중 하나를 행동으로 선택할 수 있다는 의미입니다.
        self.action_space = spaces.Discrete(actions)
        
        # --- 5. 관측 공간(Observation Space) 정의 ---
        # 에이전트에게 주어지는 관측(데이터)의 형태(shape)와 범위를 정의합니다.
        # 관측은 '하나의 윈도우'로 만든 스펙트로그램 이미지입니다.
        # 먼저 스펙트로그램의 shape을 계산해야 합니다.
        dummy_window = np.random.randn(self.window_points)
        dummy_stft = librosa.stft(
            dummy_window, 
            n_fft=self.stft_nperseg, 
            hop_length=self.stft_noverlap, 
            window=self.stft_win
            )
        
        num_channels = 6
        freq_shape = dummy_stft.shape[0]
        time_shape = dummy_stft.shape[1]
        
        # SB3의 CnnPolicy는 (채널, 높이, 너비) 순서의 입력을 선호하므로, shape을 (6, 129, 24)로 설정합니다.
        total_shape = (num_channels, freq_shape, time_shape)
        
        # spaces.Box는 값이 연속적인 N차원 배열 형태의 공간을 의미합니다. 이미지 데이터에 주로 사용됩니다.
        # low=0, high=np.inf는 스펙트로그램의 각 원소 값이 0 이상이라는 의미입니다.
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=total_shape, dtype=np.float32)
        
        # 3초 길이의 전체 신호를 받아서, 설정된 윈도우/스트라이드 값에 따라 잘라주는 내부 함수입니다.
    def create_windows(self, full_signal):
        windows = []
        num_samples = full_signal.shape[1] # 전체 샘플 수 (예: 4800)
        # for loop를 돌면서 stride_points 만큼씩 이동하며 window_points 길이로 신호를 잘라냅니다.
        for i in range(0, num_samples - self.window_points + 1, self.stride_points):
            window = full_signal[:, i:i + self.window_points]
            windows.append(window)
        return windows
    
    # 현재 윈도우(raw signal)를 스펙트로그램 이미지(관측)로 변환하는 내부 함수입니다.
    def get_observation(self):
        # 현재 윈도우 책갈피(current_window_idx)에 해당하는 윈도우 데이터를 가져옵니다.
        raw_window = self.episode_windows[self.current_window_idx] # shape: (6, 300)
        
        spec_list = []
        for ch in range(raw_window.shape[0]): # 6개 채널에 대해 각각 처리
            # test2.py와 동일한 로직으로 정수형 raw signal을 전압값으로 변환하고 정류(rectify)합니다.
            emg_arr = 10.0 - (65535.0 - raw_window[ch].astype(np.float32))/65535.0*20.0
            rect_signal = np.abs(emg_arr)
            
            # librosa 라이브러리를 사용해 STFT를 계산합니다.
            stft = librosa.stft(
                rect_signal, 
                n_fft=self.stft_nperseg, 
                hop_length=self.stft_noverlap,
                window=self.stft_win
                )
            # Power Spectrogram (파워 스펙트로그램)을 계산합니다.
            power_spec = np.abs(stft)**2
            spec_list.append(power_spec)
        
        # 6개 채널의 스펙트로그램을 하나의 3D 배열로 합칩니다.
        spec_tensor = np.stack(spec_list, axis=0) # shape: (6, F, T)
        
        # CnnPolicy가 선호하는 (채널, 높이, 너비) 형태이므로 transpose 없이 그대로 반환합니다.
        return spec_tensor.astype(np.float32)
    
    # 에피소드가 시작되거나 끝났을 때 호출되어 환경을 초기 상태로 되돌리는 함수입니다.
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
         # 만약 모든 파일을 다 사용했다면, 다시 처음 파일부터 시작합니다.
        if self.current_file_idx >= len(self.npz_files):
            self.current_file_idx = 0
            print("모든 데이터를 한 번씩 사용했습니다. 처음부터 다시 시작합니다.")

        # 현재 파일 책갈피(current_file_idx)에 해당하는 파일 경로를 가져옵니다.
        file_path = self.npz_files[self.current_file_idx]
        
        # np.load를 사용해 .npz 파일을 열고, 'raw_signal'과 'label' 키의 값을 읽어옵니다.
        with np.load(file_path) as data:
            full_raw_signal = data['raw_signal']
            self.episode_label = int(data['label'])
        # 다음 에피소드를 위해 파일 책갈피를 1 증가시킵니다.
        self.current_file_idx +=1
        
        # 새로운 에피소드를 위해 상태 변수들을 모두 초기화합니다.
        self.episode_windows = self.create_windows(full_raw_signal) # 윈도우 보관함 채우기
        self.current_window_idx = 0  # 윈도우 책갈피를 0으로
        self.episode_predictions = [] # 예측 기록통 비우기
        
         # 첫 번째 관측(첫 윈도우의 스펙트로그램)을 생성합니다.
        observation = self.get_observation()
        info = {} # 추가 정보를 담는 딕셔너리
        
        # reset 함수는 (첫 번째 관측, 정보 딕셔너리) 튜플을 반환해야 합니다.
        return observation, info
    
    def calculate_reward(self):
        if not self.episode_predictions:
            return 0.0
        # 가장 빈번하게 예측된 제스처(mode)를 찾습니다.
        gesture = Counter(self.episode_predictions).most_common(1)[0][0]
        # 전체 윈도우 중 mode_gesture가 차지하는 비율을 계산합니다.
        overlap_factor = self.episode_predictions.count(gesture) / len(self.episode_windows)
        # mode_gesture가 실제 정답과 같고, 그 비율이 70%를 넘으면 보너스 +10점을 반환합니다.
        if gesture == self.episode_label and overlap_factor > 0.7:
            return 10.0
        
        return 0.0
    # 에이전트가 행동(action)을 취했을 때 호출되는, 환경의 핵심 함수입니다.
    def step(self, action):
        # 1. 분류 보상(Classification Reward) 계산: 에이전트의 예측(action)과 실제 정답을 비교합니다.
        if action == self.episode_label:
            reward = 1.0
        else:
            reward = -1.0
        # 2. 에이전트의 예측을 기록통에 추가합니다.
        self.episode_predictions.append(action)
        # 3. 다음 윈도우를 처리하기 위해 윈도우 책갈피를 1 증가시킵니다.
        self.current_window_idx += 1
        # 4. 에피소드 종료 여부를 확인합니다.
        # 만약 윈도우 책갈피가 전체 윈도우 개수보다 크거나 같아지면, 에피소드가 끝난 것입니다.
        if self.current_window_idx >= len(self.episode_windows):
            terminated = True
            # 에피소드 종료 시에만 인식 보너스 보상을 계산하여 현재 보상에 더해줍니다.
            reward += self.calculate_reward()
            # 다음 관측은 없으므로, 0으로 채워진 빈 관측을 반환합니다.
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            terminated = False
            # 다음 윈도우의 스펙트로그램을 다음 관측으로 생성합니다.
            observation = self.get_observation()
            
        truncated = False
        info = {}
        
        # step 함수는 (다음 관측, 보상, 종료 여부, 중단 여부, 정보) 튜플을 반환해야 합니다.
        return observation, reward, terminated, truncated, info