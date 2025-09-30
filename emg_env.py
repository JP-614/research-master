# emg_env.py (최종 클린 버전)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import glob
import librosa
import random
from collections import Counter
from scipy.ndimage import gaussian_filter1d

class emg_env(gym.Env):
    def __init__(self, data_dir, train=True, window_points=300, stride_points=40):
        super(emg_env, self).__init__()
        
        self.fs = 1600
        self.window_points = window_points
        self.stride_points = stride_points
        self.stft_nperseg = 64
        self.stft_noverlap = 32
        self.stft_win = 'hamming'
        self.max_freq_hz = 400
        self.gaussian_sigma = 1.5

        file_list = glob.glob(os.path.join(data_dir, '*.npz'))
        if not file_list:
            raise ValueError(f"'{data_dir}' 경로에 NPZ 파일이 없습니다.")
        
        sorted_list = sorted(file_list)
        if train:
            random.shuffle(sorted_list)
        self.npz_files = sorted_list
        
        self.current_file_idx = 0
        self.current_window_idx = 0
        self.episode_windows = []
        self.episode_predictions = []
        self.episode_label = None
        
        actions = 7
        self.action_space = spaces.Discrete(actions)
        
        cutoff_idx = int(self.max_freq_hz * self.stft_nperseg / self.fs)
        freq_shape = cutoff_idx + 1
        time_shape = int(np.floor((self.window_points - self.stft_nperseg) / self.stft_noverlap)) + 1
        num_channels = 6
        total_shape = (num_channels, freq_shape, time_shape)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=total_shape, dtype=np.float32)

    def create_windows(self, full_signal):
        windows = []
        num_samples = full_signal.shape[1]
        for i in range(0, num_samples - self.window_points + 1, self.stride_points):
            window = full_signal[:, i:i + self.window_points]
            windows.append(window)
        return windows

    def get_observation(self):
        raw_window = self.episode_windows[self.current_window_idx]
        
        spec_list = []
        for ch in range(raw_window.shape[0]):
            emg_arr = 10.0 - (65535.0 - raw_window[ch].astype(np.float32))/65535.0*20.0
            rect_signal = np.abs(emg_arr)
            
            stft = librosa.stft(rect_signal, n_fft=self.stft_nperseg, hop_length=self.stft_noverlap, window=self.stft_win)
            power_spec = np.abs(stft)**2
            
            smoothed_spec = gaussian_filter1d(power_spec, sigma=self.gaussian_sigma, axis=0)
            
            cutoff_idx = int(self.max_freq_hz * self.stft_nperseg / self.fs)
            cropped_spec = smoothed_spec[:cutoff_idx + 1, :]
            
            spec_list.append(cropped_spec)

        spec_tensor = np.stack(spec_list, axis=0)
        return spec_tensor.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.current_file_idx >= len(self.npz_files):
            self.current_file_idx = 0
            print("모든 데이터를 한 번씩 사용했습니다. 처음부터 다시 시작합니다.")

        file_path = self.npz_files[self.current_file_idx]
        
        # emg_env 에서는 raw_signal이 아닌 spec을 사용하지 않으므로, 이 부분을 수정합니다.
        # test2.py에서 raw_signal을 저장했다면 아래 코드를 사용합니다.
        with np.load(file_path) as data:
            full_raw_signal = data['raw_signal'] 
            self.episode_label = int(data['label'])

        self.current_file_idx +=1
        
        self.episode_windows = self.create_windows(full_raw_signal)
        self.current_window_idx = 0
        self.episode_predictions = []
        
        observation = self.get_observation()
        info = {}
        return observation, info

    def calculate_reward(self):
        if not self.episode_predictions:
            return 0.0
        
        gesture = Counter(self.episode_predictions).most_common(1)[0][0]
        overlap_factor = self.episode_predictions.count(gesture) / len(self.episode_windows)
        
        if gesture == self.episode_label and overlap_factor > 0.7:
            return 10.0
        
        return 0.0

    def step(self, action):
        if action == self.episode_label:
            reward = 1.0
        else:
            reward = -1.0
        
        self.episode_predictions.append(action)
        self.current_window_idx += 1
        
        if self.current_window_idx >= len(self.episode_windows):
            terminated = True
            reward += self.calculate_reward()
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            terminated = False
            observation = self.get_observation()
            
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info