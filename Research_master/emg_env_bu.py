import gymnasium as gym # Stable Baselines3が使用する強化学習環境の標準
from gymnasium import spaces    # 行動/観測空間を定義するために必要
import numpy as np
import os   # ファイルパスを扱うためのライブラリ
import glob # 特定のパターンのファイルリストを簡単に見つけるためのライブラリ
import librosa
from collections import Counter
import random   # データの順序をシャッフルするために必要
from scipy.ndimage import gaussian_filter1d
import re  # <-- 追加
from collections import defaultdict # <-- 追加

class emg_env(gym.Env):
    def __init__(self, data_dir, train=True, window_points=800, stride_points=100):
        super(emg_env, self).__init__()
        
        self.fs = 1600
        self.window_points = window_points
        self.stride_points = stride_points
        self.stft_nperseg = 24
        self.stft_noverlap = 12
        self.stft_win = 'hamming'
        self.gaussian_sigma = 1.5
        self.max_freq_hz = 400
        
        file_list = glob.glob(os.path.join(data_dir, '*.npz'))
        if not file_list:
            raise ValueError(f"'{data_dir}' パスに ep_*.npz で始まる NPZ ファイルが存在しない。")
        # ファイルの順序がOSによって異なる場合があるため、まず名前順にソートして再現性を確保する。
# 2. データシャッフルロジック（中核の修正部分）
        if train:
            print("[INFO] データをラベルごとに均等にシャッフルする (Round-Robin)...")
            files_by_label = defaultdict(list)
            
            # (1) ファイルをラベルごとに分類
            for f_path in file_list:
                fname = os.path.basename(f_path)
                match = re.search(r'label(\d+)', fname) # ファイル名から「label数字」を探す
                label = int(match.group(1)) if match else -1
                files_by_label[label].append(f_path)
            
            # (2) 各ラベル内では順序をランダムにシャッフル
            sorted_labels = sorted(files_by_label.keys())
            max_len = 0
            for label in sorted_labels:
                random.shuffle(files_by_label[label])
                max_len = max(max_len, len(files_by_label[label]))
            
            # (3) ラベルごとに1つずつ交互に抽出 (0, 1, 2, 0, 1, 2...)
            balanced_list = []
            for i in range(max_len):
                for label in sorted_labels:
                    if i < len(files_by_label[label]):
                        balanced_list.append(files_by_label[label][i])
            
            self.npz_files = balanced_list
            print(f"[INFO] 計 {len(self.npz_files)} 個のファイル整列完了。")
            
        else:
            # テスト時は単純に名前順でソート
            self.npz_files = sorted(file_list)
        
        # もし学習(is_train)モードなら、データの順序をランダムにシャッフルする。
        # これにより、モデルがデータの順序に過学習するのを防ぎ、学習性能を向上させる。
        
        # --- 3. 環境の状態を記憶するための変数初期化 ---
        self.current_file_idx = 0   # 次のエピソードで使用するファイル番号
        self.current_window_idx = 0 # 現在のエピソード内のウィンドウ番号
        self.episode_windows = []   # 現在のエピソードの全ウィンドウデータ
        self.episode_predictions = []   # 現在のエピソードの全予測記録
        self.episode_label = None   # 現在のエピソードの正解ラベル
        
        # --- 4. 行動空間(Action Space)の定義 ---
        # エージェントに与えられる観測(データ)の形状(shape)と範囲を定義する。
        # 観測は「1つのウィンドウ」で作成したスペクトログラム画像。
        # まずスペクトログラムのshapeを計算する必要がある。
        actions = 7
        # spaces.Discrete(6)は0から5までの整数のうち1つを行動として選択できるという意味。
        self.action_space = spaces.Discrete(actions)
        
        # --- 5. 観測空間(Observation Space)の定義 ---
        dummy_window = np.random.randn(self.window_points)
        dummy_stft = librosa.stft(
            dummy_window, 
            n_fft=self.stft_nperseg, 
            hop_length=self.stft_noverlap, 
            window=self.stft_win
            )
        # 周波数cutoffと大きさを合わせる
        cutoff_idx = int(self.max_freq_hz * self.stft_nperseg / self.fs)
        cropped_stft = dummy_stft[:cutoff_idx + 1, :]
       
        num_channels = 6
        freq_shape =  cropped_stft.shape[0]
        time_shape =  cropped_stft.shape[1]
        
        # SB3のCnnPolicyは(チャンネル、高さ、幅)の順序の入力を好むため、shapeを(6, 129, 24)に設定する。
        total_shape = (num_channels, freq_shape, time_shape)
        
        # spaces.Boxは値が連続的なN次元配列形態の空間を意味し、画像データに主に使用される。
        # low=0, high=np.infはスペクトログラムの各要素の値が0以上であるという意味。
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=total_shape, dtype=np.float32)
        
    # 3秒間の全体信号を受け取り、設定されたウィンドウ/ストライド値に従って切り取る内部関数。
    def create_windows(self, full_signal):
        windows = []
        num_samples = full_signal.shape[1] # 全体のサンプル数 (例: 4800)
        # forループを回しながらstride_pointsずつ移動し、window_pointsの長さで信号を切り取る。
        for i in range(0, num_samples - self.window_points + 1, self.stride_points):
            window = full_signal[:, i:i + self.window_points]
            windows.append(window)
        return windows
    
    # 現在のウィンドウ(raw signal)をスペクトログラム画像(観測)に変換する内部関数。
    def get_observation(self):
        # 現在のウィンドウのしおり(current_window_idx)に該当するウィンドウデータを取得する。
        raw_window = self.episode_windows[self.current_window_idx] # shape: (6, 300)
        
        spec_list = []
        for ch in range(raw_window.shape[0]): # 6つのチャンネルに対してそれぞれ処理
            # test2.pyと同じロジックで整数型のraw signalを電圧値に変換し、整流(rectify)する。
            emg_arr = 10.0 - (65535.0 - raw_window[ch].astype(np.float32))/65535.0*20.0
            rect_signal = np.abs(emg_arr)
            
            # librosaライブラリを使用してSTFTを計算する。
            stft = librosa.stft(
                rect_signal, 
                n_fft=self.stft_nperseg, 
                hop_length=self.stft_noverlap,
                window=self.stft_win
                )
            # Power Spectrogram (パワースペクトログラム)を計算する。
            power_spec = np.abs(stft)**2
            smoothed_spec = gaussian_filter1d(power_spec, sigma=self.gaussian_sigma, axis=0)
            cutoff_idx = int(self.max_freq_hz * self.stft_nperseg / self.fs)
            cropped_spec = smoothed_spec[:cutoff_idx + 1, :]

            spec_list.append(cropped_spec)
        
        # 6つのチャンネルのスペクトログラムを1つの3D配列に結合する。
        spec_tensor = np.stack(spec_list, axis=0) # shape: (6, F, T)
        
        # CnnPolicyが好む(チャンネル、高さ、幅)の形態であるため、transposeせずにそのまま返す。
        return spec_tensor.astype(np.float32)
    
    # エピソードが開始または終了したときに呼び出され、環境を初期状態に戻す関数。
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
         # もしすべてのファイルを使い切った場合、再び最初のファイルから開始する。
        if self.current_file_idx >= len(self.npz_files):
            self.current_file_idx = 0
            print("すべてのデータを一度ずつ使用した。最初からやり直す。")

        # 現在のファイルのしおり(current_file_idx)に該当するファイルパスを取得する。
        file_path = self.npz_files[self.current_file_idx]
        
        # np.loadを使用して .npz ファイルを開き、'raw_signal'と'label'キーの値を読み込む。
        with np.load(file_path) as data:
            full_raw_signal = data['raw_signal']
            self.episode_label = int(data['label'])
        # 読み込んだファイル名と正解ラベルを出力する。
        file_name = os.path.basename(file_path)
        print(f"\n--- New Episode --- Loading file: {file_name}, Label: {self.episode_label} ---")
        # 次のエピソードのためにファイルのしおりを1増やす。
        self.current_file_idx +=1
        
        # 新しいエピソードのために状態変数をすべて初期化する。
        self.episode_windows = self.create_windows(full_raw_signal) # ウィンドウ保管箱を満たす
        self.current_window_idx = 0  # ウィンドウのしおりを0に
        self.episode_predictions = [] # 予測記録箱を空にする
        
         # 最初の観測(最初のウィンドウのスペクトログラム)を生成する。
        observation = self.get_observation()
        info = {} # 追加情報を格納する辞書
        
        # reset関数は(最初の観測、情報辞書)のタプルを返す必要がある。
        return observation, info
    
    def calculate_reward(self):
        if not self.episode_predictions:
            return 0.0
        # 最も頻繁に予測されたジェスチャー(mode)を探す。
        gesture = Counter(self.episode_predictions).most_common(1)[0][0]
        # 全体のウィンドウのうち、mode_gestureが占める割合を計算する。
        overlap_factor = self.episode_predictions.count(gesture) / len(self.episode_windows)
        # mode_gestureが実際の正解と同じで、その割合が70%を超えればボーナス+10点を返す。
        if gesture == self.episode_label and overlap_factor > 0.7:
            return 10.0
        
        return 0.0
    
    # エージェントが行動(action)を取ったときに呼び出される、環境の中核となる関数。
    def step(self, action):
        # 1. 分類報酬(Classification Reward)の計算: エージェントの予測(action)と実際の正解を比較する。
        if action == self.episode_label:
            reward = 1.0
        else:
            reward = -1.0
        # 2. エージェントの予測を記録箱に追加する。
        self.episode_predictions.append(action.item())
        # 3. 次のウィンドウを処理するためにウィンドウのしおりを1増やす。
        self.current_window_idx += 1
        # 4. エピソード終了の可否を確認する。
        # もしウィンドウのしおりが全体のウィンドウ数以上になれば、エピソード終了となる。
        if self.current_window_idx >= len(self.episode_windows):
            terminated = True
            # エピソード終了時にのみ認識ボーナス報酬を計算し、現在の報酬に加える。
            reward += self.calculate_reward()
            # 次の観測はないため、0で埋められた空の観測を返す。
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            terminated = False
            # 次のウィンドウのスペクトログラムを次の観測として生成する。
            observation = self.get_observation()
            
        truncated = False
        info = {}
        
        # step関数は(次の観測、報酬、終了フラグ、中断フラグ、情報)のタプルを返す必要がある。
        return observation, reward, terminated, truncated, info
