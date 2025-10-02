# -*- coding: utf-8 -*-
# CONTEC(AIO) 6ch EMG
# - 3秒(learning_time)ずつ「動作番号」を付けて保存
#   * 生筋電(6ch)+label -> CSV 追加保存
#   * (整流→LPF 15Hz)後 全3秒に対して STFT(24/12) -> (6,F,T) を .npz 保存
# - 実時間での簡易スペクトログラム表示付き（参考）
#
# 変数/スタイルは既存コードに合わせて命名 (buffer_size, MAV_buffer_size, n, fs, shift, win, dim ...)

import AIO
import sys, os, csv, datetime, time
from collections import deque
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

'''
import tensorflow as tf
from keras import layers, Model, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
'''
##############################################################################
name = 'デモ1'
num_of_channels = 6  # 計測チャンネル数
CONTEC=b'AIO002'
#######################################################################################################
buffer_size = 1600        # [Hz] (サンプリング周波数と同義で使用)
#######################################################################################################

MAV_buffer_size = 800     # モニタ用リングバッファ
n   = 128                 # STFTのパラメータ(既存スタイル維持
fs  = buffer_size         # サンプリング周波数
shift = n//2              # STFTのパラメータ
win   = 'hamming'     # STFTのパラメータ
dim = 16

# --- 학습(計測)時間: 3秒 -------------------------------------------
learning_time    = 3                              # [s]
EPISODE_SAMPLES  = int(fs * learning_time)        # 3秒分サンプル数 (=4800 at fs=1600)

# --- 論文パイプライン ------------------------------------------------------
# 内部 STFT（論文は 24/12）
STFT_NPERSEG  = 256
STFT_NOVERLAP = 128
max_freq_hz = 400
# 保存先
BASE_SAVE_DIR = 'リアルタイム_data'
train_data_dir = os.path.join(BASE_SAVE_DIR, 'train')
test_data_dir = os.path.join(BASE_SAVE_DIR, 'test')
raw_csv_save_dir = os.path.join(BASE_SAVE_DIR, 'raw_csv')
os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(test_data_dir, exist_ok=True)
os.makedirs(raw_csv_save_dir, exist_ok=True)

# 生筋電 CSV
filename1 = os.path.join(raw_csv_save_dir, name+'生筋電.csv')

with open(filename1, 'w', newline='') as f:
    pass
# contec
def init_contec_and_start(num_ch: int, sampling_hz: float):
    """ CONTEC(AIO) 初期化と開始（AIO.py 標準手順） """
    device_name = AIO.queryAioDeviceName()
    try:
        aio = AIO.AIO(device_name[0][0])
    except ValueError as ve:
        print(ve); sys.exit(1)

    aio.resetDevice()
    aio.resetAiStatus()
    aio.resetAiMemory()

    print('MaxCh:%d' % aio.getAiMaxChannels())
    aio.setAiChannels(num_ch);           print('UseCh:%d' % aio.getAiChannels())
    aio.setAiMemoryType(0);              print('MemoryType:%d' % aio.getAiMemoryType())
    aio.setAiClockType(0);               print('ClockType:%d' % aio.getAiClockType())
    aio.setAiSamplingClock(1000000.0/sampling_hz); print('SamplingClock:%f' % aio.getAiSamplingClock())
    aio.setAiStartTrigger(0);            print('StartTrigger:%d' % aio.getAiStartTrigger())
    aio.setAiStopTrigger(4);             print('StopTrigger:%d' % aio.getAiStopTrigger())

    aio.resetAiMemory()
    aio.startAi()
    print('Sampling start')
    return aio

if __name__ == '__main__':
    # AIO start
    print("스크립트 실행 시작.")
    print("CONTEC 장치 초기화를 시작합니다...")
    aio = init_contec_and_start(num_of_channels, float(buffer_size))
    
    print("CONTEC 장치 초기화 완료.")
    
    # モニタ用リングバッファ（既存スタイル）
    cs = 40960  # contec 初期値（0-5V 範囲時）
    signal_buffer = []
    for ch in range(num_of_channels):
        signal_buffer.append(deque([], maxlen=buffer_size))
        signal_buffer[ch].extend([cs]*buffer_size)

    try:
        print("이제 입력 대기 루프로 진입합니다.")
        while True:
            # --- 動作番号の入力 (measure スタイル) -----------------------------
            key = input('동작 번호를 입력하세요 (예: 0, 1, 2... / 종료: end): ')
            if key == "end":
                print("학습 데이터 측정을 종료합니다.")
                break
            try:
                prr = int(key)
            except ValueError:
                print("숫자 또는 'end'를 입력해주세요.")
                continue
            
            #train->test 2回収集
            for collection in ['train', 'test']:
                print("\n=======================================================")
                if collection == 'train':
                    print(f"동작 [{prr}]의 << 훈련(TRAIN) >> 데이터를 수집합니다.")
                    save_dir = train_data_dir
                    input("준비가 되면 Enter 키를 누르세요...") # 사용자가 준비될 때까지 대기
                else: # collection == 'test'
                    print(f"동작 [{prr}]의 << 테스트(TEST) >> 데이터를 수집합니다.")
                    save_dir = test_data_dir
                    input("이어서 테스트 데이터 수집을 시작하려면 Enter 키를 누르세요...") # 사용자가 준비될 때까지 대기
                
                print("... 수집 시작! (3초간 동작을 유지해주세요)")
                
                # 3秒エピソード用バッファ
                raw_data = [deque([], maxlen=EPISODE_SAMPLES) for _ in range(num_of_channels)]
                # フィルタ(整流+LPF)後の全3秒を後で STFT するための累積
                ep_rect = [ [] for _ in range(num_of_channels) ]

                # AIO バッファを空にして開始
                # 에피소드 시작 직전에
                try:
                    aio.stopAi()          # ← 이미 돌고 있으면 우선 정지
                except:
                    pass

                aio.resetAiMemory()
                aio.startAi()

                count = 0
                stride_cnt = 0

                # --- エピソード(3秒)本体ループ -----------------------------------
                while count < EPISODE_SAMPLES:
                    num_of_sampling = aio.getAiSamplingCount()
                    if num_of_sampling <= 0:
                        continue
                    if num_of_sampling > buffer_size:
                        num_of_sampling = buffer_size

                    try:
                        smp_count, data_blk = aio.getAiSamplingData(num_of_sampling)  # 整数スケール
                    except ValueError as ve:
                        print(ve); continue

                    if smp_count == 0:
                        continue

                    # 各 ch について: raw を貯める + 可視化用バッファ更新 + (整流→LPF) を累積
                    for ch in range(num_of_channels):
                        chunk = data_blk[ch::num_of_channels]      # このブロックの ch データ(整数)
                        raw_data[ch].extend(chunk)                  # 3秒の raw を保存
                        signal_buffer[ch].extend(chunk)             # モニタ用

                        raw_arr = np.asarray(chunk, dtype=np.float32)
                        emg_arr = 10.0 - (65535.0 - raw_arr)/65535.0*20.0
                        rect    = np.abs(emg_arr)
                       
                        ep_rect[ch].append(rect)   # 後で全3秒分を連結して STFT

                    count += smp_count
                            
                # 3秒採取終了
                aio.stopAi()
                print("3초 분량의 에피소드 수집 완료. 저장을 실행합니다...")
                
                # ---  수정된 부분: Raw Signal을 (채널, 샘플수) 형태로 저장 ---
                # 강화학습 환경(emg_env.py)에서 사용할 수 있도록 Raw Signal을 저장합니다.
                raw_signal_to_save = np.array(raw_data, dtype=np.uint16) # (6, 4800)
                
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                file_name = f"ep_{ts}_label{prr}.npz"
                np.savez_compressed(
                    os.path.join(save_dir, file_name),
                    raw_signal=raw_signal_to_save,
                    label=prr,
                    fs=fs
                )
                print(f"저장 완료: {file_name} (shape={raw_signal_to_save.shape})")
                
                # --- (1) 生筋電CSV へ追加保存（measure と同様） --------------------
                type_col = np.full((EPISODE_SAMPLES, 1), collection) # 'train' 또는 'test' 문자열
                label_col = np.full((EPISODE_SAMPLES, 1), prr) # 동작 번호
                #   semg = [raw_ch0, raw_ch1, ..., raw_ch5, label]
                semg = np.c_[tuple(np.asarray(raw_data[c], dtype=np.int32) for c in range(num_of_channels)) + (label_col, type_col)]

                with open(filename1, 'a', newline='') as f:
                    writer = csv.writer(f)
                    # 파일이 비어있으면 헤더 추가
                    if f.tell() == 0:
                        header = [f'CH{i+1}' for i in range(num_of_channels)] + ['label', 'type']
                        writer.writerow(header)
                    writer.writerows(semg)
                print(f"  -> CSV 추가 완료: {filename1}")
                
                # --- (2) 3秒分の信号でSTFT→tensor生成
                spec_list = []
                for ch in range(num_of_channels):
                    yfull = np.concatenate(ep_rect[ch])
                    #labosa STFT
                    stft = librosa.stft(
                                     yfull,
                                     n_fft=STFT_NPERSEG,
                                     hop_length=STFT_NOVERLAP,
                                     window=win)
                    #power spectrogram計算
                    sigma = 2.5
                    Sxx = (np.abs(stft)**2)    # (F,T)
                    smoothed_Sxx = gaussian_filter1d(Sxx, sigma=sigma, axis=0)
                    # +++ 잘라내기 (필터링된 smoothed_Sxx를 사용) +++
                    cutoff_idx = int(max_freq_hz * STFT_NPERSEG / fs)
                    cropped_Sxx = smoothed_Sxx[:cutoff_idx + 1, :]
                    spec_list.append(cropped_Sxx)
                    
                spec_ep = np.stack(spec_list, axis=0)  # (6,F,T)
                
                # ###################### [수정된 부분 시작] ######################
                # 1. 모든 스펙트로그램을 절대적인 기준(ref=1.0)으로 dB 변환합니다.
                #    이렇게 하면 각 데이터의 실제 에너지 크기를 유지할 수 있습니다.
                db_ep = librosa.power_to_db(spec_ep, ref=1.0)

                # 2. 모든 그래프에 적용할 색상 스케일의 최솟값과 최댓값을 상수로 정의합니다.
                #    이 값들은 직접 데이터를 보면서 적절하게 조절할 수 있습니다.
                #    예를 들어, 신호가 너무 약하게 나오면 VMAX를 낮추거나, 너무 강하면 높일 수 있습니다.
                VMIN = -30  # dB 최솟값
                VMAX = 30  # dB 최댓값
                # ####################### [수정된 부분 끝] #######################
                
                # --- (3) 6 channel spectrogram image
                fig, axes = plt.subplots(2,3, figsize=(15,8), sharex=True, sharey=True, constrained_layout=True)
                axes = axes.ravel()
                for ch in range(num_of_channels):
                    db = db_ep[ch]
                    # vmax와 vmin에 위에서 정의한 상수 값을 넣어 스케일을 고정합니다.
                    img = librosa.display.specshow(db, sr=fs, hop_length=STFT_NOVERLAP,
                                                   x_axis='time', y_axis='hz', ax=axes[ch],
                                                   vmax=VMAX, vmin=VMIN) # <--- 수정된 부분
                    axes[ch].set_ylim(0, max_freq_hz)
                    axes[ch].set_title(f'Channel {ch+1}')
                    axes[ch].set_xlabel('Time(s)')
                    axes[ch].set_ylabel('Frequency(Hz)')
                
                fig.colorbar(img, ax=axes, format='%+2.0f dB', label='Power/Frequency (dB)')
                plt.suptitle(f'Spectrogram for Motion "{prr}" ({collection.upper()})')
                plt.show()

    except KeyboardInterrupt:
        try:
            aio.stopAi()
        except:
            pass
        print("\n[INFO] 사용자에 의해 프로그램이 중지되었습니다.")

    finally:
        # 프로그램 종료 시 AIO 장치 정리
        if 'aio' in locals() and aio:
            try:
                aio.stopAi()
                aio.__del__() # 명시적 소멸자 호출
                print("[INFO] AIO 장치가 정리되었습니다.")
            except Exception as e:
                print(f"[ERROR] AIO 장치 정리 중 오류 발생: {e}")