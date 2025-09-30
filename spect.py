# -*- coding: utf-8 -*-
# CONTEC(AIO) 6ch EMG -> (논문식) Spectrogram (6, F, T) 실시간 표시 + (선택) 저장
# - 정류 -> Butterworth 4차 15Hz (sosfilt, causal) -> STFT(24/12) -> |·|^2
# - 변수/흐름: 기존 스타일의 이름 유지(buffer_size, MAV_buffer_size, n, fs, shift, win, dim 등)

import AIO
import sys, os, csv, datetime
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfilt_zi, stft

##############################################################################
name = 'デモ'
CONTEC=b'AIO003'
#COM_number='COM6'

num_of_channels = 6   # 計測チャンネル数

#######################################################################################################
buffer_size = 1600        # [Hz] (サンプリング周波数と同義で使用)
#######################################################################################################

MAV_buffer_size   = 800       # 既存スタイルのバッファ長を流用

n = 128                       # STFTのパラメータ(既存スタイル維持)
fs = buffer_size              # STFTのパラメータ(=サンプリング周波数)
"""
shift = n//2                  # STFTのパラメータ
win = np.hamming(n)           # STFTのパラメータ
dim = 16                      # 既存コード互換のため残置(本スクリプトでは未使用)
"""
# 計測周波数[Hz]
sampling_rate = float(buffer_size)

# 計測を打ち切るデータ数（デモ用）
limit_of_sampling = buffer_size*10*2*15

# ───────── 論文パイプラインのパラメータ ─────────
# ウィンドウ長・ストライド（論文は W=300, stride=40）
W_spec       = 300
STRIDE_spec  = 40

# 低域通過フィルタ：Butterworth 4次, 15 Hz（正方向のみで実時間処理）
LP_ORDER  = 4
LP_CUTOFF = 15.0  # [Hz]
sos_lpf   = butter(LP_ORDER, LP_CUTOFF/(fs/2.0), btype='low', output='sos')
zi_lpf    = [sosfilt_zi(sos_lpf) for _ in range(num_of_channels)]  # チャンネル別フィルタ状態

# 内部 STFT（論文は nperseg=24, noverlap=12）
STFT_NPERSEG  = 24
STFT_NOVERLAP = 12

# 3Dテンザ保存（任意）
SAVE_SPEC_EVERY = 20             # 0で保存無効 / フレームカウント基準で保存
spec_save_dir   = 'リアルタイム/spec_tensor'
os.makedirs(spec_save_dir, exist_ok=True)

# 画像保存（任意、論文風 png）
SAVE_FIG_EVERY = 60              # 0で保存無効
spec_fig_dir   = 'リアルタイム/spec_fig'
os.makedirs(spec_fig_dir, exist_ok=True)

def init_contec_and_start(num_ch: int, sampling_hz: float):
    """ CONTEC(AIO) 初期化と開始(提供されたAIOモジュールの標準手順に準拠) """
    device_name = AIO.queryAioDeviceName()  # 例: [[b'AIO000', b'AIO-XXXX']]
    try:
        aio = AIO.AIO(device_name[0][0])
    except ValueError as ve:
        print(ve); sys.exit(1)

    # デバイス初期化
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

    # 変換スタート
    aio.resetAiMemory()
    aio.startAi()
    print('Sampling start')
    return aio

if __name__ == '__main__':
    # 1) AIO開始
    aio = init_contec_and_start(num_of_channels, sampling_rate)

    # 2) バッファの準備
    cs = 40960
    signal_buffer = []
    for ch in range(0, num_of_channels):
        signal_buffer.append(deque([], maxlen=MAV_buffer_size))
        signal_buffer[ch].extend([cs]*MAV_buffer_size)

    #    STFT入力（“正方向LPF出力”の最新 W_spec サンプルを保持）
    emg_lp = [deque([], maxlen=W_spec) for _ in range(num_of_channels)]
    for ch in range(num_of_channels):
        emg_lp[ch].extend([0.0]*W_spec)

    # 3) 可視化（6chを2x3で表示）
    plt.ion()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.ravel()
    imshows = [None]*num_of_channels
    cbs = [None]*num_of_channels  # colorbar handler
    plt.tight_layout()

    # 4) ループ
    count = 0
    stride_cnt = 0

    try:
        while True:
            # デバイス内のサンプリング可能数
            num_of_sampling = aio.getAiSamplingCount()
            if num_of_sampling <= 0:
                continue
            if num_of_sampling > buffer_size:
                num_of_sampling = buffer_size

            # サンプル読み出し（整数スケール）
            try:
                get_data = aio.getAiSamplingData(num_of_sampling)
                # ※ 전압값으로 바로 받고 싶으면:
                # get_data = aio.getAiSamplingDataEx(num_of_sampling)
                #   → 이후 스케일링(emg_arr 계산) 생략 가능
            except ValueError as ve:
                print(ve); continue

            smp_count, data_blk = get_data[0], get_data[1]
            if smp_count == 0:
                continue

            # チャンネル別に：元値→スケール→整流→LPF(causal)→STFT入力バッファ
            for ch in range(num_of_channels):
                chunk = data_blk[ch::num_of_channels]                 # 新規整数ブロック
                signal_buffer[ch].extend(chunk)                       # 既存スタイルのモニタ用

                raw_arr = np.asarray(chunk, dtype=np.float32)
                emg_arr = 10.0 - (65535.0 - raw_arr)/65535.0*20.0     # 기존 스케일식 그대로
                rect    = np.abs(emg_arr)                              # 절대값 정류
                y, zi_lpf[ch] = sosfilt(sos_lpf, rect, zi=zi_lpf[ch])  # 저역통과(실시간 인과)
                emg_lp[ch].extend(y.tolist())                          # STFT 입력 버퍼

            count     += smp_count
            stride_cnt += smp_count

            # STRIDEごとにスペクトログラム生成・表示・保存
            if stride_cnt >= STRIDE_spec and len(emg_lp[0]) >= W_spec:
                stride_cnt = 0

                # (a) (6,F,T) 3Dテンザ化：各chについて STFT → |Z|^2
                spec_list = []
                f2 = None; t2 = None  # extent용
                for ch in range(num_of_channels):
                    ybuf = np.asarray(emg_lp[ch], dtype=np.float32)   # 長さ W_spec
                    f2, t2, Z = stft(
                        ybuf,
                        fs=fs,
                        nperseg=STFT_NPERSEG,
                        noverlap=STFT_NOVERLAP,
                        boundary=None,
                        padded=False,
                        return_onesided=True
                    )
                    Sxx = (np.abs(Z)**2)                               # (F,T)
                    spec_list.append(Sxx)
                spec_tensor = np.stack(spec_list, axis=0)              # (6,F,T)

                # (b) 実時間表示（log1pで視認性向上, 논문풍 축 단위 추가）
                #     extent = [time_min, time_max, freq_min, freq_max]
                if t2 is None or f2 is None:
                    t2 = np.arange(spec_tensor.shape[2]) * (STFT_NPERSEG - STFT_NOVERLAP) / fs
                    f2 = np.linspace(0, fs/2, spec_tensor.shape[1], endpoint=True)

                extent = [t2[0], t2[-1], f2[0], f2[-1]]

                for ch in range(num_of_channels):
                    img_data = np.log1p(spec_tensor[ch])
                    if imshows[ch] is None:
                        imshows[ch] = axes[ch].imshow(
                            img_data,
                            origin='lower', aspect='auto', interpolation='nearest',
                            extent=extent
                        )
                        axes[ch].set_title(f"CH{ch}")
                        axes[ch].set_xlabel('time [s]')
                        axes[ch].set_ylabel('freq [Hz]')
                        # colorbar (논문 느낌)
                        cbs[ch] = plt.colorbar(imshows[ch], ax=axes[ch], fraction=0.046, pad=0.04)
                        cbs[ch].set_label('log power')
                    else:
                        imshows[ch].set_data(img_data)
                        imshows[ch].set_extent(extent)
                        # 컬러스케일 자동 업데이트(선택): 고정하고 싶으면 아래 두 줄 주석
                        imshows[ch].set_clim(vmin=np.min(img_data), vmax=np.max(img_data))
                        if cbs[ch] is not None:
                            cbs[ch].update_normal(imshows[ch])

                plt.pause(0.001)

                # (c) 任意で周期保存(.npz 텐서)
                if SAVE_SPEC_EVERY and ((count // STRIDE_spec) % SAVE_SPEC_EVERY == 0):
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    np.savez_compressed(
                        os.path.join(spec_save_dir, f"spec_{ts}.npz"),
                        spec=spec_tensor, fs=fs,
                        W=W_spec, STRIDE=STRIDE_spec,
                        STFT_NPERSEG=STFT_NPERSEG, STFT_NOVERLAP=STFT_NOVERLAP,
                        f=f2, t=t2
                    )

                # (d) 任意로 현재 그림 PNG 저장
                if SAVE_FIG_EVERY and ((count // STRIDE_spec) % SAVE_FIG_EVERY == 0):
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    fig.savefig(os.path.join(spec_fig_dir, f"spec_{ts}.png"), dpi=150)

            # 終了条件（デモ用）
            if count > limit_of_sampling:
                aio.stopAi()
                print("Sampling stop")
                break

    except KeyboardInterrupt:
        try:
            aio.stopAi()
        except:
            pass
        print("\n[INFO] stopped by user")
