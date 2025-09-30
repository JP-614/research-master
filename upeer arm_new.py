#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import argparse
from collections import deque
import numpy as np
import serial
from scipy.signal import butter, sosfilt, sosfilt_zi, stft
import matplotlib.pyplot as plt
import os
from datetime import datetime
# -----------------------------
# 1) 하이퍼파라미터(논문 값)
# -----------------------------
FS = 200                 # 샘플레이트(Hz) - 아두이노와 일치
W = 300                  # 윈도우 길이(샘플) ≈ 1.5 s
STRIDE = 40              # 윈도우 간 이동(샘플) ≈ 0.2 s
LP_ORDER = 4             # 저역통과 차수
LP_CUTOFF = 15.0         # 저역통과 컷오프(Hz)
STFT_NPERSEG = 24        # 내부 STFT 창 길이(샘플) ≈ 0.12 s
STFT_HOP = 12            # 내부 STFT hop(샘플) ≈ 0.06 s
STFT_NOVERLAP = STFT_NPERSEG - STFT_HOP
N_CHANNELS = 6

# =========================
# 1) 저역통과 필터(SOS)
# =========================
def design_envelope_lpf(fs=FS, cutoff=LP_CUTOFF, order=LP_ORDER):
    wn = cutoff / (0.5 * fs)    #正規化cutoff
    sos = butter(order, wn, btype='low', output='sos')  #filter係数計算
    return sos
SOS_LP = design_envelope_lpf()
ZI = [sosfilt_zi(SOS_LP) for _ in range(N_CHANNELS)]    #チャンネル別filterの状態

# =========================
# 2) 스펙트로그램 (입력은 "이미 필터된" 신호)
# =========================
def channel_spectrogram(x_filtered_1d):
    f, t, Zxx = stft(
        x_filtered_1d,
        fs=FS,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        boundary=None,
        padded=False,
        return_onesided=True
    )
    return np.abs(Zxx) ** 2 # (F, T)

def window_to_tensor(window_2d_filtered):
    # window_2d_filtered: (W, 6)
    specs = []
    for ch in range(N_CHANNELS):
        s = channel_spectrogram(window_2d_filtered[:, ch])  # (F, T)
        specs.append(s[np.newaxis, ...])
    return np.concatenate(specs, axis=0)    #(6, F, T)

# =========================
# 3) 실시간 루프
# =========================    
def run(port, baud, save_dir, record_seconds):
    os.makedirs(save_dir, exist_ok=True)
    
    #filter出力buffer(sliding, W)
    bufs_filt = [deque(maxlen=W) for _ in range(N_CHANNELS)]
    since_last = 0
    
    #plot準備
    fig, axes, imshow = None, None, None
    
    with serial.Serial(port, baudrate=baud, timeout=1) as ser:
        print(f"[INFO] Serial opne: {port} @ {baud}")
        print("[INFO] interation : if you write the label(example : 0,1,2..., ),", "collected and reserved the data with ",record_seconds, "second ")
        print("[INFO] 'show'is processing, 'end' is ending")
        
        #初期データを廃棄
        for _ in range(50):
            try:
                ser.readline()
            except:
                pass
        last_tensor = None
        recording = False
        rec_label = None
        rec_deadline = 0.0  # 記録終了時刻
        
        while True:
            # ㅡㅡㅡㅡㅡ 사용자 입력(논블록 흉내) ㅡㅡㅡㅡㅡ
            # 터미널에 무엇인가 입력되어 있으면 처리
            if sys.stdin in select_readable():
                cmd = sys.stdin.readline().strip()
                if cmd.lower() == 'end':
                    print("[INFO] 終了します．")
                    break
                elif cmd.lower() == 'show':
                    print("[INFO] リアルタイム画面のみ表示します．")
                elif cmd.isdigit():
                    rec_label = int(cmd)
                    rec_deadline = time.time() + float(record_seconds)
                    recording = True
                    print(f"[REC] rabel={rec_label}, {record_seconds}秒 収集する")
                else:
                    print("[INFO] operation : rabel/ show/ end")
            
            # ㅡㅡㅡㅡㅡ 시리얼에서 한 줄 읽기 ㅡㅡㅡㅡㅡ
            line = ser.readline().decode('ascii', errors='ignore').strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',') #들어온 문자열을 쉼표를 찾아서 자름
            if len(parts) != 1 + N_CHANNELS:
                continue
            
            try:
                vals = [int(v) for v in parts[1:]]  #6 channel
            except ValueError:
                continue
            
            # ㅡㅡㅡㅡㅡ 정류→causal 저역통과→버퍼 ㅡㅡㅡㅡㅡ
            for ch in range(N_CHANNELS):
                rect = abs(vals[ch])    # 정류
                y, ZI[ch] = sosfilt(SOS_LP, [rect], zi=ZI[ch])  # 저역통과(상태 유지)
                bufs_filt[ch].append(float(y[0]))   # 필터 출력 저장
                
             # ㅡㅡㅡㅡㅡ STRIDE마다 스펙트로그램 계산/표시 ㅡㅡㅡㅡㅡ 
            since_last += 1
            if len(bufs_filt[0]) == W and since_last >= STRIDE:
                since_last = 0
                win_filt = np.stack(
                    [np.array(bufs_filt[ch], dtype=np.float32) for ch in range(N_CHANNELS)],
                    axis=1
                )   # (W, 6)
                spec_tensor = window_to_tensor(win_filt)    #(6, F, T)
                last_tensor = spec_tensor
                
                # (A) 실시간 시각화 
                if fig is None:
                    fig, axes = plt.subplots(2, 3, figsize=(10,6))
                    axes = axes.ravel   #２次元配列を１次元に置き換える＝＞二重ループが要らなくなる
                    imshows = []
                    for i in range(6):
                        im = axes[i].imshow(
                            np.log1p(spec_tensor[i]),
                            origin='lower', aspect='auto', interpolation='nearest'
                        )
                        axes[i].set_title(f"CH{i}")
                        axes[i].set_xlabel("Time bins")
                        axes[i].set_ylabel("Freq bins")
                        imshows.append(im)
                        plt.tight_layout
                        plt.pause(0.001)
                else:
                    for i in range(6):
                        imshows[i].set_data(np.log1p(spec_tensor[i]))
                        plt.pause(0.001)
                        
                 # (B) 레코딩 모드면 저장
                if recording:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path = os.path.join(save_dir, f"spec_{ts}_label{rec_label}.npz")
                    # spec: (6,F,T), fs:200, meta: stride 등도 같이 저장
                    np.savez_compressed(
                        path,
                        spec=spec_tensor, fs=FS,
                        W=W, STRIDE=STRIDE,
                        STFT_NPERSEG=STFT_NPERSEG, STFT_HOP=STFT_HOP
                    )
                    print(f"  saved -> {os.path.basename(path)}")

                    # 시간 다 되면 레코딩 종료
                    if time.time() >= rec_deadline:
                        print(f"[REC] 라벨={rec_label} 수집 종료")
                        recording = False
                        rec_label = None

# 표준입력에 읽을 게 있는지 확인(간단한 셀렉트)
import select
def select_readable(timeout=0):
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return r
# =========================
# 4) 엔트리 포인트
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="예: Windows 'COM6', Linux '/dev/ttyACM0'")
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--save_dir", default="record_npz", help="스펙트로그램 저장 폴더")
    parser.add_argument("--record_seconds", type=float, default=3.0, help="라벨 1회당 수집 시간(초)")
    args = parser.parse_args()

    try:
        run(args.port, args.baud, args.save_dir, args.record_seconds)
    except KeyboardInterrupt:
        print("\n[INFO] stopped by user")