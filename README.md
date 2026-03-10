# Research-Master: 動的状態でも頑丈な動作推定が可能な上腕筋電義手の制御手法の提案

修士課程の研究プロジェクトのコードリポジトリです。

本研究では、表面筋電位（sEMG）信号と深層強化学習（Deep Q-Network: DQN）を用いて、リアルタイムの手のジェスチャー認識システムを構築しています。

研究期間：2025年4月～現在

## 学習の流れ（Pipeline）
<img width="758" height="318" alt="Image" src="https://github.com/user-attachments/assets/b08f82b2-22f4-46be-a306-3a7faaf79bc5" />

## ファイル構成と役割（File Description）

本プロジェクトの学習・評価パイプラインは、以下の合計4つの主要ファイルで構成されています。(Research_masterに保存されています）

* **`test2.py` (データ取得・前処理)**
    CONTEC製の計測デバイス（AIO）を使用し、6チャンネルのリアルタイム筋電位データを取得します。3秒間のエピソードデータをサンプリングし、STFT（短時間フーリエ変換）処理を行ってスペクトログラムを生成し、`.npz`形式で保存します。
* **`emg_env_bu.py` (強化学習環境の構築)**
    GymnasiumライブラリをベースにしたカスタムRL環境（Environment）です。収集したデータを読み込み、スペクトログラムを「状態（State）」としてエージェントに提供し、ジェスチャー分類の正誤に基づく「報酬（Reward）」を計算するロジックを実装しています。
* **`cnn_ResNet.py` (CNN特徴抽出器)**
    状態（スペクトログラム）から有用な特徴を抽出するためのカスタムニューラルネットワークです。InceptionモジュールとResNetのスキップ結合（Skip Connection）を組み合わせており、DQNのPolicy Networkとして機能します。
* **`train.ipynb` (ベイズ最適化・学習・評価)**
    Stable Baselines3を用いてDQNエージェントの学習を行うメインノートブックです。ベイズ最適化（Bayesian Optimization）を用いたハイパーパラメータチューニングや、学習済みモデルの評価を記述しています。

## 開発環境 (Requirements)
本プロジェクトを実行するために必要な主なライブラリとバージョンです。

* Python 3.13
* PyTorch
* Stable-Baselines3
* Gymnasium
* Librosa
* SciPy
* CONTEC API-USBP Library (`AIO.py`)

## 実行方法 (How to Run)
システムを動かすための具体的な手順です。

1. **データ収集 (Data Collection)**
   `test2.py` を実行し、CONTECデバイスを通じてユーザーのジェスチャーデータを取得します。
   ```bash
   python test2.py
2. **強化学習環境のセットアップ (Environment Setup)**
   データ収集後、カスタム環境スクリプト `emg_env_bu.py` がデータを読み込み、学習用の環境を構築します。

3. **モデルの学習 (Training)**
   `train_bu.py`（または `train.ipynb`）を実行してDQNエージェントの学習を開始します。
   ```bash
   python train_bu.py
4. **モデルの評価 (Evaluation)**
   学習済みモデル（　ex）`dqn_emg_final_model.zip`）を用いて、テストデータセットでの認識精度（Recognition Accuracy）を評価します。

## 参考文献 (References)
* P. J. Cruz, J. P. Vásconez, et al., "A Deep Q-Network based hand gesture recognition system for control of robotic platforms," Scientific Reports, 2023.
