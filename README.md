# Research-Master: 動的状態でも頑丈な動作推定が可能な上腕筋電義手の制御手法の提案

修士課程の研究プロジェクトのコードリポジトリです。

本研究では、表面筋電位（sEMG）信号と深層強化学習（Deep Q-Network: DQN）を用いて、リアルタイムの手のジェスチャー認識システムを構築しています。

研究期間：2025年4月～現在

## 学習の流れ（Pipeline）
<img width="758" height="318" alt="Image" src="https://github.com/user-attachments/assets/b08f82b2-22f4-46be-a306-3a7faaf79bc5" />

## データの取得
チャンネル数：6

サンプリング周波数：1600Hz

計測時間：3秒

学習＆テストデータ：1セット収集

<img width="458" height="143" alt="image" src="https://github.com/user-attachments/assets/b81fe88f-6af4-44ac-9e07-83f6d11f188f" />

## データ前処理（Data Preprocessing）
生筋電位（sEMG）信号はノイズが多く、円滑に特徴を抽出する必要がある。そのままでは学習が困難なため、RL（Reinforcement Learning）環境に「状態（State）」として渡す前に、以下の厳密な前処理パイプラインを適用しています。
1. **スライディングウィンドウ分割:** 連続的な6チャンネルのsEMG信号を一定のウィンドウサイズ（幅: 300ポイント, ストライド: 40ポイント）で切り出します。これにより、リアルタイムな連続認識が可能になります。
2. **STFT（短時間フーリエ変換）:** 切り出した各ウィンドウに対してSTFT（nperseg: 24, noverlap: 12, Hamming窓）を適用し、1次元の時系列信号を2次元の時間-周波数領域（スペクトログラム）に変換します。
<img width="892" height="277" alt="image" src="https://github.com/user-attachments/assets/9fea4893-10ca-4d2b-8f15-1f2cc999b84f" />

3. **不要な周波数帯の除去＆ノイズ平滑化 (Smoothing):** 学習に不要な周波数帯を除去します。その後、スペクトログラムに対してガウシアンフィルター（$\sigma=1.5$）を適用し、突発的なノイズを軽減しつつ、ジェスチャー特有の周波数特徴を際立たせます。
<img width="833" height="281" alt="image" src="https://github.com/user-attachments/assets/13f70e9f-1616-4821-8ae2-b6a4217994dc" />

## CNN特徴抽出器 (CNN Feature Extractor)
前処理されたスペクトログラム（状態）から、DQNエージェントが最適な行動を決定するための有用な特徴ベクトルを抽出するため、独自設計のCNNアーキテクチャ（`cnn_ResNet.py`）を実装しています。（参考文献１）

この特徴抽出器は、Stable Baselines3の `BaseFeaturesExtractor` を継承して構築されており、以下の2つの高度なネットワーク構造を組み合わせています。

1. **Inception モジュール (Inception Module):**
   異なるカーネルサイズ（`1x1`, `3x3`, `5x5`）の畳み込み層（Convolutional Layer）と最大プーリング（Max Pooling）を並列に配置し、その出力を結合（Concatenation）する構造です。これにより、局所的な細かい周波数変化から、時間的な大局的パターンまで、多様なスケールの特徴を同時に抽出することができます。
2. **ResNet スキップ結合 (Skip Connection):**
   計6つのInceptionモジュールを直列に繋ぎ、ネットワークの途中に残差接続（Residual Connection）を導入しています。これにより、層を深くした際に生じる勾配消失（Vanishing Gradient）問題を防ぎ、DQNエージェントの安定かつ高速な学習を実現しています。
   抽出された高次元の特徴マップはフラット化され、最終的に128or256or518（最低化による）次元の特徴ベクトル（`features_dim=256`）としてDQNのQネットワーク（全結合層）へと入力されます。
<img width="801" height="419" alt="image" src="https://github.com/user-attachments/assets/1eb0b9ce-0a1c-472f-be43-938077e32812" />

## 深層強化学習エージェント (Deep Reinforcement Learning Agent)
CNN特徴抽出器から得られた高次元の特徴ベクトルを「状態（State）」として受け取り、最適な分類ポリシーを学習するために **DQN (Deep Q-Network)** アルゴリズムを採用しています。

本プロジェクトでは、時系列データの分類問題を「マルコフ決定過程（MDP）」として定式化し、カスタム環境（`emg_env_bu.py`）内で以下の設計に基づいてエージェントを訓練しています。

* **状態 (State):** CNN特徴抽出器から出力される256次元の特徴ベクトル。
* **行動 (Action):** エージェントは各ウィンドウに対して、7つの離散的なクラス（ジェスチャーの種類）から1つの行動を選択し、予測を行います。
* **報酬 (Reward):** モデルが連続的なデータに対して正確な分類を行うため、2段階の厳密な報酬関数を設計しています。
  1. **ステップ報酬:** 各ウィンドウの予測が実際のラベル（正解）と一致すれば $+1$、不一致なら $-1$ を与えます。
  2. **エピソード報酬:** 1エピソード終了時、全ウィンドウ中で最も多く予測されたジェスチャー（Mode）が正解と一致し、かつその割合（オーバーラップ係数）が70%を超えている場合、$+10$ の大きなボーナス報酬を与えます。

この報酬設計により、モデルは単一のウィンドウだけでなく、ノイズの多いシーケンシャルなデータ全体に対して安定した認識を行うように学習します。

## ファイル構成と役割（File Description）

本プロジェクトの学習・評価パイプラインは、以下の合計4つの主要ファイルで構成されています。(Research_masterに保存されています）

* **`test2.py` (データ取得・前処理)**
    CONTEC製の計測デバイス（AIO）を使用し、6チャンネルのリアルタイム筋電位データを取得します。3秒間のエピソードデータをサンプリングし、`.npz`形式で保存します。
* **`emg_env_bu.py` (強化学習環境の構築)**
    GymnasiumライブラリをベースにしたカスタムRL環境（Environment）です。収集したデータを読み込み、スペクトログラムを「状態（State）」としてエージェントに提供し、ジェスチャー分類の正誤に基づく「報酬（Reward）」を計算するロジックを実装しています。ここで、前処理
* **`cnn_ResNet_Relu.py` (CNN特徴抽出器)**
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
3. **モデルの最適化 (Hyperparameter Optimization) [オプション]**
   学習を本格的に開始する前に、`train.ipynb` を使用してベイズ最適化（Optuna）を実行します。これにより、DQNエージェントに最適なハイパーパラメータ（学習率、バッチサイズ、割引率など）を自動探索し、モデルの収束速度と認識精度を最大化します。
4. **モデルの学習 (Training)**
   `train_bu.py`（または `train.ipynb`）を実行してDQNエージェントの学習を開始します。
   ```bash
   python train_bu.py
5. **モデルの評価 (Evaluation)**
   学習済みモデル（　ex）`dqn_emg_final_model.zip`）を用いて、テストデータセットでの認識精度（Recognition Accuracy）を評価します。
## 今後の予定
追加学習の検証を行い、本研究の妥当性を検証します（現在進行中）。また、9月の日本ロボット学会の発表に向けて本研究を進めていく予定です。
## 参考文献 (References)
* P. J. Cruz, J. P. Vásconez, et al., "A Deep Q-Network based hand gesture recognition system for control of robotic platforms," Scientific Reports, 2023.
