# train.py (최종 클린 버전)
from stable_baselines3 import DQN
from emg_env import emg_env
from cnn_feature_extractor import CNN_feature

if __name__ =='__main__':
    # 1. 데이터 경로 설정
    TRAIN_DATA_DIR = 'リアルタイム/spec_tensor/' # 실제 NPZ 파일이 있는 경로
    
    # 2. 학습용 환경 생성
    env = emg_env(data_dir=TRAIN_DATA_DIR, train=True)

    # 3. 사용자 정의 CNN을 위한 policy_kwargs 설정
    policy_kwargs = dict(
        features_extractor_class=CNN_feature,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    # 4. DQN 모델 생성
    model = DQN(
        policy='CnnPolicy',
        env=env,
        policy_kwargs=policy_kwargs,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        learning_rate=1e-4,
        gamma=0.99,
        tau=1.0,
        train_freq=4,
        gradient_steps=1,
        verbose=1, 
        tensorboard_log="./logs/dqn_emg_custom/"
    )
    
    # 5. 모델 학습 시작
    print("===== 학습 시작 =====")
    model.learn(total_timesteps=100000, log_interval=4)
    model.save("dqn_emg_final_model")
    print("===== 학습 완료 및 모델 저장 =====")
    
    env.close()