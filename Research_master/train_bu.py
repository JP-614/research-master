from stable_baselines3 import DQN
from emg_env_bu import emg_env
from cnn_feature_extractor import CNN_feature

if __name__ =='__main__':
    # 1. 데이터 경로 설정
    TRAIN_DATA_DIR = 'リアルタイム/spec_tensor/'
    TEST_DATA_DIR = 'data/test/' # 필요 시 테스트 데이터 경로도 설정
    # 2. 학습용 환경 생성
    env = emg_env(data_dir=TRAIN_DATA_DIR, train=True)
    # 3. 사용자 정의 CNN을 위한 policy_kwargs 설정
    policy_kwargs = dict(
        features_extractor_class = CNN_feature,
        features_extractor_kwargs = dict(features_dim=256),   #최종 특징 벡터 크기
    )
    
 # 4. DQN 모델 생성
    model = DQN(
        'CnnPolicy',
        env,
        policy_kwargs=policy_kwargs,
        buffer_size=50000,       # 리플레이 버퍼 크기
        learning_starts=1000,    # 학습 시작 전 최소 경험 수
        batch_size=64,           # 미니배치 크기
        learning_rate=1e-4,      # 학습률
        gamma=0.99,              # 할인 계수
        tau=1.0,                 # 타겟 네트워크 업데이트 강도
        train_freq=4,            # 훈련 빈도
        gradient_steps=1,        # 그래디언트 업데이트 스텝
        verbose=1,
        tensorboard_log="./logs/dqn_emg_custom/"
    )
    
    # 5. 모델 학습 시작
    print("===== 学習開始 =====")
    model.learn(total_timesteps=100000, log_interval=4)
    model.save("dqn_emg_final_model")
    print("===== 学習完了およびモデル保存 =====")
    
    env.close()
    
    # (필요 시) 학습된 모델 테스트
    test_env = emg_env(data_dir=TEST_DATA_DIR, is_train=False)
    obs, info = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True) 
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        print(f"예측: {action}, 실제: {test_env.episode_label}, 보상: {reward}")
        test_env.close()