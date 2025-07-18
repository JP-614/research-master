import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from collections import deque
import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd
import os
from base64 import b64encode
from glob import glob
from IPython.display import HTML
from IPython import display as ipy_display
import warnings
warnings.filterwarnings(action='ignore')

# ==============================================================================
# 1. 유틸리티 함수 및 클래스 정의
# ==============================================================================

#### 비디오 재생 함수
def show_video(mode='train', filename=None):
    mp4_list = glob(mode+'/*.mp4')
    if mp4_list:
        if filename :
            file_lists = glob(mode+'/'+filename)
            if not file_lists:
                print('No {} found'.format(filename))
                return -1
            mp4 = file_lists[0]
        else:
            mp4 = sorted(mp4_list)[-1]

        print("재생 비디오:", mp4)
        video = open(mp4, 'r+b').read()
        encoded = b64encode(video)
        ipy_display.display(HTML(data='''
            <video alt="gameplay" autoplay controls style="height: 400px;">
                <source src="data:video/mp4;base64,%s" type="video/mp4" />
            </video>
        ''' % (encoded.decode('ascii'))))
    else:
        print('No video found') 
        return -1

#### 조기 종료 클래스
class EarlyStopping_by_avg():
    def __init__(self, patience=10, verbose=0):
        super().__init__()
        self.best_avg = 0
        self.step = 0
        self.patience = patience
        self.verbose = verbose

    def check(self, avg, avg_scores):
        if avg >= self.best_avg:
            self.best_avg = avg
            self.step = 0
        elif len(avg_scores) > 1 and avg > avg_scores[-2]:
            self.step = 0
        else:
            self.step += 1
            if self.step > self.patience:
                if self.verbose:
                    print('조기 종료')
                return True
        return False

#### 사용자 정의 보상 함수
def get_reward(pos, angle, done):
    if done:
        return -100.0  # 실패 시 큰 음수 보상
    
    # 막대가 중앙에 가깝고, 각도가 작을수록 높은 보상
    reward = (1 - abs(pos) / 2.4) + (1 - abs(angle) / (12 * 2 * np.pi / 360))
    return reward

#### DQN 에이전트 클래스
class DQN():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.buffer_size = 2000
        self.buffer_size_train_start = 200
        self.buffer = deque(maxlen=self.buffer_size)
        self.loss_fn = MeanSquaredError()
        self.learning_rate = 0.001
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.batch_size = 32
        
        self.q_network = self.get_network()
        self.target_q_network = self.get_network()
        self.update_target_network()
        
        self.dir_name = os.getcwd()
        self.folder_checkpoint = os.path.join(self.dir_name, 'checkpoint')
        self.checkpoint = tf.train.Checkpoint(model=self.q_network, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.folder_checkpoint, max_to_keep=40)
        
    def update_target_network(self):
        self.target_q_network.set_weights(self.q_network.get_weights())

    def get_network(self):
        network = Sequential([
            Dense(24, activation='relu', input_shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(12, activation='relu'),
            Dense(self.action_size)
        ])
        return network

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def policy(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.q_network(state)
            return np.argmax(q_values[0])

    def train(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        states = np.array(states).reshape(self.batch_size, -1)
        next_states = np.array(next_states).reshape(self.batch_size, -1)

        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        q_next = self.target_q_network(next_states)
        max_q_next = tf.reduce_max(q_next, axis=1)
        targets = rewards + (self.gamma * max_q_next) * (1 - dones)

        with tf.GradientTape() as tape:
            q = self.q_network(states)
            one_hot_a = tf.one_hot(actions, self.action_size)
            q_sa = tf.reduce_sum(q * one_hot_a, axis=1)
            loss = self.loss_fn(targets, q_sa)
        
        grads = tape.gradient(loss, self.q_network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_weights))

# ==============================================================================
# 2. DQN 훈련 실행
# ==============================================================================

# CartPole 환경 정의 (비디오 녹화를 위해 render_mode="rgb_array" 설정)
ENV_NAME = 'CartPole-v1'
env = gym.make(ENV_NAME, render_mode="rgb_array")
# 매 에피소드 비디오 레코딩
env = RecordVideo(env, './train', episode_trigger=lambda episode_number: True)

# 환경의 상태와 행동 크기 정의
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# DQN 에이전트 정의
agent = DQN(state_size, action_size)

scores, avg_scores, episodes = [], [], []

# 반복 학습 에피소드 수 정의
num_episode = 200
early_stopping = EarlyStopping_by_avg(patience=10, verbose=1)
avg_score = 0

for epoch in range(num_episode):
    # 조기 종료 체크
    if early_stopping.check(avg_score, avg_scores):
        print("조기 종료가 실행되었습니다.")
        break

    # 환경 리셋 및 상태 초기화
    state, info = env.reset()
    done = False
    score = 0

    while not done:
        # 현재 상태에 대한 행동 결정
        action = agent.policy(state[np.newaxis, :])

        # 행동 실행 및 결과 획득
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 사용자 정의 보상 적용
        pos, _, angle, _ = next_state
        custom_reward = get_reward(pos, angle, done)
        
        # 리플레이 버퍼에 경험 저장
        agent.remember(state, action, custom_reward, next_state, done)

        # 다음 상태를 현재 상태로 업데이트
        state = next_state
        score += reward # 원본 보상을 점수로 기록

        # 버퍼가 충분히 쌓이면 학습 진행
        if len(agent.buffer) >= agent.buffer_size_train_start:
            agent.train()

        if done:
            # 타겟 네트워크 업데이트
            agent.update_target_network()

            scores.append(score)
            # 최근 10개 에피소드의 평균 점수 계산
            avg_score = np.mean(scores[-10:]) if scores else 0
            avg_scores.append(avg_score)
            episodes.append(epoch)
            
            print(f'episode: {epoch:3d} | score: {score:3.0f} | avg_score: {avg_score:6.2f} | buffer: {len(agent.buffer):4d} | epsilon: {agent.epsilon:.4f}')

env.close()

# 결과 그래프 출력 및 저장
plt.figure(figsize=(10, 6))
plt.title('DQN Training Results for CartPole-v1')
plt.xlabel('Episodes')
plt.ylabel('Average Score (last 10 episodes)', color='blue')
plt.plot(episodes, avg_scores, color='skyblue', marker='o', markerfacecolor='blue', markersize=4)
plt.tick_params(axis='y', labelcolor='blue')
plt.grid(True)
plt.savefig('cartpole_graph.png')
plt.show()

# 마지막으로 녹화된 비디오 재생
show_video()
