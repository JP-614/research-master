import tensorflow as tf

import gym

max_steps_per_episode = 200

render_env = gym.make("CartPole-v1", render_mode='rgb_array')
from IPython import display as ipythondisplay
from PIL import Image

def render_episode(env: gym.Env, max_steps: int): 
  state, info = env.reset()
  state = tf.constant(state, dtype=tf.float32)
  screen = env.render()
  images = [Image.fromarray(screen)]
 
  for i in range(1, max_steps + 1):
    state = tf.expand_dims(state, 0)
    #     action_probs, _ = model(state)    
#     action = np.argmax(np.squeeze(action_probs))
    action = 1 # 오른쪽으로 가속
        
    state, reward, done, truncated, info = env.step(action)
    state = tf.constant(state, dtype=tf.float32)

    # Render screen every 10 steps
    if i % 1 == 0:
      screen = env.render()
      images.append(Image.fromarray(screen))
  
    if done:
        pass
#       break
  
  return images
images = render_episode(render_env, max_steps_per_episode)
image_file = 'cartpole-v1.gif'
# loop=0: loop forever, duration=1: play each frame for 1ms


images[0].save(
    image_file, save_all=True, append_images=images[1:], loop=0, duration=1)
import tensorflow_docs.vis.embed as embed
embed.embed_file(image_file)