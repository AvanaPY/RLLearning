import imageio
import os 
import numpy as np

def compute_avg_return(env, policy, num_episodes=10):
    returns = []
    for i in range(num_episodes):
        time_step = env.reset()
        episode_return = 0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward
        returns.append(episode_return.numpy()[0])

    return returns

def create_policy_eval_video(policy, filename, env, py_env, 
                            num_episodes=5, 
                            fps=30,
                            append_score:bool=False):

    frame_buffer = []
    rewards = []
    for episode in range(num_episodes):
        ts = env.reset()
        frame_buffer.append(py_env.render())
        frames = 1
        while not ts.is_last():
            action_step = policy.action(ts)
            ts = env.step(action_step.action)
            frame_buffer.append(py_env.render())
            rewards.append(ts.reward.numpy()[0])
            frames += 1
            print(f'\rEpisode {episode}: Making frame {frames}...', end='')
        print(f'\rEpisode {episode}: Lasted {frames} frames')
    sum_reward = np.sum(np.array(rewards))

    if append_score:
        filename = f'{filename}{sum_reward:6.2f}'
    filename = filename + ".mp4"
    
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with imageio.get_writer(filename, fps=fps) as video:
        for frame in frame_buffer:
            video.append_data(frame)
    print(f'Generated {filename}: {len(frame_buffer)} frames, {len(frame_buffer) / fps:.2f}s')