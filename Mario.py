import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
from tensorflow.python.framework.ops import disable_eager_execution

from collections import deque

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from config import *

from moyo import DQNet
from processes import process_frame, stack_frame
from mem import Memory

disable_eager_execution()


class Agent:
    def __init__(self, level_name):
        self.level_name = level_name
        self.env = gym_super_mario_bros.make(level_name)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.possible_actions = np.array(np.identity(self.env.action_space.n, dtype=int).tolist())

        tf.compat.v1.reset_default_graph()

        self.DQNet = DQNet(state_size, action_size, learning_rate)
        self.memory = Memory(max_size=memory_size)
        self.stacked_frames = deque([np.zeros((100, 128), dtype=np.int) for i in range(stack_size)], maxlen=4)

        for i in range(pretrain_length):
            if i == 0:
                state = self.env.reset()
                state, self.stacked_frames = stack_frame(self.stacked_frames, state, True)

                choice = random.randint(1, len(self.possible_actions)) - 1
                action = self.possible_actions[choice]
                next_state, reward, done, _ = self.env.step(choice)

                next_state, self.stacked_frames = stack_frame(self.stacked_frames, next_state, False)

                if done:
                    next_state = np.zeros(state.shape)
                    self.memory.add((state, action, reward, next_state, done))
                    state = self.env.reset()
                    state, self.stacked_frames = stack_frame(self.stacked_frames, state, True)

                else:
                    self.memory.add((state, action, reward, next_state, done))
                    state = next_state

        self.saver = tf.compat.v1.train.Saver()
        self.writer = tf.compat.v1.summary.FileWriter("logs/")
        tf.summary.scalar("Loss", self.DQNet.loss)
        self.write_op = tf.compat.v1.summary.merge_all()

    def predict_action(self, sess, explore_start, explore_stop, decay_rate, decay_step, state, actions, ):
        exp_exp_tradeoff = np.random.rand()

        explore_probs = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        if explore_probs > exp_exp_tradeoff:
            choice = random.randint(1, len(self.possible_actions)) - 1
            action = self.possible_actions[choice]

        else:
            QS = sess.run(self.DQNet.output, feed_dict={self.DQNet.inputs: state.reshape((1, *state.shape))})
            choice = np.argmax(QS)
            action = self.possible_actions[choice]

        return action, choice, explore_probs

    def play_note(self):
        import matplotlib.pyplot as plt
        from JSAnimation.IPython_display import display_animation
        from matplotlib import animation
        from IPython.display import display

        def display_frame_gif(frames):
            patch = plt.imshow(frames[0])
            plt.axis('off')

            def animate(i):
                patch.set_data(frames[i])

            anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
            display(display_animation(anim, default_mode='loop'))

        frames = []
        with tf.compat.v1.Session as sess:
            total_test_rewards = []

            self.saver.restore(sess, "model/{0}.cpkt".format(self.level_name))

            for episode in range(1):
                total_rewards = 0
                state = self.env.reset()
                state, self.stacked_frames = stack_frame(self.stacked_frames, state, True)
                print("*************************************")
                print('EPISODE', episode)

                while True:
                    state = state.reshape((1, *state_size))
                    QS = sess.run(self.DQNet.output, feed_dict={self.DQNet.inputs: state})
                    choice = np.argmax(QS)
                    next_state, reward, done, _ = self.env.step(choice)
                    frames.append(self.env.render(mode='rgb_array'))

                    if done:
                        print("Score", total_rewards)
                        total_test_rewards.append(total_rewards)
                        break
                    next_state, self.stacked_frames = stack_frame(self.stacked_frames, next_state, False)
                    state = next_state
            self.env.close()

    def play(self):
        with tf.compat.v1.Session() as sess:
            total_test_rewards = []
            self.saver.restore(sess, "model/{0}.cpkt".format(self.level_name))
            for episode in range(1):
                total_rewards = 0
                state = self.env.reset()
                state, self.stacked_frames = stack_frame(self.stacked_frames, state, True)
                print("*************************************")
                print('EPISODE', episode)

                while True:
                    state = state.reshape((1, *state_size))
                    QS = sess.run(self.DQNet.output, feed_dict={self.DQNet.inputs: state})
                    choice = np.argmax(QS)
                    next_state, reward, done, _ = self.env.step(choice)
                    self.env.render()

                    total_rewards += reward

                    if done:
                        print("Score", total_rewards)
                        total_test_rewards.append(total_rewards)
                        break

                    next_state, self.stacked_frames = stack_frame(self.stacked_frames, next_state, False)
                    state = next_state
            self.env.close()

    def train(self):
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            decay_step = 0

            for episode in range(total_episodes):
                step = 0
                episodes_rewards = []
                state = self.env.reset()
                state, self.stacked_frames = stack_frame(self.stacked_frames, state, True)
                print("EPISODE", episode)

                while step < max_steps:
                    step += 1

                    decay_step += 1
                    action, choice, explore_probs = self.predict_action(sess,
                                                                        explore_start,
                                                                        explore_stop,
                                                                        decay_rate,
                                                                        decay_step,
                                                                        state,
                                                                        self.possible_actions)

                    next_state, reward, done, _ = self.env.step(choice)

                    if episode_render:
                        self.env.render()

                    episodes_rewards.append(reward)

                    if done:
                        print('done')

                        next_state = np.zeros((100, 128), dtype=np.int)

                        next_state, self.stacked_frames = stack_frame(self.stacked_frames, next_state, False)

                        step = max_steps

                        total_rewards = np.sum(episodes_rewards)

                        print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_rewards),
                              'Explore P: {:.4f}'.format(explore_probs),
                              'Training Loss {:.4f}'.format(loss))

                        self.memory.add((state, action, reward, next_state, done))

                    else:
                        next_state, self.stacked_frames = stack_frame(self.stacked_frames, next_state, False)
                        self.memory.add((state, action, reward, next_state, done))
                        state = next_state

                    batch = self.memory.sample(batch_size)
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch])
                    next_state_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.array([each[4] for each in batch])

                    target_Qs_batch = []

                    Qs_next_state = sess.run(self.DQNet.output, feed_dict={self.DQNet.inputs: next_state_mb})

                    for i in range(len(batch)):
                        terminal = dones_mb[i]

                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state)
                        target_Qs_batch.append(target)

                    target_mb = np.array([each for each in target_Qs_batch])

                    loss, _ = sess.run([self.DQNet.loss, self.DQNet.optimizer],
                                       feed_dict={self.DQNet.inputs: states_mb,
                                                  self.DQNet.target_q: target_mb,
                                                  self.DQNet.action: actions_mb})

                    summary = sess.run(self.write_op, feed_dict={self.DQNet.inputs: states_mb,
                                                                 self.DQNet.target_q: target_mb,
                                                                 self.DQNet.action: actions_mb})

                    self.writer.add_summary(summary, episode)
                    self.writer.flush()

                if episode % 5 == 0:
                    self.saver.save(sess, "models/{0}.cpkt".format(self.level_name))
                    print("model Saved")
