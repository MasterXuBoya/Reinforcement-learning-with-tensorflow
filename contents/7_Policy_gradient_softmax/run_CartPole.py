"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""
'''
@Author MasterXu 
@Modify Time:2018/6/16
Observation，Action，Reward
1.gym can be only installed in linux and mac system,not for window now
2.程序从gym获取observation，负责决策action，然后将action传递给gym环境；
  类似从控制系统中获得速度、加速度等信号，然后进行控制算法，输出到控制系统中
  
  gym是一个环境，返回action后重新获得observation；类似于在模拟控制系统，CartPole内部应该有倒立摆的数学模型
    observation_, reward, done, info = env.step(action)
3.在CartPole倒立摆系统中，Observation是4维向量，表示位置，速度，杆的角度，角速度等
  Action存在两个状态0和1，表示向左和向后
  Reward：当倒立摆位于中心位置15°返回1，否则为0
4.神经网络输入是4维，隐层10维，输出是2维，对应Action
  _discount_and_norm_rewards专门计算某一个Action的长远奖励vt
  损失函数是vt*交叉熵（神经网络输出，此状态的Action）
  neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
  loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
  输出就是每一个Action对应的概率，最后按照对应概率输出
  如果Action是连续的，那么输出就是Action
'''
import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

for i_episode in range(3000):

    observation = env.reset()

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            if i_episode == 0:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
