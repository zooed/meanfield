"""Self Play
"""

import os
import magent
import argparse
import numpy as np
import tensorflow as tf

import tools
from four_model import spawn_ai
from senario_battle import play



BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='ac' ,choices={'ac', 'mfac', 'mfq', 'il'}, help='选择算法')
    parser.add_argument('--n_round', type=int, default=10, help='设置训练局数')
    parser.add_argument('--max_steps', type=int, default=5, help='设置最大步数') 
    parser.add_argument('--map_size', type=int, default=40, help='设置地图的大小')  
    parser.add_argument('--update_every', type=int, default=5, help='设置Q学习更新间隔, optional')
    parser.add_argument('--save_every', type=int, default=10, help='设置self-play更新间隔')
    parser.add_argument('--render',default=True, help='渲染与否(if true, will render every save)')
    args = parser.parse_args()

    # 初始化环境
    env = magent.GridWorld('battle', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR, 'battle_model', 'build/render')) # 设置存储目录路径
    handles = env.get_handles()  #返回的是c_int类型的列表 作战双方的控制句柄 [c_int(0),c_int(1)]
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    log_dir = os.path.join(BASE_DIR,'data/tmp'.format(args.algo))
    model_dir = os.path.join(BASE_DIR, 'data/models/{}'.format(args.algo))

    if args.algo in ['mfq', 'mfac']: # 是否使用mean field方法
        use_mf = True
    else:
        use_mf = False

    start_from = 0

    sess = tf.Session(config=tf_config)
    # 实例化模型 Q网络与AC算法
    models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me', args.max_steps), 
              spawn_ai(args.algo, sess, env, handles[1], args.algo + '-opponent', args.max_steps)]

    sess.run(tf.global_variables_initializer())
    # ============ 网络参数的可视化 =================================
    mergerd = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/home/future/mfrl/Battle_model/logs",sess.graph)
     
    # 传入sess  model env play handles 初始化
    runner = tools.Runner(sess, 
                          env, 
                          handles, 
                          args.map_size, 
                          args.max_steps, 
                          models, 
                          play,
                          render_every=args.save_every if args.render else 0, 
                          save_every=args.save_every, 
                          tau=0.01, 
                          log_name=args.algo, 
                          log_dir=log_dir, 
                          model_dir=model_dir, 
                          train=True)

    # 每一局开始训练
    for k in range(start_from, start_from + args.n_round):
        #0.005 +(1-i/1400)*0.995 随i从1逐渐递减  直到i>1400 不再变化等于0.005
        eps = magent.utility.piecewise_decay(k, [0, 700, 1400], [1, 0.2, 0.05])
        runner.run(eps, k, use_mf)


"""
#============= 结构分析 ==============
总体main:
env = magent.GridWorld('battle', map_size=args.map_size)
handles = env.get_handles()
sess = tf.Session(config=tf_config)
model = model = AC(human_name,sess,env, handle) 算法模型类的实例化对象
runner = tools.Runner(sees, env, handles, models, play)  play是进行游戏的函数的引用
for k in range(episode):
  runner.run(eps,k,use_mf)  在其中调用 paly()方法



=================================
模型算法：
  值函数Critic
  策略函数Actor
class AC(object):
  self.view_space = env.get_view_space(handle)
  self.feature_space = env.get_feature_space(handle)
  self.num_actions = env.get_action_space(handle)[0]
  self._create_network(view_space, feature_space)
  
  def act():
    sess.run(policy) 
    策略函数 获取动作
  def _create_network():
    inputs = 
    policy = 
    value = 

  def train():
      执行训练网络的操作 输入view feature action reward 
        _, pg_loss, vf_loss, ent_loss, state_value = self.sess.run(
            [self.train_op, self.pg_loss, self.vf_loss, self.reg_loss, self.value], feed_dict={
                self.input_view: view,
                self.input_feature: feature,
                self.action: action,
                self.reward: reward,
            })
  def save(): 保存与加载模型
  def laod():

智能体与环境进行一步一个动作的交互
智能体执行动作 环境进行反馈更新
def play(env, n_round, map_size, max_steps, handles, models):
  env.reset()
  generate_map()
  while not done:
    act = model[i].act() 获取智能体的动作
  done = env.setp()环境进行反馈
  model[i].train()模型训练更新 梯度更新
  进行一局的每一步
  结束后进行AC算法的梯度更新

class Runner():
  def run():
    max_nums, nums, agent_r_records, total_rewards = self.play(env=self.env,
                                                                   n_round=iteration, 
                                                                   map_size=self.map_size,
                                                                   max_steps=self.max_steps, 
                                                                   handles=self.handles,
                                                                   models=self.models, 
                                                                   print_every=50, 
                                                                   eps=variant_eps, 
                                                                   render=(iteration + 1) % self.render_every if self.render_every > 0 else False, 
                                                                   use_mean=use_mean, 
                                                                   train=self.train)
  每一局调用一次play
  每一局结束之后 反馈最后的奖励结果
  更新模型                                                                 


"""       