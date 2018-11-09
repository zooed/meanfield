import os
import numpy as np
import tensorflow as tf

import tools

class ActorCritic:
    """
    AC模型
    调用的时候 每一步acts() 输出动作
             一局结束后 train()
    """
    def __init__(self, name, sess, env, handle,  value_coef=0.1, ent_coef=0.08, gamma=0.95, batch_size=64, learning_rate=1e-4):
        self.sess = sess
        self.env = env

        self.name = name
        self.view_space = env.get_view_space(handle)
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]
        self.gamma = gamma

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.value_coef = value_coef  # coefficient of value in the total loss
        self.ent_coef = ent_coef  # coefficient of entropy in the total loss

        # 初始化训练缓存
        self.view_buf = np.empty((1,) + self.view_space)
        self.feature_buf = np.empty((1,) + self.feature_space)
        self.action_buf = np.empty(1, dtype=np.int32)
        self.reward_buf = np.empty(1, dtype=np.float32)
        self.replay_buffer = tools.EpisodesBuffer()

        with tf.variable_scope(name):
            self.name_scope = tf.get_variable_scope().name
            self._create_network(self.view_space, self.feature_space)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)
    
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    # 输入状态S 输出要执行的动作 调用模型输出 而不是训练模型 
    # sess.run()运行模型 输入数据到占位符
    def act(self, **kwargs):
        action = self.sess.run(self.calc_action, {
            self.input_view: kwargs['state'][0],
            self.input_feature: kwargs['state'][1]
        })
        return action.astype(np.int32).reshape((-1,))

        
# 创建网络value--Critic评价  policy--Actor选取动作
    def _create_network(self, view_space, feature_space):
        # 状态S
        input_view = tf.placeholder(tf.float32, (None,) + view_space)
        input_feature = tf.placeholder(tf.float32, (None,) + feature_space)
        # 执行的动作
        action = tf.placeholder(tf.int32, [None])
        # 获取的奖励
        reward = tf.placeholder(tf.float32, [None])

        hidden_size = [256]
        flatten_view = tf.reshape(input_view, [-1, np.prod([v.value for v in input_view.shape[1:]])])
        h_view = tf.layers.dense(flatten_view, units=hidden_size[0], activation=tf.nn.relu)
        h_emb = tf.layers.dense(input_feature,  units=hidden_size[0], activation=tf.nn.relu)
        dense = tf.concat([h_view, h_emb], axis=1)
        dense = tf.layers.dense(dense, units=hidden_size[0] * 2, activation=tf.nn.relu)
        
        # 策略函数 输入S:view feature 使用S sess.run()输入S即可调用
        policy = tf.layers.dense(dense / 0.1, units=self.num_actions, activation=tf.nn.softmax)
        policy = tf.clip_by_value(policy, 1e-10, 1-1e-10)
        # 从策略函数 获取动作
        self.calc_action = tf.multinomial(tf.log(policy), 1)

        # 值函数 输入S:view feature
        value = tf.layers.dense(dense, units=1)
        value = tf.reshape(value, (-1,))

        # 优势函数
        advantage = tf.stop_gradient(reward - value)

        # 对数似然
        log_policy = tf.log(policy + 1e-6)
        action_mask = tf.one_hot(action, self.num_actions)
        log_prob = tf.reduce_sum(log_policy * action_mask, axis=1)

        # 策略梯度损失 + 值函数损失 + 负熵
        pg_loss = -tf.reduce_mean(advantage * log_prob)
        vf_loss = self.value_coef * tf.reduce_mean(tf.square(reward - value))
        neg_entropy = self.ent_coef * tf.reduce_mean(tf.reduce_sum(policy * log_policy, axis=1))
        total_loss = pg_loss + vf_loss + neg_entropy

        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        # 梯度下降训练两个网络 (clip gradient)修剪梯度
        # 因为损失函数的建立是
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(total_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))


        self.input_view = input_view
        self.input_feature = input_feature
        self.action = action
        self.reward = reward
        self.policy, self.value = policy, value
        self.train_op = train_op # ？？？多余
        self.pg_loss, self.vf_loss, self.reg_loss = pg_loss, vf_loss, neg_entropy
        self.total_loss = total_loss

# 训练AC网络 在每一局完成所有的步数之后进行训练AC网络
# 一局进行探索 获得数据进行训练
# 进行梯度下降更新参数 
# 使用经验重放
    def train(self):
        # calc buffer size
        n = 0
        batch_data = self.replay_buffer.episodes() # 字典 存储的是EpisodeBuffer对象：每一个智能体的一局的记录 长度为智能体的数量
        self.replay_buffer = tools.EpisodesBuffer() 
        # 无论进行几局都是智能体的数目 说明可能将所有局累加
        # 应该在每一局结束后清除数据 
        # 经验池重放  是累计一局所有智能体的所有局重放？ 还是每一局重放？
        # 由于在每一局结束后 进行训练的时候 重新创建了一个同名的对象 self.replay_buffer 之前的对象被取代 之前的对象的数据也就没有引用 进行了垃圾回收
        # 所以Buffer中保存的都是 这一局的所有agent的Buffer
        # 每一局结束之后 重新创建了一新的对象 之前的数据并没有保存
        # 一局得到的结果 agent_num X reward 其中每一个智能体的长度都不一样 最长的为最后结束时的剩余的agent的轨迹
        for episode in batch_data: # 每一个智能体的一局的记录长度累加 
            n += len(episode.rewards) # 所有一局中的所有智能体64个中的所有奖励的累计长度


        self.view_buf.resize((n,) + self.view_space)
        self.feature_buf.resize((n,) + self.feature_space)
        self.action_buf.resize(n)
        self.reward_buf.resize(n)
        view, feature = self.view_buf, self.feature_buf
        action, reward = self.action_buf, self.reward_buf

        ct = 0
        gamma = self.gamma
        j = 0
        # 从多局的离散buffer中收集一个连续的buffer缓存 ？ 从一局的多个智能体中累计结果？
        # 重点是为了获取V(S') 更新结果中的V 此时没有训练更新网络
        # 还要考虑多个智能体 每一个智能体都有一串序列
        for episode in batch_data: # 遍历所有智能体
            m = len(episode.rewards) # 一个智能体每一局执行的步数=获取的奖励
            # 训练几局 没有呈现除 逐渐累加的现象 说明 只使用了一局的数据 并不是逐局累加
            # 注意每一个智能体的轨迹长度可能并不相同
            v, f, a, r = episode.views, episode.features, episode.actions, episode.rewards
            """
            if len(v) == len(f):
                print("len(view)==len(feature)")
            if len(a) == len(r):
                print("一局动作的长度 等于 奖励的长度")
            正确！ 长度相等
            """

            r = np.array(r)
            # 假设一局一个智能体执行了s0 a0 r s1 a1 r s2 a2 r s3 a3 r s4
            # A的长度 == S的长度
            # R?
            keep = self.sess.run(self.value, feed_dict={self.input_view: [v[-1]], self.input_feature: [f[-1]],})[0]
            for i in reversed(range(m)): # 倒序索引 从每一个智能体的结束奖励开始向前更新R
                keep = keep * gamma + r[i]
                r[i] = keep

            # 更新每一局的每一个智能体的每一个奖励值
            # 是将一局的所有智能体的数据累计在一起
            reward[ct:ct + m] = r 
            view[ct:ct + m] = v
            feature[ct:ct + m] = f
            action[ct:ct + m] = a
            ct += m #到下一个智能体

        assert n == ct

        # 训练是将所有一局中所有的智能体的数据累加在一起 相当于一个智能体进行的决策 
        _, pg_loss, vf_loss, ent_loss, state_value = self.sess.run(
            [self.train_op, self.pg_loss, self.vf_loss, self.reg_loss, self.value], feed_dict={self.input_view: view,self.input_feature: feature,self.action: action,self.reward: reward,
            })


        print('[*] 策略梯度损失:', np.round(pg_loss, 6), '/ 值函数损失:', np.round(vf_loss, 6), '/ 熵损失:', np.round(ent_loss), '/ 值:', np.mean(state_value))


    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "ac-{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "ac-{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))











class MFAC:
    def __init__(self, name, sess, env, handle, value_coef=0.1, ent_coef=0.08, gamma=0.95, batch_size=64, learning_rate=1e-4):
        self.sess = sess
        self.env = env
        self.name = name

        self.view_space = env.get_view_space(handle)
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]
        self.reward_decay = gamma

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.value_coef = value_coef  # coefficient of value in the total loss
        self.ent_coef = ent_coef  # coefficient of entropy in the total loss

        # init training buffers
        self.view_buf = np.empty((1,) + self.view_space)
        self.feature_buf = np.empty((1,) + self.feature_space)
        self.action_buf = np.empty(1, dtype=np.int32)
        self.reward_buf = np.empty(1, dtype=np.float32)
        self.replay_buffer = tools.EpisodesBuffer(use_mean=True)

        with tf.variable_scope(name):
            self.name_scope = tf.get_variable_scope().name
            self._create_network(self.view_space, self.feature_space, )
    
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)
    
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def act(self, **kwargs):
        action = self.sess.run(self.calc_action, {
            self.input_view: kwargs['state'][0],
            self.input_feature: kwargs['state'][1]
        })
        return action.astype(np.int32).reshape((-1,))

    def _create_network(self, view_space, feature_space):
        # input
        input_view = tf.placeholder(tf.float32, (None,) + view_space)
        input_feature = tf.placeholder(tf.float32, (None,) + feature_space)
        input_act_prob = tf.placeholder(tf.float32, (None, self.num_actions))
        action = tf.placeholder(tf.int32, [None])

        reward = tf.placeholder(tf.float32, [None])

        hidden_size = [256]

        # fully connected
        flatten_view = tf.reshape(input_view, [-1, np.prod([v.value for v in input_view.shape[1:]])])
        h_view = tf.layers.dense(flatten_view, units=hidden_size[0], activation=tf.nn.relu)

        h_emb = tf.layers.dense(input_feature,  units=hidden_size[0], activation=tf.nn.relu)

        concat_layer = tf.concat([h_view, h_emb], axis=1)
        dense = tf.layers.dense(concat_layer, units=hidden_size[0] * 2, activation=tf.nn.relu)

        policy = tf.layers.dense(dense / 0.1, units=self.num_actions, activation=tf.nn.softmax)
        policy = tf.clip_by_value(policy, 1e-10, 1-1e-10)

        self.calc_action = tf.multinomial(tf.log(policy), 1)

        # for value obtain
        emb_prob = tf.layers.dense(input_act_prob, units=64, activation=tf.nn.relu)
        dense_prob = tf.layers.dense(emb_prob, units=32, activation=tf.nn.relu)
        concat_layer = tf.concat([concat_layer, dense_prob], axis=1)
        dense = tf.layers.dense(concat_layer, units=hidden_size[0], activation=tf.nn.relu)
        value = tf.layers.dense(dense, units=1)
        value = tf.reshape(value, (-1,))

        action_mask = tf.one_hot(action, self.num_actions)
        advantage = tf.stop_gradient(reward - value)

        log_policy = tf.log(policy + 1e-6)
        log_prob = tf.reduce_sum(log_policy * action_mask, axis=1)

        pg_loss = -tf.reduce_mean(advantage * log_prob)
        vf_loss = self.value_coef * tf.reduce_mean(tf.square(reward - value))
        neg_entropy = self.ent_coef * tf.reduce_mean(tf.reduce_sum(policy * log_policy, axis=1))
        total_loss = pg_loss + vf_loss + neg_entropy

        # train op (clip gradient)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(total_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        self.input_view = input_view
        self.input_feature = input_feature
        self.input_act_prob = input_act_prob
        self.action = action
        self.reward = reward

        self.policy, self.value = policy, value
        self.train_op = train_op
        self.pg_loss, self.vf_loss, self.reg_loss = pg_loss, vf_loss, neg_entropy
        self.total_loss = total_loss

    def train(self):
        # calc buffer size
        n = 0
        batch_data = self.replay_buffer.episodes()
        self.replay_buffer = tools.EpisodesBuffer(use_mean=True)

        for episode in batch_data:
            n += len(episode.rewards)

        self.view_buf.resize((n,) + self.view_space)
        self.feature_buf.resize((n,) + self.feature_space)
        self.action_buf.resize(n)
        self.reward_buf.resize(n)
        view, feature = self.view_buf, self.feature_buf
        action, reward = self.action_buf, self.reward_buf
        act_prob_buff = np.zeros((n, self.num_actions), dtype=np.float32)

        ct = 0
        gamma = self.reward_decay
        # collect episodes from multiple separate buffers to a continuous buffer
        for k, episode in enumerate(batch_data):
            v, f, a, r, prob = episode.views, episode.features, episode.actions, episode.rewards, episode.probs
            m = len(episode.rewards)

            assert len(prob) > 0 

            r = np.array(r)

            keep = self.sess.run(self.value, feed_dict={
                self.input_view: [v[-1]],
                self.input_feature: [f[-1]],
                self.input_act_prob: [prob[-1]]
            })[0]

            for i in reversed(range(m)):
                keep = keep * gamma + r[i]
                r[i] = keep

            view[ct:ct + m] = v
            feature[ct:ct + m] = f
            action[ct:ct + m] = a
            reward[ct:ct + m] = r
            act_prob_buff[ct:ct + m] = prob
            ct += m

        assert n == ct

        # train
        _, pg_loss, vf_loss, ent_loss, state_value = self.sess.run([self.train_op, self.pg_loss, self.vf_loss, self.reg_loss, self.value], 
                feed_dict={
                self.input_view: view,
                self.input_feature: feature,
                self.input_act_prob: act_prob_buff,
                self.action: action,
                self.reward: reward,
            })

        # print("sample", n, pg_loss, vf_loss, ent_loss)

        print('[*] PG_LOSS:', np.round(pg_loss, 6), '/ VF_LOSS:', np.round(vf_loss, 6), '/ ENT_LOSS:', np.round(ent_loss, 6), '/ VALUE:', np.mean(state_value))

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "mfac_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "mfac_{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))