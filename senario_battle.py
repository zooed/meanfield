import random
import math
import numpy as np

# 可视化
import magent
from magent.renderer import PyGameRenderer
from battle_server import MyBattleServer as Server

def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 3

    leftID = random.randint(0, 1)
    rightID = 1 - leftID

    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)


def play(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, use_mean=False, train=False):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles) # 作战双方
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    # 一局一局的开始训练 在run函数中每次被调用一次就会执行一局 包含许多步
    print("\n\n[*] Round #{0}, Eps: {1:.2f} Number: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    if use_mean:
        # 将每一方的动作空间的动作取值概率大小 决定选取动作的概率
        former_act_prob = [np.zeros( (1, env.get_action_space(handles[0])[0]) ), 
                           np.zeros( (1, env.get_action_space(handles[1])[0]) )]
   
    # 一步一步执行动作
    while not done and step_ct < max_steps:
        
        # take actions for every model n_group = 2
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])

        if use_mean:
            for i in range(n_group):
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
                # np.tile重复扩展数组 np.tile(A,(m,n)) 将数组A看作单独一个元素，将其重复扩充为mxn的数组，每个元素为A 
                # 如果A的大小为axb 则最后的数组大小为 maxnb
                acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)
        else:
            for i in range(n_group):
                # 调用策略函数获取动作
                acts[i] = models[i].act(state=state[i], eps=eps)

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # 双发各执行一步之后 更新
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])
        
        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }
        
        if use_mean:
            buffer['prob'] = former_act_prob[0]
            # 保存选择动作的历史概率
            for i in range(n_group):
                # 双方每一步所有智能体之后 更新动作概率
                former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)

        # 保存每一步的数据
        if train:
            models[0].flush_buffer(**buffer)

        # agent 数量 
        nums = [env.get_num(handle) for handle in handles]
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "Num": nums}

        step_ct += 1 # 一局累计的步数

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    # 一局执行结束之后 进行AC梯度训练 一局一局的更新AC 策略与价值网络
    if train:
        models[0].train()
    # 一局之后 统计平均奖励
    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards


def battle(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, use_mean=None, train=False):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))
    
    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards