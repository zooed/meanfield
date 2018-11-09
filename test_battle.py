"""加载训练过 的模型进行训练
"""
import magent

import os
import argparse
import numpy as np
import tensorflow as tf

import tools
from four_model import spawn_ai
from senario_battle import battle   

import os
import magent
from magent.renderer import PyGameRenderer
from battle_server import MyBattleServer as Server



BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='mfac',choices={'ac', 'mfac', 'mfq', 'il'}, help='choose an algorithm from the preset')
    parser.add_argument('--oppo', type=str, default='mfac',choices={'ac', 'mfac', 'mfq', 'il'}, help='indicate the opponent model')
    parser.add_argument('--n_round', type=int, default=5, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--idx', nargs='*',default=[4,4])

    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('battle', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR, 'battle_model', 'build/render'))
    handles = env.get_handles()

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    main_model_dir = os.path.join(BASE_DIR, 'data/models/{}-0'.format(args.algo))
    oppo_model_dir = os.path.join(BASE_DIR, 'data/models/{}-1'.format(args.oppo))

    sess = tf.Session(config=tf_config)
    models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me', args.max_steps), 
    		  spawn_ai(args.oppo, sess, env, handles[1], args.oppo + '-opponent', args.max_steps)]
    sess.run(tf.global_variables_initializer())

    models[0].load(main_model_dir, step=args.idx[0])
    models[1].load(oppo_model_dir, step=args.idx[1])


    # 测试代码 加载训练的模型显示 作战细节
    PyGameRenderer().start(Server(models))


"""

    runner = tools.Runner(sess, env, handles, args.map_size, args.max_steps, models, battle, render_every=0)
    win_cnt = {'main': 0, 'opponent': 0}

    for k in range(0, args.n_round):
        runner.run(0.0, k, win_cnt=win_cnt)
    
    print('\n[*] >>> WIN_RATE: [{0}] {1} / [{2}] {3}'.format(args.algo, win_cnt['main'] / args.n_round, args.oppo, win_cnt['opponent'] / args.n_round))
"""c