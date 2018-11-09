import ac
import q_learning

MFAC = ac.MFAC
AC = ac.ActorCritic
IL = q_learning.DQN
MFQ = q_learning.MFQ


def spawn_ai(algo_name, sess, env, handle, human_name, max_steps):
    if algo_name == 'mfq':
        model = MFQ(sess, env, handle, human_name, max_steps, memory_size=80000)
    elif algo_name == 'mfac':
        model = MFAC(human_name, sess, env, handle)
    elif algo_name == 'ac':
        model = AC(human_name,sess,env, handle)
    elif algo_name == 'il':
        model = IL(sess, env, handle, human_name, max_steps, memory_size=80000)
    return model
