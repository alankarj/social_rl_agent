import random
import dialog_config


def get_prob_feedback_good(user_type, state):
    rapport_care = user_type['rapport_care']
    rapport_built = state['rapport_built']
    prob_feedback = dialog_config.prob_feedback
    total_recos = state['total_reco']
    history = state['prev_feedback']

    feedback_good = False

    if rapport_built and rapport_care:
        index = 0
    elif not rapport_built and rapport_care:
        index = 1
    elif rapport_built and not rapport_care:
        index = 2
    else:
        index = 3

    if history is None:
        p = prob_feedback[index]['R1']
    else:
        n = int(total_recos)
        print("Number of recos so far: ")
        # if n > 6:
        #     return False
        # if n == 0:
        #     return feedback_good
        if history:
            # if total_recos == 0:
            #     return False
            ind = 0
        else:
            # if total_recos == 0:
            #     return False
            ind = 1
        # print(ind)
        # print('R' + str(n))
        # print(index)
        p = prob_feedback[index]['R' + str(n)][ind]

    if p > random.random():
        feedback_good = True

    return feedback_good


def get_prob_accept_msg(user_type, state):
    phase = state['phase']
    rapport_built = state['rapport_built']
    rapport_care = user_type['rapport_care']
    prob_user_type = dialog_config.prob_user_type

    if rapport_built and rapport_care:
        index = 0
    elif not rapport_built and rapport_care:
        index = 1
    elif rapport_built and not rapport_care:
        index = 2
    else:
        index = 3

    accept_msg = False
    reco_type = phase.split('_recommendation')[0]

    if reco_type == 'food':
        return False

    p = prob_user_type['accept_tlink_' + reco_type][index]

    if p > random.random():
        accept_msg = True

    return accept_msg


def get_prob_another_reco(user_type, state):
    phase = state['phase']
    reco_type = phase.split('_recommendation')[0]
    another_reco = False

    if reco_type == 'food':
        return False

    if state[reco_type] < user_type['num_' + reco_type]:
        another_reco = True

    return another_reco
