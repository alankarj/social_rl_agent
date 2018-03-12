from user_simulator.my_rule_based_user_sim import RuleBasedUserSimulator
from agent.agent_dqn import AgentDQN
from dialog_arbiter.arbiter import DialogArbiter
import dialog_config

import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import json


def warm_start_simulation(num_epochs, test=False):
    """ Warm_Start Simulation (by Rule Policy) """
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    erp_full = False

    res = {}
    warm_start_run_epochs = 0

    for dialog in range(num_epochs):
        arbiter.initialize()
        dialog_over = False

        while not dialog_over:
            reward, dialog_over, state = arbiter.next(print_info=False)
            if agent.is_buffer_full(test):
                print("Breaking everything up.")
                erp_full = True
                break

        if erp_full:
            break

        cumulative_reward += reward

        if reward > 0:
            successes += 1
            print("Warm start simulation dialog %s: Success" % dialog)
            print("Reward: ", reward)

        else:
            print("Warm start simulation dialog %s: Fail" % dialog)
            print("Reward: ", reward)

        cumulative_turns += int(state['turn'] * 0.5 - 1)
        warm_start_run_epochs += 1

    agent.warm_start = 2
    res['success_rate'] = float(successes) / warm_start_run_epochs
    res['avg_reward'] = float(cumulative_reward) / warm_start_run_epochs
    res['avg_turns'] = float(cumulative_turns) / warm_start_run_epochs
    print("Warm start: %s epochs, success rate: %s, avg reward: %s, "
          "ave turns: %s" % (warm_start_run_epochs, res['success_rate'],
                             res['avg_reward'], res['avg_turns']))

    print("Current experience replay buffer size: %s" % (len(agent.erp_sl[
                                                                 'data'])))
    print("Current priority list: ", agent.erp_sl['priority'])


def run_dialog():
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    sl_losses = []
    test_losses = []
    rl_losses = []

    if warm_start == 1:
        print('Starting warm start...')
        warm_start_simulation(warm_start_epochs)

    if test == 1:
        print('Generating test set...')
        warm_start_simulation(test_epochs, test=True)

    print('Warm start finished, starting pre-training now...')

    if warm_start == 1:
        for i in range(num_iter_SL):
            beta = beta0 + (float(i) / num_iter_SL) * (1 - beta0)
            loss = agent.train(beta, lr)
            print("SL Iteration: {}, Mean loss: {:.5f} ".format(i, loss))
            sl_losses.append(loss)
            if i % tau == 0:
                agent.update_target_network()
                print("Target Network updated!")
            if i % save_intervals == 0:
                agent.save_model(sl_file_name)
                print("Model saved!")
        agent.close_session()
        print('Pre-training finished, starting RL training now...')
        print("Average SL training loss: ", np.average(sl_losses))
        plt.plot(sl_losses)
        plt.show()

    if test == 1:
        agent.close_session()
        agent.restore_session(mfile_sl)
        loss = agent.test_data()
        print("Test Iteration: {}, Mean loss: {:.5f} ".format(i, loss))
        test_losses.append(loss)
        print("Average SL test loss: ", np.average(test_losses))

    # agent.close_session()
    agent.restore_session(mfile_sl)
    for dialog in range(num_dialogs):
        print("Epoch: %s" % dialog)
        beta = beta0 + (float(dialog) / num_dialogs) * (1 - beta0)

        arbiter.initialize()
        dialog_over = False

        while not dialog_over:
            reward, dialog_over, state = arbiter.next(print_info=True)

        cumulative_reward += reward
        turn = int(state['turn'] * 0.5 - 1)
        cumulative_turns += turn

        if reward > 0:
            successes += 1

        loss = agent.train(beta, lr)
        print("RL Iteration: {}, Mean loss: {:.5f} ".format(dialog, loss))
        if dialog % tau == 0:
            agent.update_target_network()
            print("Target Network updated!")

        # if dialog % save_intervals == 0:
        #     agent.save_model(rl_file_name, sl=False)
        #     print("Model saved!")

        print("Progress: %s / %s, Success rate: %s / %s Avg reward: %.2f "
              "Avg turns: %.2f" % (dialog + 1, num_dialogs, successes,
                                   dialog + 1, float(cumulative_reward) /
                                   (dialog + 1), float(cumulative_turns)
                                   / (dialog + 1)))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Deep '
                                                 'Dialog Policy Optimization '
                                                 'through Demonstrations')
    parser.add_argument('--L', dest='L', type=float, default=0.8,
                        help='Loss value l(a,a_E) for a != a_E')
    parser.add_argument('--beta0', dest='beta0', type=float, default=0.6,
                        help='Starting value for exponent of importance '
                             'sampling weights')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int,
                        default=256, help='Number of nodes in hidden layers')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.9)
    parser.add_argument('--epsilon0_train', dest='epsilon0_train', type=float,
                        default=1.0)
    parser.add_argument('--epsilon0_test', dest='epsilon0_test', type=float,
                        default=0.05)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--num_iter_sl', dest='num_iter_sl', type=int,
                        default=100000)
    parser.add_argument('--targetq_pt', dest='targetq_pt', type=int,
                        default=100)
    parser.add_argument('--save_pt', dest='save_pt', type=int,
                        default=1000)
    parser.add_argument('--sl_file_name', dest='sl_file_name', type=str,
                        default='/pre-trained-model_sl')
    parser.add_argument('--rl_file_name', dest='rl_file_name', type=str,
                        default='/pre-trained-model_rl')
    parser.add_argument('--warm_start_epochs', dest='warm_start_epochs',
                        type=int, default=1000)
    parser.add_argument('--warm_start', dest='warm_start', type=int, default=1)
    parser.add_argument('--test', dest='test', type=int, default=0)
    parser.add_argument('--test_size', dest='test_size', type=int, default=200)
    parser.add_argument('--num_dialogs', dest='num_dialogs', type=int,
                        default=5000)
    parser.add_argument('--agt', dest='agt', type=int, default=1)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--erp_size_sl', dest='erp_size_sl', type=int,
                        default=10000)
    parser.add_argument('--erp_size_rl', dest='erp_size_rl', type=int,
                        default=100000)

    return parser.parse_args()


def main(args):
    args = parse_arguments()

    # RL parameters
    L = args.L
    beta0 = args.beta0
    gamma = args.gamma
    epsilon0_train = args.epsilon0_train
    epsilon0_test = args.epsilon0_test

    # Neural Net parameters
    hidden_dim = args.hidden_dim
    lr = args.lr

    # Training and testing parameters
    num_iter_sl = args.num_iter_sl
    targetq_pt = args.targetq_pt
    save_pt = targetq_pt
    sl_file_name = args.sl_file_name
    rl_file_name = args.rl_file_name
    mfile_sl = sl_file_name + '-' + str(save_pt)
    mfile_rl = rl_file_name + '-' + str(save_pt)
    warm_start_epochs = args.warm_start_epochs
    warm_start = args.warm_start
    test = args.test
    test_size = args.test_size
    num_dialogs = args.num_dialogs
    agt = args.agt
    batch_size = args.batch_size
    erp_size_sl = args.erp_size_sl
    erp_size_rl = args.erp_size_rl

    # Slot sets and act sets
    slot_set = dialog_config.slot_set
    phase_set = dialog_config.phase_set
    agent_act_set = dialog_config.sys_act_set
    user_act_set = dialog_config.user_act_set
    agent_cs_set = dialog_config.agent_cs_set
    user_cs_set = dialog_config.user_cs_set

    print("Slot set: ", json.dumps(slot_set, indent=2))
    print("Phase set: ", json.dumps(phase_set, indent=2))
    print("Agent act set: ", json.dumps(agent_act_set, indent=2))
    print("User act set: ", json.dumps(user_act_set, indent=2))
    print("Agent CS set: ", json.dumps(agent_cs_set, indent=2))
    print("User CS set: ", json.dumps(user_cs_set, indent=2))

    # User Parameters:
    param_user = {}
    pass
    param_user['user_goal_slots'] = dialog_config.user_goal_slots
    param_user['user_type_slots'] = dialog_config.user_type_slots
    param_user['prob_user_type'] = dialog_config.prob_user_type
    param_user['small_penalty'] = dialog_config.small_penalty
    param_user['large_penalty'] = dialog_config.large_penalty
    param_user['decision_points'] = dialog_config.decision_points
    param_user['prob_funcs'] = dialog_config.prob_funcs
    param_user['threshold_cs'] = dialog_config.threshold_cs
    param_user['count_slots'] = dialog_config.count_slots
    param_user['reward_slots'] = dialog_config.reward.keys()
    param_user['reward'] = dialog_config.reward
    param_user['max_turns'] = dialog_config.max_turns-2
    param_user['max_recos'] = dialog_config.max_recos
    param_user['constraint_violation_penalty'] = dialog_config.constraint_violation_penalty
    param_user['min_turns'] = dialog_config.min_turns

    # # Agent Parameters:
    # param_agent = {}
    # if agt == 1:
    #     param_agent['slot_set'] = slot_set
    #     param_agent['phase_set'] = phase_set
    #     param_agent['user_act_set'] = user_act_set
    #     param_agent['agent_act_set'] = agent_act_set
    #     param_agent['user_cs_set'] = user_cs_set
    #     param_agent['agent_cs_set'] = agent_cs_set
    #     param_agent['feasible_actions'] = dialog_config.feasible_actions
    #     param_agent['reward_slots'] = dialog_config.reward.keys()
    #     param_agent['ERP_size_SL'] = erp_size_sl
    #     param_agent['test_pool_size'] = test_pool_size
    #     param_agent['ERP_size_RL'] = erp_size_rl
    #     param_agent['hidden_dim'] = hidden_dim
    #     param_agent['gamma'] = gamma
    #     param_agent['warm_start'] = warm_start
    #     param_agent['epsilon'] = epsilon
    #     param_agent['trained_model_path'] = None
    #     param_agent['bool_slots'] = dialog_config.bool_slots
    #     param_agent['max_turns'] = dialog_config.max_turns
    #     param_agent['max_recos'] = dialog_config.max_recos
    #     param_agent['action_group_dict'] = dialog_config.action_group_dict
    #     param_agent['batch_size'] = batch_size
    #     param_agent['global_step'] = save_intervals
    #     param_agent['test'] = test
    #     param_agent['L'] = L
    #     agent = AgentDQN(param_agent)
    # else:
    #     agent = RuleBasedAgent()
    #
    # user = RuleBasedUserSimulator(param_user)
    # param_state_tracker = {}
    # pass
    # param_state_tracker['count_slots'] = dialog_config.count_slots
    # param_state_tracker['reward_slots'] = dialog_config.reward.keys()
    #
    # arbiter = DialogArbiter(user, agent, slot_set, param_state_tracker)
    # run_dialog()


if __name__ == "__main__":
    main(sys.argv)
