from sequence_classifier import classifiers
import dict_reader
import os

data_path = os.path.abspath('./') + "/data/"
# All slots
slot_set_path = data_path + "slot_set"
slot_set = dict_reader.text_to_dict(slot_set_path)

# All phases
phase_set_path = data_path + "phase_set"
phase_set = dict_reader.text_to_dict(phase_set_path)

# List of system acts
sys_act_set_path = data_path + "sys_act_set"
sys_act_set = dict_reader.text_to_dict(sys_act_set_path)

# List of user acts
user_act_set_path = data_path + "user_act_set"
user_act_set = dict_reader.text_to_dict(user_act_set_path)

# List of agent CS
agent_cs_set_path = data_path + "agent_cs_set"
agent_cs_set = dict_reader.text_to_dict(agent_cs_set_path)

# List of user acts
user_cs_set_path = data_path + "user_cs_set"
user_cs_set = dict_reader.text_to_dict(user_cs_set_path)

# Reward and turn length derived from data
true_reward_path = data_path + "true_reward.csv"
true_turn_path = data_path + "true_turn.csv"

# Dialog rewards
reward = {}
pass
reward['feedback'] = 5
reward['send_msg_tlink'] = 10

# Count tracking
count_slots = ['session', 'person', 'food']

# Boolean slots
bool_slots = {'primary_goal': 0, 'goal_session': 1, 'goal_person': 2,
              'goal_food': 3, 'feedback': 4, 'send_msg_tlink': 5,
              'another_reco': 6}

max_turns = 60  # Maximum agent turns allowed
max_recos = 5
constraint_violation_penalty = -10
min_turns = 8  # Minimum agent turns required to build rapport with a user

# Cons

# User Goal slots
user_goal_slots = {}
pass
user_goal_slots['interest'] = "X"  # Just a placeholder
user_goal_slots['goal_food'] = [True, False]  # Possible values of this slot
user_goal_slots['goal_person'] = [True, False]
user_goal_slots['goal_session'] = [True, False]

# User Type slots
user_type_slots = {}
pass
user_type_slots['first_time'] = [True, False]
user_type_slots['met_before'] = [True, False]
user_type_slots['num_reco'] = [1, 2, 3, 4, 5, 6]
user_type_slots['num_reco_person'] = [0, 1, 2, 3, 4, 5, 6]
user_type_slots['num_reco_session'] = [0, 1, 2, 3, 4, 5, 6]
user_type_slots['primary_goal'] = ['goal_session', 'goal_person', 'goal_food']
# Less Time (LT): 1-2 recos, Enough Time (ET): 3-4, More Time (MT): 5-6
user_type_slots['time'] = ['LT', 'ET', 'MT']
# Whether or not the user cares about building rapport
user_type_slots['rapport_care'] = [True, False]
# Type-1: None, Type-2: SD-QESD Type, Type-3: SD-PR Type, Type-4: PR-HE Type
user_type_slots['cs_type'] = [1, 2, 3, 4]

# Probability dict (for random user type initialization)
prob_user_type = {}
pass
prob_user_type['first_time'] = 0.5
prob_user_type['met_before'] = 0.0

# Probability that user cares about building rapport indexed by the number of
# recommendations the user is going to ask for (1 to 6).
prob_user_type['rapport_care'] = [0.25, 0.5, 0.5, 0.58, 0.44, 0.5]
# prob_user_type['rapport_care'] = [1, 1, 1, 1, 1, 1]
# prob_user_type['rapport_care'] = [0, 0, 0, 0, 0, 0]

# Probability distribution for total number of recos (1 to 6)
prob_user_type['num_reco'] = [0.17, 0.23, 0.26, 0.17, 0.13, 0.04]
# Probability distribution for total number of person recos (0 to num_reco)
prob_user_type['num_reco_1'] = [0.5, 0.5]
prob_user_type['num_reco_2'] = [0.125, 0.1875, 0.6875]
prob_user_type['num_reco_3'] = [0.17, 0.33, 0.28, 0.22]
prob_user_type['num_reco_4'] = [0.08, 0.08, 0.5, 0.25, 0.09]
prob_user_type['num_reco_5'] = [0, 0, 0.11, 0.67, 0.22, 0]
prob_user_type['num_reco_6'] = [0, 0, 0, 0, 1, 0, 0]

# Probability distribution for user CS personality/types
prob_user_type['cs_type'] = [0.25, 0.25, 0.25, 0.25]
# prob_user_type['cs_type'] = [0.7, 0.1, 0.1, 0.1]
# prob_user_type['cs_type'] = [0, 0, 0, 0]

# Probability distribution for agent CS
prob_agent_cs = [0.2, 0.2, 0.2, 0.2, 0.2]
agent_cs = ['None', 'SD', 'QESD', 'PR', 'HE']

# Agent CS for each user type should be above these thresholds for rapport to
# be built
# threshold_cs = [0.5, 0.5, 0.5, 0.5]
# threshold_cs = [1.0, 1.0, 1.0, 1.0]
threshold_cs = [0.4, 0.4, 0.4, 0.4]
# threshold_cs = [0, 0, 0, 0]

# Probability of acceptance for 4 cases:
# 0: Rapport built, care
# 1: Rapport not built, care
# 2: Rapport built, not care
# 3: Rapport not built, not care
prob_user_type['accept_tlink_person'] = [0.8, 0.625, 0.25, 0.74]
prob_user_type['accept_tlink_session'] = [1, 0.8, 0.67, 0.89]

# prob_feedback = {
#     0: {
#         'R1': 0.63,
#         'R2': [1, 1], # Entry-1 is P(R2|R1), Entry-2 is P(R2|~R1)
#         'R3': [0.83, 0],
#         'R4': [1, 0],
#         'R5': [0, 0],
#         'R6': [0, 0]
#     },
#     1: {
#         'R1': 0.32,
#         'R2': [0.25, 0.43],  # Entry-1 is P(R2|R1), Entry-2 is P(R2|~R1)
#         'R3': [0.14, 0.5],
#         'R4': [0, 0],
#         'R5': [0, 0.4],
#         'R6': [0, 0]
#     },
#     2: {
#         'R1': 0.22,
#         'R2': [0, 0.8],  # Entry-1 is P(R2|R1), Entry-2 is P(R2|~R1)
#         'R3': [0, 1],
#         'R4': [0, 0],
#         'R5': [0, 0],
#         'R6': [0, 0]
#     },
#     3: {
#         'R1': 0.83,
#         'R2': [0.82, 1],  # Entry-1 is P(R2|R1), Entry-2 is P(R2|~R1)
#         'R3': [0.73, 1],
#         'R4': [0.7, 1],
#         'R5': [0.8, 1],
#         'R6': [0, 0]
#     }
# }

prob_feedback = {
    0: {
        'R1': 0.63,
        'R2': [0.92, 1], # Entry-1 is P(R2|R1), Entry-2 is P(R2|~R1)
        'R3': [0.92, 1],
        'R4': [0.92, 1],
        'R5': [0.92, 1],
        'R6': [0.92, 1]
    },
    1: {
        'R1': 0.32,
        'R2': [0.17, 0.29],  # Entry-1 is P(R2|R1), Entry-2 is P(R2|~R1)
        'R3': [0.17, 0.29],
        'R4': [0.17, 0.29],
        'R5': [0.17, 0.29],
        'R6': [0.17, 0.29]
    },
    2: {
        'R1': 0.22,
        'R2': [0, 0.71],  # Entry-1 is P(R2|R1), Entry-2 is P(R2|~R1)
        'R3': [0, 0.71],
        'R4': [0, 0.71],
        'R5': [0, 0.71],
        'R6': [0, 0.71]
    },
    3: {
        'R1': 0.83,
        'R2': [0.74, 1],  # Entry-1 is P(R2|R1), Entry-2 is P(R2|~R1)
        'R3': [0.74, 1],
        'R4': [0.74, 1],
        'R5': [0.74, 1],
        'R6': [0.74, 1]
    }
}

# Decision points
decision_points = ['feedback', 'send_msg_tlink', 'another_reco']

# # Probability thresholds for decision points
# threshold = {}
# pass
# threshold['feedback'] = 0.5
# threshold['send_msg_tlink'] = 0.5
# threshold['another_reco'] = 0.5

# Probability functions for decision points
prob_funcs = {}
pass
prob_funcs['feedback'] = classifiers.get_prob_feedback_good
prob_funcs['send_msg_tlink'] = classifiers.get_prob_accept_msg
prob_funcs['another_reco'] = classifiers.get_prob_another_reco

# Penalty per turn
small_penalty = -0.25
large_penalty = -1

# print(prob_funcs)


################################################################################
################################################################################
# Config parameters for RL-based agent
slot_set_list = list(slot_set.keys())
phase_list = list(phase_set.keys())
sys_act_list = list(sys_act_set.keys())

sys_request_slots = slot_set_list[0:11]  # First 11 are request slots
sys_inform_slots = slot_set_list[-3:]   # Last 3 are inform slots

unique_phase_acts = sys_act_list[0:4]
unique_phases = phase_list[0:4]
non_req_inform_acts = sys_act_list[4:9]  # Last 2 are request, inform

unique_phase_req_slots = slot_set_list[0:8]
ambiguous_phase_req_slots = slot_set_list[8:11]
unique_phase_slots = ['introductions', 'met_before', 'goal_elicitation',
                      'interest_elicitation', 'session_recommendation',
                      'person_recommendation', 'food_recommendation', 'selfie']

# Feasible actions
feasible_actions = []

for i, act in enumerate(unique_phase_acts):
    for cs in agent_cs:
        feasible_actions.append(
            {
                'act': act,
                'request_slots': '',
                'inform_slots': {},
                'CS': cs,
                'phase': unique_phases[i]
            }
        )
truncated_phase_list = ['session_recommendation', 'person_recommendation',
                        'food_recommendation']
# Add only-act actions to the feasible_actions list
for act in non_req_inform_acts:
    for cs in agent_cs:
        for phase in truncated_phase_list:
            feasible_actions.append(
                {
                    'act': act,
                    'request_slots': '',
                    'inform_slots': {},
                    'CS': cs,
                    'phase': phase
                }
            )

additional_phase_list = ['introductions', 'met_before', 'goal_elicitation',
                         'interest_elicitation']

for add_phases in additional_phase_list:
    for cs in agent_cs:
        feasible_actions.append(
            {
                'act': 'give_feedback',
                'request_slots': '',
                'inform_slots': {},
                'CS': cs,
                'phase': add_phases
            }
        )

for i, slot in enumerate(unique_phase_req_slots):
        for cs in agent_cs:
            feasible_actions.append(
                {
                    'act': 'request',
                    'request_slots': slot,
                    'inform_slots': {},
                    'CS': cs,
                    'phase': unique_phase_slots[i]
                }
            )


# Add request actions to the feasible_actions list
for slot in ambiguous_phase_req_slots:
    for cs in agent_cs:
        for phase in truncated_phase_list:
            feasible_actions.append(
                {
                    'act': 'request',
                    'request_slots': slot,
                    'inform_slots': {},
                    'CS': cs,
                    'phase': phase
                }
            )


# Add inform actions to the feasible_actions list
for slot in sys_inform_slots:
    for cs in agent_cs:
        feasible_actions.append(
            {
                'act': 'inform',
                'request_slots': '',
                'inform_slots': {slot: 'info_' + slot},
                'CS': cs,
                'phase': slot + '_recommendation'
            }
        )

eq_class = {}
for i, fa in enumerate(feasible_actions):
    inf_slot_key = ""
    inf_slots = list(fa['inform_slots'].keys())
    if len(inf_slots) > 0:
        inf_slot_key = inf_slots[0]
    key = fa['act'] + fa['request_slots'] + inf_slot_key + fa['phase']
    if key not in eq_class.keys():
        eq_class[key] = []
    eq_class[key].append(i)


action_group_dict = {}
for k, v in eq_class.items():
    for a in v:
        action_group_dict[a] = v
