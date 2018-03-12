import random
import numpy as np
import json

num_reco_str = 'num_reco'
primary_goal_str = 'primary_goal'
time_str = 'time'
rapport_care_str = 'rapport_care'
cs_type_str = 'cs_type'


class RuleBasedUserSimulator:
    def __init__(self, param_user):
        self.goal = None
        self.agenda = None
        self.type = None
        self.int_state = None
        self.num_cs_type = None
        self.param = param_user
        self.max_turns = param_user['max_turns']
        self.max_recos = param_user['max_recos']
        self.min_turns = param_user['min_turns']
        self.large_neg_penalty = param_user['constraint_violation_penalty']
        self.prev_agent_action = None
        self.prev_user_action = None
        self.count_slots = None
        self.reward_slots = None

    def initialize(self):
        self.prev_agent_action = None
        self.prev_user_action = None

        user_type = self.generate_random_user_type()
        self.type = user_type

        user_goal = self.generate_random_user_goal()
        self.goal = user_goal

        agenda = self.generate_user_agenda()
        self.agenda = agenda

        self.int_state = {}
        pass
        self.int_state['num_agent_cs'] = 0
        self.int_state['num_turns'] = 0
        self.int_state['rapport_built'] = None
        self.int_state['prev_feedback'] = None
        self.int_state['f-s'] = None
        self.int_state['phase'] = None
        self.int_state['total_reco'] = 0
        self.int_state["violation_" + 'total_reco'] = 0
        self.int_state['interest_asked'] = False
        self.reward_slots = self.param['reward_slots']
        self.count_slots = self.param['count_slots']

        for r_slot in self.reward_slots:
            self.int_state[r_slot] = 0
            self.int_state["violation_" + r_slot] = 0

        for c_slot in self.count_slots:
            self.int_state[c_slot] = 0
            self.int_state["violation_" + c_slot] = 0


    def generate_random_user_goal(self):
        user_goal_slots = self.param['user_goal_slots']
        user_goal = {}
        for slot in user_goal_slots:
            if len(user_goal_slots[slot]) == 2:
                user_goal[slot] = False
            if len(user_goal_slots[slot]) == 1:
                user_goal.update({slot: user_goal_slots[slot]})

        if self.type['num_person'] > 0:
            user_goal['goal_person'] = True

        if self.type['num_session'] > 0:
            user_goal['goal_session'] = True

        return user_goal

    def generate_random_user_type(self):
        user_type_slots = self.param['user_type_slots']
        prob_user_type = self.param['prob_user_type']
        small_penalty = self.param['small_penalty']
        large_penalty = self.param['large_penalty']

        user_type = {}
        binary_slots = []

        for slot in user_type_slots:
            if len(user_type_slots[slot]) == 2:
                binary_slots.append(slot)

        binary_slots.remove(rapport_care_str)

        # Pick a slot value according to the result of the biased coin toss
        for b_slot in binary_slots:
            r = random.random()
            if prob_user_type[b_slot] > r:
                user_type[b_slot] = True
            else:
                user_type[b_slot] = False

        # Select the total number of recos using a multinomial distribution
        x = np.random.multinomial(1, prob_user_type[num_reco_str])
        num_reco = int(np.where(x == 1)[0][0] + 1)
        user_type[num_reco_str] = num_reco
        # Select the number of person recos using a multinomial distribution
        x = np.random.multinomial(1, prob_user_type[num_reco_str + "_" +
                                                    str(num_reco)])
        num_person = int(np.where(x == 1)[0][0])
        num_session = num_reco - num_person
        user_type['num_person'] = num_person
        user_type['num_session'] = num_session

        # Select the user CS type using a multinomial distribution
        x = np.random.multinomial(1, prob_user_type[cs_type_str])
        cs_type = int(np.where(x == 1)[0][0] + 1)
        user_type[cs_type_str] = cs_type - 1
        # print("CS Type: ", cs_type)

        # Primary Goal
        if num_person >= num_session:
            user_type[primary_goal_str] = 'goal_person'
        else:
            user_type[primary_goal_str] = 'goal_session'

        # Time at hand (Less / Enough / More)
        if num_reco <= 2:
            user_type[time_str] = user_type_slots[time_str][0]
        elif num_reco <= 4:
            user_type[time_str] = user_type_slots[time_str][1]
        else:
            user_type[time_str] = user_type_slots[time_str][2]

        # Does the user care about building rapport or not?
        r = random.random()
        if prob_user_type[rapport_care_str][num_reco-1] > r:
            user_type[rapport_care_str] = True
        else:
            user_type[rapport_care_str] = False

        # If the user cares about building rapport, it will apply a small per-
        # turn penalty only if it has less time. On the other hand, if the user
        # doesn't care about building rapport, it will apply a large per-turn
        # penalty if it has less time, small per-turn penalty if it has enough
        # time and no penalty if it has more time.

        if user_type[rapport_care_str]:
            if user_type[time_str] == 'LT':
                user_type['penalty'] = small_penalty
            else:
                user_type['penalty'] = 0
        else:
            if user_type[time_str] == 'LT':
                user_type['penalty'] = large_penalty
            elif user_type[time_str] == 'ET':
                user_type['penalty'] = small_penalty
            else:
                user_type['penalty'] = 0

        return user_type
    
    @staticmethod
    def generate_user_agenda():
        user_agenda = []
        return user_agenda

    def next(self, agent_action):
        r_t = self.update_agenda(agent_action)
        return self.agenda.pop(), r_t

    def generic_user_action(self, act):
        user_action = {}
        pass
        user_action['act'] = act
        user_action['CS'] = 'None'
        user_action['inform_slots'] = {}
        user_action['request_slots'] = ''
        self.agenda.append(user_action)
        return user_action

    def abnormal_update(self, act):
        r_t = 0
        r_t += self.large_neg_penalty
        user_action = self.generic_user_action(act)
        r_t += self.reward_t(user_action)
        return r_t

    def update_agenda(self, agent_action):
        user_action = {}
        pass
        user_action['act'] = 'null'
        user_action['inform_slots'] = {}
        user_action['request_slots'] = ''

        act = agent_action['act']
        cs = agent_action['CS']
        user_cs_type = self.type['cs_type']
        agent_cs_count = self.int_state['num_agent_cs']

        r_t = 0
        self.int_state['num_turns'] += 1

        # End the conversation if it becomes too long.
        if self.int_state['num_turns'] > self.max_turns:
            return self.abnormal_update('bye')

        # End the conversation if total number of times agent has asked for
        # feedback or send_msg_tlink or total recos exceeds the maximum
        # possible recos.
        if self.int_state["violation_" + "total_reco"] + self.int_state[
            "total_reco"] > self.max_recos:
            return self.abnormal_update('bye')

        for c_slot in self.count_slots:
            if self.int_state[c_slot] + self.int_state["violation_" + c_slot] > \
                    self.max_recos:
                return self.abnormal_update('bye')

        for r_slot in self.reward_slots:
            if self.int_state[r_slot] + self.int_state["violation_" + r_slot]\
                    > self.max_recos:
                return self.abnormal_update('bye')

        # Check if any other constraint gets violated.
        if self.constraint_violated(agent_action):
            if act == 'request':
                request_slot = agent_action['request_slots']
                for r_slot in self.reward_slots:
                    if request_slot == r_slot:
                        self.int_state["violation_" + r_slot] += 1

            if act == 'inform':
                inform_slots = agent_action['inform_slots'].keys()
                for c_slot in self.count_slots:
                    if c_slot in inform_slots:
                        self.int_state["violation_" + c_slot] += 1
                        self.int_state["violation_" + 'total_reco'] += 1

            return self.abnormal_update('constraint_violated')

        # Now, normal updates begin.
        self.prev_agent_action = agent_action

        if act == 'request':
            request_slot = agent_action['request_slots']
            for r_slot in self.reward_slots:
                if request_slot == r_slot:
                    self.int_state[r_slot] += 1

        if act == 'inform':
            inform_slots = agent_action['inform_slots'].keys()
            for c_slot in self.count_slots:
                if c_slot in inform_slots:
                    self.int_state[c_slot] += 1
                    self.int_state['total_reco'] += 1

        if user_cs_type == 0 and cs == 'None':
            agent_cs_count += 1
        if user_cs_type == 1 and (cs == 'SD' or cs == 'QESD'):
            agent_cs_count += 1
        if user_cs_type == 2 and (cs == 'SD' or cs == 'PR'):
            agent_cs_count += 1
        if user_cs_type == 3 and (cs == 'PR' or cs == 'HE'):
            agent_cs_count += 1

        self.int_state['num_agent_cs'] = agent_cs_count

        if act == 'greeting':
            user_action['act'] = 'greeting'

        elif act == 'bye':
            user_action['act'] = 'bye'

        elif act == 'request':
            user_action['act'] = 'inform'
            user_action['inform_slots'] = self.process_request_act(
                agent_action)

        user_action['CS'] = self.generate_conv_strat(user_action)
        self.agenda.append(user_action)

        r_t += self.reward_t(user_action)
        self.prev_user_action = user_action
        return r_t


    def generate_conv_strat(self, user_action):
        cs_type = self.type[cs_type_str]
        user_act = user_action['act']
        slot = ""
        if user_act == 'inform':
            slot = list(user_action['inform_slots'].keys())[0]

        always_sd_slots = ['first_time', 'met_before', 'interest',
                           'primary_goal']
        cs_type_slots = ['feedback', 'send_msg_tlink']

        cs = "None"

        if user_act == 'greeting' or slot in always_sd_slots:
            cs = "SD"

        elif cs_type == 0 or self.type['rapport_care'] is False:  # None type
        # elif cs_type == 0:
            cs = "None"

        elif cs_type == 1:  # SD-QESD type
            if slot in cs_type_slots:
                cs = "SD"

        elif cs_type == 2:  # SD-PR type
            if user_act == 'bye':
                cs = "PR"
            elif slot in cs_type_slots:
                if user_action['inform_slots'][slot]:
                    cs = "PR"
                else:
                    cs = "SD"

        elif cs_type == 3:  # PR-HE type
            if user_act == 'bye':
                cs = "PR"
            if slot == 'another_reco':
                cs = "HE"
            elif slot in cs_type_slots:
                if user_action['inform_slots'][slot]:
                    cs = "PR"
                else:
                    cs = "HE"

        return cs

    def process_request_act(self, agent_action):

        slot = agent_action['request_slots']
        decision_points = self.param['decision_points']
        prob_funcs = self.param['prob_funcs']
        threshold_cs = self.param['threshold_cs']

        min_cs_turns_for_rapport = threshold_cs[self.type['cs_type']] \
                       * self.int_state['num_turns']
        self.int_state['phase'] = agent_action['phase']

        if slot == 'interest':
            self.int_state['interest_asked'] = True

        if slot == 'selfie':
            return {slot: True}

        for t in self.type.keys():
            if slot == t:
                return {slot: self.type[t]}

        for g in self.goal.keys():
            if slot == g:
                return {slot: self.goal[g]}

        for dp in decision_points:
            if slot == 'feedback' and self.int_state['rapport_built'] is None:
                # Rapport built or not?
                if self.int_state['num_agent_cs'] >= min_cs_turns_for_rapport:
                    if self.int_state['num_turns'] >= self.min_turns:
                        self.int_state['rapport_built'] = True
                else:
                    self.int_state['rapport_built'] = False
            if slot == dp:
                val = prob_funcs[slot](self.type, self.int_state)
                for r_slot in self.reward_slots:
                    if slot == 'feedback':
                        self.int_state['prev_feedback'] = val
                        self.int_state['f-s'] = True
                    if slot == 'send_msg_tlink':
                        self.int_state['f-s'] = False
                return {slot: val}


    def reward_t(self, user_action):
        user_act = user_action['act']
        inform_slots = list(user_action['inform_slots'].keys())
        reward = self.param['reward']
        r_t = self.type['penalty']

        if user_act == 'inform':
            for r_slot in self.reward_slots:
                if r_slot in inform_slots:
                    if user_action['inform_slots'][r_slot]:
                        r_t += reward[r_slot]

        if user_act == 'bye':
            for c_slot in self.count_slots:
                if c_slot == 'food':
                    continue
                if self.int_state[c_slot] < self.type['num_' + c_slot]:
                    r_t += self.large_neg_penalty

        return r_t

    def constraint_violated(self, agent_action):

        agent_act = agent_action['act']
        req_slot = agent_action['request_slots']
        inform_slots = list(agent_action['inform_slots'].keys())
        agent_phase = agent_action['phase']

        not_first_acts = ['give_feedback', 'glad', 'send_msg', 'no_worries', 'sorry']
        not_first_req_slots = ['feedback', 'send_msg_tlink', 'another_reco']

        if self.prev_agent_action is None: # First turn of the conversation
            if agent_act in not_first_acts:
                return True
            elif agent_act == 'request':
                if req_slot in not_first_req_slots:
                    return True

        else:
            prev_agent_act = self.prev_agent_action['act']
            prev_agent_req_slot = self.prev_agent_action['request_slots']
            prev_agent_phase = self.prev_agent_action['phase']

            prev_user_act = self.prev_user_action['act']
            prev_user_inform_slot = self.prev_user_action['inform_slots']

            # Greeting cannot be come later than first turn in the dialog
            if agent_act == 'greeting':
                return True

            # Introductions can only come after greeting
            if agent_act == 'introduce':
                if prev_agent_act != 'greeting':
                    return True

            # Ask for interests before making a recommendation
            if agent_act == 'inform':
                for c_slot in self.count_slots:
                    if c_slot in inform_slots:
                        if self.int_state['interest_asked'] is False:
                            return True


            # If the agent is giving feedback, it must have requested for
            # some information in the preceding turn.
            if agent_act == 'give_feedback':
                if prev_agent_act != 'request':
                    return True
                elif prev_agent_req_slot == 'feedback' or prev_agent_req_slot == \
                        'send_msg_tlink':
                    return True
                elif agent_phase != prev_agent_phase:
                    return True

            # If the agent is requesting for sending a Toplink message,
            # it better have asked for feedback immediately before that.
            if req_slot == 'send_msg_tlink':
                if self.int_state['f-s'] is False:
                    return True

                if agent_phase != prev_agent_phase:
                    return True

            # If the agent is glad, user better have responded with a
            # positive feedback about a recommendation in the previous turn.
            if agent_act == 'glad':
                if prev_user_inform_slot != {}:
                    k = list(prev_user_inform_slot.keys())[0]
                    if k != 'feedback' and prev_user_inform_slot[k] is False:
                        return True

                else:
                    return True

                if agent_phase != prev_agent_phase:
                    return True

            # If the agent is sorry, user better have responded with a
            # negative feedback about a recommendation in the previous turn.
            if agent_act == 'sorry':
                if prev_user_inform_slot != {}:
                    k = list(prev_user_inform_slot.keys())[0]
                    if k != 'feedback' and prev_user_inform_slot[k] is True:
                        return True

                else:
                    return True

                if agent_phase != prev_agent_phase:
                    return True

            if agent_act == 'send_msg':
                if prev_user_inform_slot != {}:
                    k = list(prev_user_inform_slot.keys())[0]
                    if k == 'send_msg_tlink':
                        if prev_user_inform_slot[k] is False:
                            return True

                else:
                    return True

                if agent_phase != prev_agent_phase:
                    return True

            if agent_act == 'no_worries':
                if prev_user_inform_slot != {}:
                    k = list(prev_user_inform_slot.keys())[0]
                    if k == 'send_msg_tlink':
                        if prev_user_inform_slot[k] is True:
                            return True

                else:
                    return True

                if agent_phase != prev_agent_phase:
                    return True

            if prev_user_inform_slot != {}:
                k = list(prev_user_inform_slot.keys())[0]
                if k == 'send_msg_tlink':
                    if prev_user_inform_slot[k] is True:
                        if agent_act != 'send_msg':
                            return True

            if req_slot == 'another_reco':
                if self.int_state['prev_feedback'] is None:
                    return True

                elif agent_phase != prev_agent_phase:
                    return True

            if agent_act == 'take_selfie':
                if prev_agent_req_slot != 'selfie':
                    return True

                elif prev_user_inform_slot != {}:
                    k = list(prev_user_inform_slot.keys())[0]
                    if k == 'selfie':
                        if prev_user_inform_slot[k] is False:
                            return True

                else:
                    return True

            if prev_user_inform_slot != {}:
                k = list(prev_user_inform_slot.keys())[0]
                if k == 'selfie':
                    if prev_user_inform_slot[k] is True:
                        if agent_act != 'take_selfie':
                            return True

            if req_slot == 'feedback':
                if prev_agent_act != 'inform':
                    return True

                elif agent_phase != prev_agent_phase:
                    return True

            if prev_agent_act == 'take_selfie':
                if agent_act != 'bye':
                    return True