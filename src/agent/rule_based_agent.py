import dialog_config
import random

console_width = 137  # Used for printing phase string

class RuleBasedAgent:
    def __init__(self, params=None):
        self.history = None
        self.phase = None
        self.current_action = None
        self.prev_action = None
        self.max = None

    def initialize(self, state=None):
        self.history = {}
        self.phase = 'greetings'
        agent_action = {}
        pass
        agent_action['act'] = 'greeting'
        agent_action['phase'] = 'greetings'
        agent_action['request_slots'] = ''
        agent_action['inform_slots'] = {}
        # self.print_phase()

        # (Uniformly) random social reasoner
        N = len(dialog_config.agent_cs)
        agent_action['CS'] = dialog_config.agent_cs[random.randrange(0, N-1)]

        self.current_action = agent_action
        self.max = {}
        pass
        self.max['session'] = 5
        self.max['person'] = 5
        return agent_action

    def next(self, state):
        user_action = state['user_action']
        self.prev_action = self.current_action

        user_act = user_action['act']
        inform_slots = {}
        request_slots = ''
        agent_action = {}
        phase = self.phase
        user_inform_slots = user_action['inform_slots']

        if self.phase == 'greetings':
            if user_act == 'greeting':
                phase = 'introductions'
                act = 'introduce'
                agent_action = self.construct_action(act, phase, inform_slots,
                                                     request_slots)
        elif self.phase == 'introductions':
            if user_act == 'null':
                if state['first_time'] == '':
                    act = 'request'
                    request_slots = 'first_time'
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)

                else:
                    act = 'request'
                    phase = 'met_before'
                    request_slots = 'met_before'
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)
            if user_act == 'inform':
                act = 'give_feedback'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

        elif self.phase == 'met_before':
            if self.prev_action['act'] == 'give_feedback':
                phase = 'goal_elicitation'
                act = 'request'
                request_slots = 'primary_goal'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            else:
                act = 'give_feedback'
                agent_action = self.construct_action(act, phase, inform_slots,
                                                     request_slots)

        elif self.phase == 'goal_elicitation':
            if self.prev_action['act'] == 'give_feedback':
                phase = 'interest_elicitation'
                act = 'request'
                request_slots = 'interest'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            else:
                act = 'give_feedback'
                agent_action = self.construct_action(act, phase, inform_slots,
                                                     request_slots)

        elif self.phase == 'interest_elicitation':
            if self.prev_action['act'] == 'give_feedback':
                phase = 'session_recommendation'
                act = 'request'
                request_slots = 'goal_session'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            else:
                act = 'give_feedback'
                agent_action = self.construct_action(act, phase, inform_slots,
                                                     request_slots)

        elif self.phase == 'session_recommendation':
            reco_type = "session"
            alt_reco_type = "person"
            agent_action, phase = self.process_recommendation(reco_type,
                                                              alt_reco_type,
                                                              state,
                                                              user_action)

        elif self.phase == 'person_recommendation':
            reco_type = "person"
            alt_reco_type = "food"
            agent_action, phase = self.process_recommendation(reco_type,
                                                              alt_reco_type,
                                                              state,
                                                              user_action)

        elif self.phase == 'food_recommendation':
            if not state['goal_food']:
                phase = 'selfie'
                act = 'request'
                request_slots = 'selfie'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            elif self.prev_action['act'] == 'inform':
                act = 'request'
                request_slots = 'feedback'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            elif user_act == 'inform':
                if 'feedback' in user_inform_slots.keys():
                    if user_inform_slots['feedback']:
                        act = 'glad'
                        agent_action = self.construct_action(act, phase,
                                                             inform_slots,
                                                             request_slots)
                    else:
                        act = 'sorry'
                        agent_action = self.construct_action(act, phase,
                                                             inform_slots,
                                                             request_slots)
                else:
                    act = 'inform'
                    inform_slots = {'food': 'info_food'}
                    agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)
            else:
                phase = 'selfie'
                act = 'request'
                request_slots = 'selfie'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

        elif self.phase == 'selfie':
            if user_act == 'null':
                phase = 'farewell'
                act = 'bye'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            else:
                act = 'take_selfie'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

        self.current_action = agent_action
        old_phase = self.phase
        self.phase = phase
        # if self.phase != old_phase:
        #     self.print_phase()
        return agent_action

    def process_recommendation(self, reco_type, alt_reco_type, state,
                               user_action):
        inform_slots = {}
        request_slots = ''
        user_act = user_action['act']
        user_inform_slots = user_action['inform_slots']
        phase = self.phase

        if not state['goal_' + reco_type]:
            phase = alt_reco_type + '_recommendation'
            act = 'request'
            request_slots = 'goal_' + alt_reco_type
            agent_action = self.construct_action(act, phase,
                                                 inform_slots,
                                                 request_slots)

        elif user_act == 'inform':
            if 'feedback' in user_inform_slots.keys():
                if user_inform_slots['feedback']:
                    act = 'glad'
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)
                else:
                    act = 'sorry'
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)

            elif 'send_msg_tlink' in user_inform_slots.keys():
                if user_inform_slots['send_msg_tlink']:
                    act = 'send_msg'
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)
                else:
                    act = 'no_worries'
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)

            else:
                if state[reco_type] == 0 or user_inform_slots['another_reco']:
                    #print(state[reco_type])
                    act = 'inform'
                    inform_slots = {reco_type: "info_" + reco_type}
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)

                else:
                    phase = alt_reco_type + '_recommendation'
                    act = 'request'
                    request_slots = 'goal_' + alt_reco_type
                    agent_action = self.construct_action(act, phase,
                                                         inform_slots,
                                                         request_slots)

        else:
            if self.prev_action['act'] == 'inform':
                act = 'request'
                request_slots = 'feedback'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            elif self.prev_action['act'] == 'glad':
                act = 'request'
                request_slots = 'send_msg_tlink'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            elif state[reco_type] < self.max[reco_type]:
                act = 'request'
                request_slots = 'another_reco'
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

            else:
                phase = alt_reco_type + '_recommendation'
                act = 'request'
                request_slots = 'goal_' + alt_reco_type
                agent_action = self.construct_action(act, phase,
                                                     inform_slots,
                                                     request_slots)

        return agent_action, phase

    def construct_action(self, act, phase, inform_slots, request_slots):
        agent_action = {}
        pass
        agent_action['act'] = act
        agent_action['phase'] = phase
        agent_action['inform_slots'] = inform_slots
        agent_action['request_slots'] = request_slots

        # (Uniformly) random social reasoner
        N = len(dialog_config.agent_cs)
        agent_action['CS'] = dialog_config.agent_cs[random.randrange(0, N-1)]

        return agent_action

    def print_phase(self):
        phase = self.phase
        print(phase.center(console_width, '-'))
