import json
from dialog_arbiter.state_tracker import StateTracker


class DialogArbiter:
    def __init__(self, user, agent, slot_set, param_state_tracker):
        self.user = user
        self.agent = agent
        self.state_tracker = StateTracker()

        self.user_action = None
        self.agent_action = None

        self.dialog_over = False
        self.reward = None

        self.s_t = None

        self.slot_set = slot_set
        self.param_state_tracker = param_state_tracker

    def initialize(self):
        dialog_over, state = self.state_tracker.initialize(self.slot_set,
                                                           self.param_state_tracker)
        self.s_t = state
        self.agent_action = self.agent.initialize(state)
        self.user.initialize()
        self.reward = 0

        # print("New dialog. User goal, user type:")
        # print(json.dumps(self.user.goal, indent=2))
        # print(json.dumps(self.user.type, indent=2))

    def next(self, record_training_data=True, print_info=False):
        if print_info:
            self.print_info(agent_action=self.agent_action)
        self.state_tracker.update(agent_action=self.agent_action)

        user_action, r_t = self.user.next(self.agent_action)
        self.reward += r_t
        self.user_action = user_action

        dialog_over, state = self.state_tracker.update(user_action=self.user_action)
        if print_info:
            self.print_info(user_action=self.user_action)
            print("Reward: ", r_t)

        s_tplus1 = state
        if record_training_data:
            self.agent.register_experience_replay_tuple(self.s_t,
                                                        self.agent_action,
                                                        r_t,
                                                        s_tplus1, dialog_over)

        self.s_t = s_tplus1
        # print("State: ", json.dumps(state, indent=2))

        if not dialog_over:
            agent_action = self.agent.next(state)
            self.agent_action = agent_action
            # print("Upcoming agent action: ", agent_action)

        return self.reward, dialog_over, state

    @staticmethod
    def print_info(agent_action=None, user_action=None):
        if agent_action:
            print("Agent: ", end="")
            action = agent_action
        elif user_action:
            print("User: ", end="")
            action = user_action

        act = action['act']

        if act == 'inform':
            act_slots = action[act + '_slots']
            print(act, end='')
            for slot in act_slots:
                print('(' + str(slot) + '=' + str(act_slots[slot]), end='')
            print(')', end='')

        elif act == 'request':
            act_slots = action[act + '_slots']
            print(act, end='')
            print('(' + act_slots, end='')
            print(')', end='')
        else:
            print(act + '()', end='')

        print(', CS: ', action['CS'])
