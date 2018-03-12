class StateTracker:
    def __init__(self):
        self.state = None
        self.param = None
        self.dialog_over = None

    def initialize(self, slot_set, param):
        self.state = {k: '' for k in slot_set.keys()}
        self.param = param

        self.state['turn'] = 0

        count_slots = self.param['count_slots']
        reward_slots = self.param['reward_slots']
        for c_slot in count_slots:
            self.state[c_slot] = 0
        self.state['num_accepted'] = {}
        for c_slot in count_slots:
            self.state['num_accepted'][c_slot] = {}
            for r_slot in reward_slots:
                self.state['num_accepted'][c_slot][r_slot] = 0

        self.dialog_over = False
        return self.dialog_over, self.state

    def update(self, agent_action=None, user_action=None):
        count_slots = self.param['count_slots']
        reward_slots = self.param['reward_slots']

        self.state['turn'] += 1

        if agent_action:
            self.state['agent_action'] = agent_action
            self.state['phase'] = agent_action['phase']

            act = agent_action['act']
            inform_slots = agent_action['inform_slots'].keys()

            if act == 'inform':
                for slot in count_slots:
                    if slot in inform_slots:
                        self.state[slot] += 1

        elif user_action:
            self.state['user_action'] = user_action
            act = user_action['act']
            inform_slots = user_action['inform_slots'].keys()

            if act == 'bye':
                self.dialog_over = True

            elif act == 'inform':
                for slot in self.state.keys():
                    if slot in inform_slots:
                        self.state[slot] = user_action['inform_slots'][slot]

                for r_slot in reward_slots:
                    if r_slot in inform_slots:
                        if user_action['inform_slots'][r_slot]:
                            for c_slot in count_slots:
                                if self.state['phase'] == c_slot + "_recommendation":
                                    self.state['num_accepted'][c_slot][r_slot] += 1

        return self.dialog_over, self.state
