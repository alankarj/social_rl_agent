from src import dialog_config
import sys


class RuleBasedUserSimulator:
    def __init__(self, agent_act_set, user_act_set, slot_set, user_sim_params):
        self.agent_act_set = agent_act_set
        self.user_act_set = user_act_set
        self.slot_set = slot_set

        self.max_turn = user_sim_params['max_turn']
        self.state = {}
        self.dialog_over = None
        self.dialog_status = None
        self.goal = None

    def initialize_dialog(self):
        self.state = {}
        pass
        self.state['history_slots'] = {}
        self.state['request_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['rest_slots'] = {}
        self.state['turn'] = -1
        self.dialog_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET
        self.goal = self.initialize_goal()

    def initialize_goal(self):
        goal = {}
        pass
        goal['inform_slots'] = [{'goal_session': True},
                                {'goal_person': True},
                                {'goal_person': False}]
        goal['request_slots'] = [{'info_session': 'UNK'},
                                 {'info_person': 'UNK'}]

        return goal

    def next(self, agent_action):
        self.state['turn'] += 2
        self.dialog_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        agent_act = agent_action['act']

        if self.state['turn'] > self.max_turn:
            self.dialog_over = True
            self.dialog_status = dialog_config.FAILED_DIALOG
            self.state['act'] = 'bye'

        else:
            self.state['history_slots'].updated(self.state['inform_slots'])
            self.state['inform_slots'].clear()

        if agent_act == "greeting":
            self.state['act'] = "greeting"

        user_action = {}
        pass
        user_action['act'] = self.state['act']
        user_action['inform_slots'] = self.state['inform_slots']
        user_action['request_slots'] = self.state['request_slots']
        user_action['turn'] = self.state['turn']

        return user_action, self.dialog_over, self.dialog_status

    def deterministic_response(self, agent_action):
        agent_act = agent_action['act']
        if agent_act == "greeting":
            self.state['act'] = "greeting"
            self.state['inform_slots'] = {}
            self.state['request_slots'] = {}



def main():
    x = input("Enter agent dialog act: ")
    print(x)


if __name__ == "__main__":
    main()
