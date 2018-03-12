from src import dialog_config
import json

feasible_actions = dialog_config.feasible_actions
total_actions = len(feasible_actions)
print(total_actions)
print(json.dumps(dialog_config.feasible_actions, indent=2))

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

print(eq_class)

action_group_dict = {}
for k, v in eq_class.items():
    for a in v:
        action_group_dict[a] = v

print(action_group_dict)
