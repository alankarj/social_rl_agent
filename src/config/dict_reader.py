def text_to_dict(path):
    slot_set = {}

    with open(path, 'r') as f:
        index = 0
        for line in f.readlines():
            slot_set[line.rstrip('\n').rstrip('\r')] = index
            index += 1

    return slot_set
