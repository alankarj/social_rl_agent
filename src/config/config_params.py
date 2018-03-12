# Neural Net parameters
L = 0.8  # Loss value l(a,a_E) for a != a_E
beta0 = 0.6
hidden_dim = 256
gamma = 0.9
epsilon = 1
lr = 1e-3

# Other parameters
multiplier = 100  # Number of SL training steps = batch_size * multiplier
sl_file_name = '/pre-trained-model'
rl_file_name = '/pre-trained-model-rl'

# Pre-training or not? Testing or not?
warm_start_epochs = 1000
warm_start = 2
test = 2

num_dialogs = 1000
agt = 1  # 0: Rule-based, 1: Flat-RL, 2: Hierarchical RL
batch_size = 64

erp_size_sl = 1000
erp_size_rl = 100000