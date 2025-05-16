# model
OB_RADIUS = 2       # observe radius, neighborhood radius
OB_HORIZON = 25      # number of observation frames 25
PRED_HORIZON = 50   # number of prediction frames 50
# group name of inclusive agents; leave empty to include all agents
# non-inclusive agents will appear as neighbors only
INCLUSIVE_GROUPS = []
RNN_HIDDEN_DIM = 256

# training
LEARNING_RATE = 3e-4#8e-5#3e-4# 
BATCH_SIZE = 32 #80
EPOCHS = 150     # total number of epochs for training
EPOCH_BATCHES = 10 # 10 number of batches per epoch, None for data_length//batch_size
TEST_SINCE = 50    # the epoch after which performing testing during training

# testing
PRED_SAMPLES = 20   # best of N samples
FPC_SEARCH_RANGE = range(40, 50)   # FPC sampling rate

# evaluation
WORLD_SCALE = 1
