NUM_OF_NOTE_VALUES=128 # 128 possible note values in midi
NUM_OF_VELOCITY_VALUES=5 # 128 possible velocity values in midi, but 5 take up 99%

NOTE_DURATION_INTERVAL_SIZE=0.125
MIN_NOTE_DURATION=0
MAX_NOTE_DURATION=2 # max 2 seconds
NUM_OF_DURATION_INTERVALS=int(MAX_NOTE_DURATION/NOTE_DURATION_INTERVAL_SIZE) # 16

MODEL_INPUT_SIZE=NUM_OF_NOTE_VALUES + NUM_OF_VELOCITY_VALUES + NUM_OF_DURATION_INTERVALS # 272
MODEL_OUTPUT_SIZE=MODEL_INPUT_SIZE # 272

# Hyper Parameters
SEQUENCE_LENGTH=64
MODEL_HIDDEN_SIZE=596
BATCH_SIZE=32
LEARNING_RATE=0.00005 # 0.001

