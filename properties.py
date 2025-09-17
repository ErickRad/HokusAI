epochs                = 500
learningRate          = 1e-5
etaMin                = 1e-6
dropout               = 0.35
warmupRatio           = 0.05
plateauRatio          = 0.80

threshold             = 1
maxLength             = 30
maxSamples            = None
topK                  = 25
topP                  = 0.96
temperature           = 1.2

patchSize             = 16
batchSize             = 64
embedSize             = 512
hiddenSize            = 512

maxLen                = 5000
dimFF                 = 3072
numLayers             = 6
heads                 = 8

IMAGE_DIR_TRAIN       = "data/train2017"
IMAGE_DIR_VAL         = "data/val2017"
CAPTION_FILE_TRAIN    = "data/annotations/captions_train2017.json"
CAPTION_FILE_VAL      = "data/annotations/captions_val2017.json"

VOCAB_PATH            = 'data/vocab.pkl'
LOGS_PATH             = 'logs/'
