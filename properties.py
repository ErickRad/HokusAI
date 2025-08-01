epochs                = 120 # 100
learningRate          = 2e-4
etaMin                = 1e-6
alpha                 = 0.5
dropout               = 0.3
warmupRatio           = 0.10 # 22_193
plateauRatio          = 0.45 # 720_066

threshold             = 1
maxLength             = 30
maxSamples            = None

topK                  = 30
topP                  = 0.9
temperature           = 1.2

patchSize             = 16
batchSize             = 64
embedSize             = 512
hiddenSize            = 512
dimFF                 = 3072
numLayers             = 8
heads                 = 8

IMAGE_DIR_TRAIN       = "data/train2017"
IMAGE_DIR_VAL         = "data/val2017"
CAPTION_FILE_TRAIN    = "data/annotations/captions_train2017.json"
CAPTION_FILE_VAL      = "data/annotations/captions_val2017.json"

VOCAB_PATH            = 'data/vocab.pkl'
LOGS_PATH             = 'logs/'