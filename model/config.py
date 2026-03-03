import torch
import os
PRUNED_DIR = "E:/preprocess/pruned_network"  
EMBEDDING_PATH = "E:/preprocess/data/keywords_textembeddings.json" 
SAVE_DIR = r"E:/model/train"
HIDDEN_DIM = 256
OUT_DIM = 128
NUM_EPOCHS = 50
LR = 0.001
VAL_RATIO = 0.2
NEG_SAMPLE_RATIO = 0.5
NUM_HEADS = 4
TIME_EMB_DIM = 16
USE_CUDA = torch.cuda.is_available()
DEBUG_MODE = False

OUT_DIM_LIST = [128]
LR_LIST = [0.001]
NUM_HEADS_LIST = [8]

device = torch.device("cuda" if USE_CUDA and not DEBUG_MODE else "cpu")
print(f"{device}, {torch.cuda.is_available()}")
if USE_CUDA:
    print(f"CUDA version: {torch.version.cuda}")
    torch.cuda.empty_cache()

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "epoch_embeddings1"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "epoch_embeddings_final"), exist_ok=True)