import subprocess
import torch
import os
import pickle
import properties
import torch.nn as nn
from colorama import init, Fore
from dataset import *
from model import *
from vocabBuilder import Vocabulary, loadCaptions

init(autoreset=True)
device = None

def main():
    subprocess.run(["clear"], shell=True)

    if os.path.exists(properties.VOCAB_PATH):
        print(f"[{Fore.GREEN}✓{Fore.RESET}] Looking for vocabulary... {properties.VOCAB_PATH}.")
        with open(properties.VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)
    else:
        os.makedirs(os.path.dirname(properties.VOCAB_PATH), exist_ok=True)

        print(f"[+] Loading captions...", end="\r")
        captions = loadCaptions(properties.CAPTION_FILE_TRAIN)
        print(f"[{Fore.GREEN}✓{Fore.RESET}] Loading captions... done.")

        print(f"[+] Building vocabulary...", end="\r")
        vocab = Vocabulary(threshold=properties.threshold)
        vocab.buildVocab(captions)
        print(f"[{Fore.GREEN}✓{Fore.RESET}] Building vocabulary... {len(vocab)} words on {properties.VOCAB_PATH}.")

        with open(properties.VOCAB_PATH, 'wb') as f:
            pickle.dump(vocab, f)

    print(f"[+] Looking for CUDA...", end="\r")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[{Fore.GREEN}✓{Fore.RESET}] Looking for CUDA... {torch.version.cuda}.")
    else:
        device = torch.device("cpu")
        print(f"[{Fore.RED}X{Fore.RESET}] Looking for CUDA... not available. Resuming on CPU 💀😨.")

    print(f"[+] Starting training...\n")
    
    model = HokusAI(len(vocab), device)
    model.train(loader(vocab), properties.epochs, vocab)

    print(f"[{Fore.GREEN}✓{Fore.RESET}] Starting training... done.")

    os.makedirs("checkpoints", exist_ok=True)
    model.save(
        visionTransformerPath="checkpoints/visionTransformer.pth",
        transformerEncoderPath="checkpoints/transformerEncoder.pth",
        transformerDecoderPath="checkpoints/transformerDecoder.pth",
        linearProjPath="checkpoints/linearProj.pth"
    )

if __name__ == "__main__":
    main()