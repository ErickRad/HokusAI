import torch
from torchvision import transforms
from PIL import Image
import pickle
import argparse
import properties
from vocabBuilder import Vocabulary
from model import HokusAI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loadVocab(path="data/vocab.pkl"):
    with open(path, "rb") as f:
        vocab = pickle.load(f)

    return vocab

def preprocessImage(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    
    return image

def main():
    parser = argparse.ArgumentParser(description="Image Captioning")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
    parser.add_argument("--use-fallback", action="store_true", help="Use top K fallback to gerante captions instead of top P nucleus sampling")
    parser.add_argument("--num-captions", type=int, help="number of captions to generate", default=1)
    args = parser.parse_args()

    vocab = loadVocab()
    vocab_size = len(vocab)

    model = HokusAI(vocab_size, device)
    model.load(
        visionTransformerPath="best/visionTransformer.pth",
        transformerEncoderPath="best/transformerEncoder.pth",
        transformerDecoderPath="best/transformerDecoder.pth",
        linearProjPath="best/linearProj.pth"
    )

    image = preprocessImage(args.image_path)
    image = image.unsqueeze(0).to(device)

    print("Captions:\n")

    for i in range(args.num_captions):
        caption = model.generateCaption(image, vocab, args.use_fallback)
        print(f"{i+1}.{caption}")

    print("\n")

if __name__ == "__main__":
    main()