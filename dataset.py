import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from functools import partial
from torchvision import transforms
import random
import properties

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotationFile, vocab, transform=None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform

        with open(annotationFile, 'r') as f:
            data = json.load(f)

        self.image_id_to_file = {
            img['id']: img['file_name'] for img in data['images']
        }

        self.samples = []
        for ann in data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            if image_id in self.image_id_to_file:
                filename = self.image_id_to_file[image_id]
                self.samples.append((filename, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, caption = self.samples[idx]
        path = os.path.join(self.image_dir, filename)
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption

def collateFn(batch, vocab):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)

    tokenized_captions = []
    lengths = []

    for cap in captions:
        tokens = [vocab.stoi['<SOS>']] + vocab.numericalize(cap) + [vocab.stoi['<EOS>']]
        tensor = torch.tensor(tokens)
        tokenized_captions.append(tensor)
        lengths.append(len(tensor))

    max_len = max(lengths)
    padded = torch.zeros(len(captions), max_len).long()
    for i, cap in enumerate(tokenized_captions):
        padded[i, :lengths[i]] = cap

    return images, padded, lengths

def loader(vocab, mode="train", batchSize=properties.batchSize, numWorkers=os.cpu_count() // 2):
    assert mode in ["train", "val"]

    imageDir = properties.IMAGE_DIR_TRAIN if mode == "train" else properties.IMAGE_DIR_VAL
    annotationFile = properties.CAPTION_FILE_TRAIN if mode == "train" else properties.CAPTION_FILE_VAL

    if mode == "train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    dataset = CustomDataset(
        image_dir=imageDir,
        annotationFile=annotationFile,
        vocab=vocab,
        transform=transform
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batchSize,
        shuffle=(mode == "train"),
        num_workers=numWorkers,
        collate_fn=partial(collateFn, vocab=vocab)
    )