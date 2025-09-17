import datasets as ds
import os

# Defina cache em data/
dataDir = "data"
os.makedirs(dataDir, exist_ok=True)

dataset = ds.load_dataset(
    "shunk031/MSCOCO",
    year=2017,
    coco_task="captions",
    cache_dir=dataDir
)

print(dataset)
print("Exemplo:", dataset["train"][0])