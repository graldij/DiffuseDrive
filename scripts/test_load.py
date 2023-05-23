import torch
from diffuser.datasets.sequence import CollectedSequenceDataset
import time
from torchvision import transforms


dataset = CollectedSequenceDataset(past_image_cond=True)
# dataset.dataset.cleanup_cache_files()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=0, pin_memory=True)

preprocess = transforms.Compose([
    # transforms.Resize((self.img_size,self.img_size)),
    # transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

start = time.time()
for i in iter(dataloader):
    # print(i[0])
    # preprocessed = preprocess(i[2].reshape((i[2].shape[0] * i[2].shape[1], i[2].shape[2], i[2].shape[3], i[2].shape[4])))
    preprocessed = preprocess(i[2])
    # print(preprocessed)
    end = time.time()
    print(end-start)
    start = time.time()
    # break