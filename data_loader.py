import torch.utils.data as data
import torchvision.transforms as T
from torch.utils.data.distributed import DistributedSampler

from datasets.simple_dataset import Simple


def get_image_transform(mode='train', image_size=256, random_flip_ratio=0.0):
    transforms = [T.Resize((image_size, image_size))]
    if random_flip_ratio > 0.0 and mode == 'train':
        transforms.append(T.RandomHorizontalFlip(random_flip_ratio))
    transforms += [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

    return T.Compose(transforms)


def get_dataloader(data_root, dataset_name, image_size, mode, multi_nodes, batch_size):
    random_flip_ratio = 1.0
    dataset = None
    transform_img = get_image_transform(mode=mode, random_flip_ratio=random_flip_ratio)
    if dataset_name == 'simple':
        dataset = Simple(data_root, dataset_name, mode, transform_img)
    else:
        raise NotImplementedError
    
    if mode == 'train' and multi_nodes:
        train_sampler = DistributedSampler(dataset=dataset)
        return data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler, num_workers=batch_size), train_sampler
    else:
        return data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=batch_size, shuffle=mode == 'train')


if __name__ == "__main__":
    loader = get_dataloader('data', 'dataset_name', 256, 'train', False, 2)
