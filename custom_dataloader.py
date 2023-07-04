from custom_dataset import Spoof_Dataset
from custom_transforms import Custom_Transforms
from torch.utils.data import DataLoader

# Just in case you need the dataset and not the dataloader (to make a custom iterable?)
def get_datasets(transforms):
    trainDS = Spoof_Dataset(base_dir='./datasets/data/', 
                        train_img_transform=transforms.train_img_transform,
                        train_mask_transform=transforms.train_mask_transform,
                        isTrain=True)

    testDS = Spoof_Dataset(base_dir='./datasets/data/',
                       test_img_transform=transforms.test_img_transform,
                       test_mask_transform=transforms.test_mask_transform,
                       isTrain=False)
    
    return trainDS, testDS


def get_dataloaders(batch_size=16, num_workers=8):
    transforms = Custom_Transforms()
    trainDS, testDS = get_datasets(transforms)

    trainLoader = DataLoader(trainDS, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             num_workers=num_workers)

    # NOTE: Do NOT shuffle test data
    testLoader = DataLoader(testDS,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    return trainLoader, testLoader