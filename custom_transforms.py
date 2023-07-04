import torchvision.transforms as T

# Define transforms
### TODO: Use albumentations for augmentations
class Custom_Transforms:
    def __init__(self):
        
        self.train_img_transform = T.Compose([T.Resize((224,224), antialias=True), 
                                              T.ToTensor()])

        self.train_mask_transform = T.Compose([T.Resize((224,224)),
                                               T.ToTensor()])

        self.test_img_transform = T.Compose([T.Resize((224,224), antialias=True),  
                                             T.ToTensor()])

        self.test_mask_transform = T.Compose([T.Resize((224,224)),
                                              T.ToTensor()])