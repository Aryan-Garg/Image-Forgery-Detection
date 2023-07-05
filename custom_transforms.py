import torchvision.transforms as T

# Define transforms
### TODO: Use albumentations for augmentations
class Custom_Transforms:
    def __init__(self):
        
        self.train_img_transform = T.Compose([T.ToPILImage(),
                                              T.Resize((256,256), antialias=True), 
                                              T.RandomHorizontalFlip(),
                                              T.ToTensor(),
                                              T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                                              ])

        self.train_mask_transform = T.Compose([T.ToPILImage(),
                                               T.Resize((256,256)),
                                               T.ToTensor(),
                                               T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                                               ])

        self.test_img_transform = T.Compose([T.ToPILImage(),
                                             T.Resize((256,256), antialias=True),  
                                             T.ToTensor(),
                                             T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                                             ])

        self.test_mask_transform = T.Compose([T.ToPILImage(),
                                              T.Resize((256,256)),
                                              T.ToTensor(),
                                              T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                                              ])