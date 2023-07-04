# Image Forgery Detection     

![Authentic Image](photo-grid.png "Dataset 3 classes")

### Problem Statement:    

Vision Systems can be fooled with various input side attacks like using physical printed(2D/3D) photos of another person for facial security recognition models. Some other systems use liveliness of the person (blinking, twitching etc.) so people use cut-out masks to counter that. The task is to detect authentic images from manipulated ones. Manipulations could include noise perturbations, slices, merging, composites, copied segments, in-paintings, out-paintings ( GANs ;) ) and other processing techniques available at any open-source image editing software like GIMP.

**TL;DR**

0. Detection of authentic/natural/un-altered images.

1. Detection of copied segments in the same image (copy-move) **Harder** (based on qualitative dataset surveying)

2. Detection of merged images (slicing)   

Overall, classify images into 3 categories: authentic (0), copy-moved (1) or sliced (2). 

> Traditional machine learning and then deep learning based approaches are used in this repo.

---    

### Dataset:   

* [Download SpoofSense Dataset](https://drive.google.com/file/d/1lUFc9Gx9pK9PlW0MDtoOwolgbHig4W3m/view?pli=1)

To run **everything**, create a `datasets` directory and then unzip the dataset from the link above inside this dir.


Your directory structure should look like this:   

```
.
└── datasets
    └── data
        ├── test
        │   ├── authentic
        │   ├── copy-moved
        │   │   ├── images
        │   │   └── masks
        │   └── spliced
        │       ├── images
        │       └── masks
        └── traindev
            ├── authentic
            ├── copy-moved
            │   ├── images
            │   └── masks
            └── spliced
                ├── images
                └── masks
```

---    

### Dataset Statistics:

#### Size:

*Total dataset size*: `4980` images.


| traindev   | 4482 |
|------------|------|
| authentic  | 1494 |
| copy-moved | 1494 |
| spliced    | 1494 |


| test       | 498 |
|------------|-----|
| authentic  | 166 |
| copy-moved | 166 |
| spliced    | 166 |


> *Observation:* Dataset is **balanced** OR **no class imbalance**

#### Dynamic Range: *Low* `uint8`    


#### Resolution: `384` x `256`     

---


### Approach 1:

---

### Approach 2:

---

### References:

---    


