## Data selectoin
- remove all classes below 20 images
    - fewer images are not usefull for training
    - will be used for zero-shot testing set
- remove radnom images from clases to max number 5000
    - otherwise to unbalanced
    - only removes a lot of usa images

- remaining 87 countries and ca. 42.707 images
    - 80% training
    - 20% test
        - min 4 images per class
    - split evenly (per calss)
- use the remaining 87 country list for all batches