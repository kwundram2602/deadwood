try:
    import albumentations as A

    def get_train_transform():
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])

except ImportError:
    def get_train_transform():
        return None
