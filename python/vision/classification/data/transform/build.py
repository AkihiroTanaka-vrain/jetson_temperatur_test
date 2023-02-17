import torchvision.transforms as T


def build_transforms(c_in, is_train=True):
    transforms = []
    randoms = []

    brightness = c_in.COLORJITTER_BRIGHTNESS
    contrast = c_in.COLORJITTER_CONTRAST
    saturation = c_in.COLORJITTER_SATURATION
    hue = c_in.COLORJITTER_HUE
    collorjitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    mean = c_in.NORMALIZE_MEAN
    std = c_in.NORMALIZE_STD
    normalize = T.Normalize(mean=mean, std=std)

    if c_in.CENTER_CROP_USE:
        transform = T.CenterCrop(c_in.CENTER_CROP_SIZE)
        transforms.append(transform)
    if c_in.RANDOM_CROP_USE:
        random = T.RandomCrop(c_in.RANDOM_CROP_SIZE)
        randoms.append(random)
    if c_in.RANDOM_RESIZE_CROP_USE:
        random = T.RandomResizedCrop(c_in.RANDOM_RESIZE_CROP_SIZE)
        randoms.append(random)
    if c_in.RESIZE_USE:
        transform = T.Resize(c_in.RESIZE_SIZE)
        transforms.append(transform)
    if c_in.PAD_USE:
        random = T.Pad(c_in.PAD_PADDING)
        randoms.append(random)
    if c_in.GRAYSCALE_USE:
        transform = T.Grayscale(c_in.GRAYSCALE_CHANNELS)
        transforms.append(transform)
    if c_in.RANDOM_GRAYSCALE_USE:
        transform = T.RandomGrayscale(c_in.RANDOM_GRAYSCALE_P)
        transforms.append(transform)
    if c_in.RANDOM_VERTICAL_FLIP_USE:
        transform = T.RandomVerticalFlip(c_in.RANDOM_VERTICAL_FLIP_P)
        transforms.append(transform)
    if c_in.RANDOM_HORIZONTAL_FLIP_USE:
        transform = T.RandomHorizontalFlip(c_in.RANDOM_HORIZONTAL_FLIP_P)
        transforms.append(transform)
    if c_in.RANDOM_ROTATION_USE:
        random = T.RandomRotation(c_in.RANDOM_ROTATION_DEGREES)
        randoms.append(random)
    if c_in.COLORJITTER_USE:
        transforms.append(collorjitter)
    transforms.append(T.ToTensor())
    transforms.append(normalize)
    if not is_train:
        return T.Compose(transforms)
    if randoms:
        return T.Compose([T.RandomChoice(randoms), *transforms])
    else:
        return T.Compose(transforms)
