import PIL

def CutoutAbs(img, ratio, color, position):
    if ratio < 0:
        return img

    w, h = img.size
    size = ratio * img.size[0]

    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - size / 2.))
    y0 = int(max(0, y0 - size / 2.))
    x1 = min(w, x0 + size)
    y1 = min(h, y0 + size) 

    xy = (x0, y0, x1, y1)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

class MaskAugment:
    """
    mask_ratio: str or float (0., 1.), ratio of mask wrt. image size, or 'random'
    mask_color: str or tuple, specify color in 3 channels, or 'average' for average of this image
    mask_position: str, defaults to and only supports 'random'
    """
    def __init__(self, mask_ratio='random', mask_color=(0, 0, 0), mask_position='random'):
        self.mask_ratio = mask_ratio
        self.mask_color = mask_color
        self.mask_position = mask_position
        
    def __call__(self, img):
        if self.mask_ratio == "random":
            self.mask_ratio = random.random() * 0.5 

        assert 0 <= self.mask_ratio < 1
            
        img = Cutout(img, self.mask_ratio, self.mask_color, self.mask_position)
        return img