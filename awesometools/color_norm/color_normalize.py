# from scipy import misc
import imageio as io
from awesometools.color_norm.stainNorm_Reinhard import Normalizer


class reinhard_normalizer():
    def __init__(self, target_file):
        target_40X = io.imread(target_file)
        self.n_40X = Normalizer()
        self.n_40X.fit(target_40X)

    def normalize(self, image):
        # image RGB in uint8
        return self.n_40X.transform(image)
