from .__version__ import __version__

from . import resnet as rn
from . import senet as sn
#from . import keras_applications as ka


__all__ = ['__version__', 'Classifiers']


class Classifiers:

    _models = {

        # ResNets
        'resnet18': [rn.ResNet18, rn.preprocess_input],
        'resnet34': [rn.ResNet34, rn.preprocess_input],
        'resnet50': [rn.ResNet50, rn.preprocess_input],
        'resnet101': [rn.ResNet101, rn.preprocess_input],
        'resnet152': [rn.ResNet152, rn.preprocess_input],

        # SE-Nets
        'seresnet18': [rn.SEResNet18, rn.preprocess_input],
        'seresnet34': [rn.SEResNet34, rn.preprocess_input],
        'seresnet50': [sn.SEResNet50, sn.preprocess_input],
        'seresnet101': [sn.SEResNet101, sn.preprocess_input],
        'seresnet152': [sn.SEResNet152, sn.preprocess_input],
        'seresnext50': [sn.SEResNeXt50, sn.preprocess_input],
        'seresnext101': [sn.SEResNeXt101, sn.preprocess_input],
        'senet154': [sn.SENet154, sn.preprocess_input],

    }

    @classmethod
    def names(cls):
        return sorted(cls._models.keys())

    @classmethod
    def get(cls, name):
        """
        Access to classifiers and preprocessing functions

        Args:
            name (str): architecture name

        Returns:
            callable: function to build keras model
            callable: function to preprocess image data

        """
        return cls._models.get(name)

    @classmethod
    def get_classifier(cls, name):
        return cls._models.get(name)[0]

    @classmethod
    def get_preprocessing(cls, name):
        return cls._models.get(name)[1]
