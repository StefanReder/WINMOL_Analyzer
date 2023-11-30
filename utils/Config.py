##################################################################################
'''Imports'''


class Config(object):
    #Coniguration for the semantic segmentation
    tile_size = 15
    IMG_width=512
    IMG_height=512
    IMG_bit=8
    n_Channels=3
    num_classes=1
    overlapp_pred=8
    #Configuration for the stem vectorization
    min_length = 2.0
    max_distance=8
    max_tree_height=32
    tolerance_angle=7
    
    def __init__(self):
        """Set values of computed attributes."""

    def to_dict(self):
        return {a: getattr(self, a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self, a))}

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        print("\n")