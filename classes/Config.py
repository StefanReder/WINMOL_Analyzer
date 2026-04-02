class Config(object):
    # execution
    execution_mode = "auto"              # auto | legacy_full | stream | tiled
    prediction_backend = "auto"          # auto | cpu | single_gpu | multi_gpu
    vector_backend = "auto"              # auto | global | tiled

    # tiling
    tile_inner_px = 4096
    tile_overlap_m = 12.0
    prediction_prefetch = 2

    # resources
    max_gpu_workers = 8
    max_cpu_workers = 32
    gpu_memory_fraction = 0.9

    # behavior
    stream_prediction = True
    keep_temp_tiles = False
    legacy_full_array_threshold_gb = 8.0
    compress_output = True

    # Configuration for the semantic segmentation
    tile_size = 15
    img_width = 512
    img_height = 512
    img_bit = 8
    n_channels = 3
    num_classes = 1
    overlap_pred = 8

    # Configuration for the stem vectorization
    min_length = 2.0
    max_distance = 8
    max_tree_height = 32
    tolerance_angle = 7

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
