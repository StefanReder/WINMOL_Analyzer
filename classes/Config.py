class Config(object):
    # execution / backend selection
    prediction_backend = "auto"          # auto | cpu | single_gpu | multi_gpu

    # tiled stream production pipeline
    tile_inner_px = 4096
    tile_overlap_m = 12.0
    prediction_prefetch = 2
    producer_queue_batches = 4
    prediction_producer_workers_cpu = 1
    prediction_producer_workers_gpu = 4
    prediction_producer_workers_multi_gpu = 6

    # resources
    max_gpu_workers = 8
    max_cpu_workers = 32
    gpu_memory_fraction = 0.9

    # runtime worker / batching knobs
    prediction_batch_cpu = 1
    prediction_batch_gpu = 4
    prediction_batch_max_gpu = 12
    prediction_batch_multi_gpu = 12     # local per-worker batch
    prediction_batch_autotune = True
    prediction_batch_autotune_patience = 4
    prediction_batch_autotune_repeats = 5
    prediction_batch_autotune_min_improve = 0.005
    prediction_batch_autotune_stop_on_oom = True
    progress_interval_s_cpu = 45.0
    progress_interval_s_gpu = 60.0
    progress_interval_s_multi_gpu = 20.0
    single_gpu_cpu_workers = 24
    multi_gpu_cpu_workers = 48

    # runtime state populated by planner
    cpu_workers = None
    gpu_workers = None
    vector_mode = 'none'
    vector_tile_workers = 1
    prediction_batch_size = None
    prediction_producer_workers = None
    progress_interval_s = 30.0

    # behavior
    stream_prediction = True
    keep_temp_tiles = False
    compress_output = True

    # logging / diagnostics
    prediction_tile_log = True
    vector_debug = False
    vector_summary_log = True

    # semantic segmentation
    tile_size = 15
    img_width = 512
    img_height = 512
    img_bit = 8
    n_channels = 3
    num_classes = 1
    overlap_pred = 8

    # binary stem-map prediction
    stem_map_binary = True
    stem_binary_threshold = 0.5

    # stem vectorization
    min_length = 2.0
    max_distance = 8
    max_tree_height = 32
    tolerance_angle = 7

    # measuring points / diameter estimation
    measuring_point_spacing_m = 0.5
    diameter_method = "contour"         # contour | edt
    diameter_vector_half_length_m = 1.0
    edt_clip_max_m = None                # optional clip for extreme EDT radii
    direction_confidence_threshold = 0.15


    # vector partitioned merge
    vector_partition_size_m = 80.0
    vector_partition_overlap_m = None
    vector_partition_border_band_m = None
    vector_partition_workers = 1
    partition_connect_second_pass = True
    partition_dedup_buffer_m = 0.02

    # tile/vector export behavior
    write_tile_nodes_vectors = False
    write_tile_stems_only = True

    # direction confidence bands
    direction_confidence_high = 0.75
    direction_confidence_low = 0.35

    def __init__(self):
        pass

    def to_dict(self):
        return {a: getattr(self, a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self, a))}

    def display(self):
        print("\nConfigurations:")
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        print("\n")
