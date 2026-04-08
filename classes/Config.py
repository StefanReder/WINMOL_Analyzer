class Config(object):
    # execution
    execution_mode = "auto"              # auto | legacy_full | stream | tiled
    prediction_backend = "auto"          # auto | cpu | single_gpu | multi_gpu
    vector_backend = "auto"              # auto | global | tiled

    # tiling / prediction
    tile_inner_px = 4096
    tile_overlap_m = 12.0
    prediction_prefetch = 2
    producer_queue_batches = 8
    prediction_producer_workers_cpu = 1
    prediction_producer_workers_gpu = 3
    prediction_producer_workers_multi_gpu = 2

    # resources
    max_gpu_workers = 8
    max_cpu_workers = 32
    gpu_memory_fraction = 0.9

    # runtime worker / batching knobs
    prediction_batch_cpu = 1
    prediction_batch_gpu = 2
    prediction_batch_max_gpu = 16
    prediction_batch_autotune = True
    prediction_batch_autotune_patience = 4
    prediction_batch_autotune_repeats = 5
    prediction_batch_autotune_min_improve = 0.005
    prediction_batch_autotune_stop_on_oom = True
    progress_interval_s_cpu = 45.0
    progress_interval_s_gpu = 60.0
    progress_interval_s_multi_gpu = 20.0
    single_gpu_cpu_workers = 12
    multi_gpu_cpu_workers = 48

    # runtime state populated by planner
    cpu_workers = None
    gpu_workers = None
    vector_tile_workers = 1
    prediction_batch_size = None
    prediction_producer_workers = None
    progress_interval_s = 30.0

    # behavior
    stream_prediction = True
    keep_temp_tiles = False
    legacy_full_array_threshold_gb = 8.0
    compress_output = True

    # binary stem-map / stripe + coarse-grid vector pipeline
    stem_map_binary = True
    stem_binary_threshold = 0.5
    stripe_pipeline = True
    stripe_inner_steps = 16
    grid_pipeline = True
    grid_inner_m = 250.0
    grid_halo_m = 12.0
    grid_inflight_tiles = 6
    grid_vector_workers = 4
    grid_vector_workers_min = 1
    grid_vector_workers_max = 4
    grid_queue_multiplier = 3.0
    grid_adaptive_workers = True
    grid_adaptive_margin = 1.15
    grid_adaptive_ema = 0.2
    grid_priority_dense_first = False
    grid_priority_use_inner = True
    grid_dense_split = False
    grid_dense_split_factor = 2.5
    grid_dense_split_min_fg = 12000
    grid_dense_split_min_samples = 4
    grid_dense_split_max_depth = 1
    grid_failure_reduce_after = 3
    grid_oom_cooldown_s = 120.0
    grid_memory_pressure_enabled = True
    grid_memory_pressure_threshold_pct = 92.0
    grid_memory_available_min_gb = 4.0
    grid_log_schedule_events = True
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
