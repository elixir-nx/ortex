defmodule Ortex.CPUExecutionProvider do
  @moduledoc """
  """
  defstruct(use_arena: false)

  @type t :: %Ortex.CPUExecutionProvider{
          use_arena: boolean()
        }
end

defmodule Ortex.CUDAExecutionProvider do
  @moduledoc """
  """
  defstruct [
    :device_id,
    :gpu_mem_limit,
    :arena_extend_strategy,
    :cudnn_conv_algo_search,
    :do_copy_in_default_stream,
    :cudnn_conv_use_max_workspace,
    :cudnn_conv1d_pad_to_nc1d,
    :enable_cuda_graph,
    :enable_skip_layer_norm_strict_mode
  ]

  @type t :: %Ortex.CUDAExecutionProvider{
          device_id: integer(),
          gpu_mem_limit: integer(),
          arena_extend_strategy: String.t(),
          cudnn_conv_algo_search: String.t(),
          do_copy_in_default_stream: boolean(),
          cudnn_conv_use_max_workspace: boolean(),
          cudnn_conv1d_pad_to_nc1d: boolean(),
          enable_cuda_graph: boolean(),
          enable_skip_layer_norm_strict_mode: boolean()
        }
end

defmodule Ortex.TensorRTExecutionProvider do
  @moduledoc """
  """
  defstruct [
    :device_id,
    :max_workspace_size,
    :max_partition_iterations,
    :min_subgraph_size,
    :fp16_enable,
    :int8_enable,
    :int8_calibration_table_name,
    :int8_use_native_calibration_table,
    :dla_enable,
    :dla_core,
    :engine_cache_enable,
    :engine_cache_path,
    :dump_subgraphs,
    :force_sequential_engine_build,
    :enable_context_memory_sharing,
    :layer_norm_fp32_fallback,
    :timing_cache_enable,
    :force_timing_cache,
    :detailed_build_log,
    :enable_build_heuristics,
    :enable_sparsity,
    :builder_optimization_level,
    :auxiliary_streams,
    :tactic_sources,
    :extra_plugin_lib_paths,
    :profile_min_shapes,
    :profile_max_shapes,
    :profile_opt_shapes
  ]

  @type t :: %Ortex.TensorRTExecutionProvider{
          device_id: integer(),
          max_workspace_size: integer(),
          max_partition_iterations: integer(),
          min_subgraph_size: integer(),
          fp16_enable: boolean(),
          int8_enable: boolean(),
          int8_calibration_table_name: String.t(),
          int8_use_native_calibration_table: boolean(),
          dla_enable: boolean(),
          dla_core: integer(),
          engine_cache_enable: boolean(),
          engine_cache_path: String.t(),
          dump_subgraphs: boolean(),
          force_sequential_engine_build: boolean(),
          enable_context_memory_sharing: boolean(),
          layer_norm_fp32_fallback: boolean(),
          timing_cache_enable: boolean(),
          force_timing_cache: boolean(),
          detailed_build_log: boolean(),
          enable_build_heuristics: boolean(),
          enable_sparsity: boolean(),
          builder_optimization_level: integer(),
          auxiliary_streams: integer(),
          tactic_sources: String.t(),
          extra_plugin_lib_paths: String.t(),
          profile_min_shapes: String.t(),
          profile_max_shapes: String.t(),
          profile_opt_shapes: String.t()
        }
end
