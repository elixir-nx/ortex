use ort::{
    ArenaExtendStrategy, CPUExecutionProvider, CUDAExecutionProvider,
    CUDAExecutionProviderCuDNNConvAlgoSearch, ExecutionProvider, GraphOptimizationLevel,
    SessionBuilder, TensorRTExecutionProvider,
};
use rustler::resource::ResourceArc;
use rustler::NifStruct;

#[derive(Debug, NifStruct, Clone)]
#[module = "Ortex.CUDAExecutionProvider"]
pub struct OrtexCUDAExecutionProvider {
    pub device_id: Option<i32>,
    pub gpu_mem_limit: Option<usize>,
    pub arena_extend_strategy: Option<String>,
    pub cudnn_conv_algo_search: Option<String>,
    pub do_copy_in_default_stream: Option<bool>,
    pub cudnn_conv_use_max_workspace: Option<bool>,
    pub cudnn_conv1d_pad_to_nc1d: Option<bool>,
    pub enable_cuda_graph: Option<bool>,
    pub enable_skip_layer_norm_strict_mode: Option<bool>,
}

// interim struct to move elixir Strings into Enums
#[allow(dead_code)]
struct _OrtexCUDAExecutionProvider {
    pub device_id: Option<i32>,
    pub gpu_mem_limit: Option<usize>,
    pub arena_extend_strategy: Option<ArenaExtendStrategy>,
    pub cudnn_conv_algo_search: Option<CUDAExecutionProviderCuDNNConvAlgoSearch>,
    pub do_copy_in_default_stream: Option<bool>,
    pub cudnn_conv_use_max_workspace: Option<bool>,
    pub cudnn_conv1d_pad_to_nc1d: Option<bool>,
    pub enable_cuda_graph: Option<bool>,
    pub enable_skip_layer_norm_strict_mode: Option<bool>,
}

impl From<OrtexCUDAExecutionProvider> for CUDAExecutionProvider {
    fn from(config: OrtexCUDAExecutionProvider) -> CUDAExecutionProvider {
        let aes = match config.arena_extend_strategy.as_deref() {
            Some("NextPowerOfTwo") => Some(ArenaExtendStrategy::NextPowerOfTwo),
            Some("SameAsRequested") => Some(ArenaExtendStrategy::NextPowerOfTwo),
            _ => None,
        };
        let ccas = match config.cudnn_conv_algo_search.as_deref() {
            Some("Exhaustive") => Some(CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive),
            Some("Heuristic") => Some(CUDAExecutionProviderCuDNNConvAlgoSearch::Heuristic),
            Some("Default") => Some(CUDAExecutionProviderCuDNNConvAlgoSearch::Default),
            _ => None,
        };
        let c = _OrtexCUDAExecutionProvider {
            device_id: config.device_id,
            gpu_mem_limit: config.gpu_mem_limit,
            arena_extend_strategy: aes,
            cudnn_conv_algo_search: ccas,
            do_copy_in_default_stream: config.do_copy_in_default_stream,
            cudnn_conv_use_max_workspace: config.cudnn_conv_use_max_workspace,
            cudnn_conv1d_pad_to_nc1d: config.cudnn_conv1d_pad_to_nc1d,
            enable_cuda_graph: config.enable_cuda_graph,
            enable_skip_layer_norm_strict_mode: config.enable_skip_layer_norm_strict_mode,
        };
        unsafe { std::mem::transmute(c) }
    }
}

#[derive(Debug, NifStruct, Clone)]
#[module = "Ortex.CPUExecutionProvider"]
pub struct OrtexCPUExecutionProvider {
    pub use_arena: Option<bool>,
}

impl From<OrtexCPUExecutionProvider> for CPUExecutionProvider {
    fn from(config: OrtexCPUExecutionProvider) -> CPUExecutionProvider {
        unsafe { std::mem::transmute(config) }
    }
}

#[derive(Debug, NifStruct, Clone)]
#[module = "Ortex.TensorRTExecutionProvider"]
pub struct OrtexTensorRTExecutionProvider {
    pub device_id: Option<u32>,
    pub max_workspace_size: Option<usize>,
    pub max_partition_iterations: Option<u32>,
    pub min_subgraph_size: Option<usize>,
    pub fp16_enable: Option<bool>,
    pub int8_enable: Option<bool>,
    pub int8_calibration_table_name: Option<String>,
    pub int8_use_native_calibration_table: Option<bool>,
    pub dla_enable: Option<bool>,
    pub dla_core: Option<u32>,
    pub engine_cache_enable: Option<bool>,
    pub engine_cache_path: Option<String>,
    pub dump_subgraphs: Option<bool>,
    pub force_sequential_engine_build: Option<bool>,
    pub enable_context_memory_sharing: Option<bool>,
    pub layer_norm_fp32_fallback: Option<bool>,
    pub timing_cache_enable: Option<bool>,
    pub force_timing_cache: Option<bool>,
    pub detailed_build_log: Option<bool>,
    pub enable_build_heuristics: Option<bool>,
    pub enable_sparsity: Option<bool>,
    pub builder_optimization_level: Option<u8>,
    pub auxiliary_streams: Option<i8>,
    pub tactic_sources: Option<String>,
    pub extra_plugin_lib_paths: Option<String>,
    pub profile_min_shapes: Option<String>,
    pub profile_max_shapes: Option<String>,
    pub profile_opt_shapes: Option<String>,
}

impl From<OrtexTensorRTExecutionProvider> for TensorRTExecutionProvider {
    fn from(config: OrtexTensorRTExecutionProvider) -> TensorRTExecutionProvider {
        // I don't like these but otherwise this conversion gets too verbose
        unsafe { std::mem::transmute(config) }
    }
}

#[rustler::nif]
fn make_cpu_ep(ep: OrtexCPUExecutionProvider) -> ResourceArc<OrtexExecutionProvider> {
    ResourceArc::new(OrtexExecutionProvider::Cpu(ep))
}

#[rustler::nif]
fn make_cuda_ep(ep: OrtexCUDAExecutionProvider) -> ResourceArc<OrtexExecutionProvider> {
    ResourceArc::new(OrtexExecutionProvider::Cuda(ep))
}

#[rustler::nif]
fn make_tensorrt_ep(ep: OrtexTensorRTExecutionProvider) -> ResourceArc<OrtexExecutionProvider> {
    ResourceArc::new(OrtexExecutionProvider::TensorRT(ep))
}

#[derive(Debug, Clone)]
pub enum OrtexExecutionProvider {
    Cpu(OrtexCPUExecutionProvider),
    Cuda(OrtexCUDAExecutionProvider),
    TensorRT(OrtexTensorRTExecutionProvider),
}

/// Takes a vec of Atoms and transforms them into a vec of ExecutionProvider Enums
pub fn register_eps(builder: &SessionBuilder, eps: Vec<ResourceArc<OrtexExecutionProvider>>) {
    for e in eps.iter() {
        match (**e).clone() {
            OrtexExecutionProvider::Cpu(config) => {
                // TODO: send these print strings back to elixir as EP status tuples
                println!("CPU Registering");
                let ep = CPUExecutionProvider::from(config);
                if ep.register(builder).is_err() {
                    eprintln!("failed to register {}!", ep.as_str())
                }
            }
            OrtexExecutionProvider::Cuda(config) => {
                println!("CUDA Registering");
                let ep = CUDAExecutionProvider::from(config);
                if ep.register(builder).is_err() {
                    eprintln!("failed to register {}!", ep.as_str())
                }
            }
            OrtexExecutionProvider::TensorRT(config) => {
                println!("TensorRT Registering");
                let ep = TensorRTExecutionProvider::from(config);
                if ep.register(builder).is_err() {
                    eprintln!("failed to register {}!", ep.as_str())
                }
            }
        }
    }
}

/// Take an optimization level and returns the
pub fn map_opt_level(opt: i32) -> GraphOptimizationLevel {
    match opt {
        1 => GraphOptimizationLevel::Level1,
        2 => GraphOptimizationLevel::Level2,
        3 => GraphOptimizationLevel::Level3,
        _ => GraphOptimizationLevel::Disable,
    }
}
