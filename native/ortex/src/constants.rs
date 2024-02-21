pub const CUDA: &str = "cuda";
pub const CPU: &str = "cpu";
pub const TENSORRT: &str = "tensorrt";
pub const ACL: &str = "acl";
pub const ONEDNN: &str = "onednn";
pub const COREML: &str = "coreml";
pub const DIRECTML: &str = "directml";
pub const ROCM: &str = "rocm";

pub mod ortex_atoms {
    rustler::atoms! {
        // Tensor types available
        s8, s16, s32, s64,
        u8, u16, u32, u64,
        f16, f32, f64,
        bf16,
        c64, c128,
        s, u, f, bf, c,
        // Execution provider atoms
        cpu, cuda, tensorrt, acl, dnnl,
        onednn, coreml, directml, rocm
    }
}
