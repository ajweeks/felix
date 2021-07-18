use ash::vk;
use std::ffi::CString;

pub struct ShaderEntryPoint {
    pub entry_point: String,
}

pub unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts((p as *const T).cast::<u8>(), ::std::mem::size_of::<T>())
}

pub struct ShaderModules {
    pub vertex_module: Option<vk::ShaderModule>,
    pub vertex_entry_point: Option<CString>,
    pub fragment_module: Option<vk::ShaderModule>,
    pub fragment_entry_point: Option<CString>,
    pub compute_module: Option<vk::ShaderModule>,
    pub compute_entry_point: Option<CString>,
}

impl ShaderModules {
    pub fn vert_frag(
        vertex_module: vk::ShaderModule,
        vertex_entry_point: CString,
        fragment_module: vk::ShaderModule,
        fragment_entry_point: CString,
    ) -> ShaderModules {
        ShaderModules {
            vertex_module: Some(vertex_module),
            vertex_entry_point: Some(vertex_entry_point),
            fragment_module: Some(fragment_module),
            fragment_entry_point: Some(fragment_entry_point),
            compute_module: None,
            compute_entry_point: None,
        }
    }

    pub fn compute(
        compute_module: vk::ShaderModule,
        compute_entry_point: CString,
    ) -> ShaderModules {
        ShaderModules {
            vertex_module: None,
            vertex_entry_point: None,
            fragment_module: None,
            fragment_entry_point: None,
            compute_module: Some(compute_module),
            compute_entry_point: Some(compute_entry_point),
        }
    }
}
