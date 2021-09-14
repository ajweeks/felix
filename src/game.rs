use glam::*;

use spirv_builder::{MetadataPrintout, SpirvBuilder};
use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
};

use std::{collections::HashMap, fs::File, path::PathBuf, rc::Rc, time::Instant};

use crate::mesh::{Mesh, read_spv};
use crate::settings;
use crate::vulkan_helpers::*;
use crate::{compute_skinning_pass::ComputeSkinningPass, mesh::AnimatedVertex, mesh::Vertex};
use crate::{render_mesh_pass::RenderMeshPass, vulkan_base::*};

use settings::*;

use ash::vk;

const NUM_DESCRIPTORS_PER_TYPE: u32 = 1024;
const NUM_DESCRIPTOR_SETS: u32 = 1024;

pub struct Camera {
    position: Vec3,
    rotation: Quat,
    starting_position: Vec3,
    starting_rotation: Quat,
}

impl Default for Camera {
    fn default() -> Camera {
        Camera {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            starting_position: Vec3::ZERO,
            starting_rotation: Quat::IDENTITY,
        }
    }
}

impl Camera {
    fn new(starting_pos: Vec3) -> Camera {
        Camera {
            position: starting_pos,
            rotation: Quat::IDENTITY,
            starting_position: starting_pos,
            starting_rotation: Quat::IDENTITY,
        }
    }

    fn reset(&mut self) {
        self.position = self.starting_position;
        self.rotation = self.starting_rotation;
    }
}

pub struct Game {
    base: VulkanBase,
    camera: Camera,
    settings_wrapper: SettingsWrapper,
    compute_skinning_pass: ComputeSkinningPass,
    static_meshes: Vec<Rc<Mesh<Vertex>>>,
    animated_meshes: Vec<Rc<Mesh<AnimatedVertex>>>,
    render_meshes: Vec<RenderMeshPass>,
    descriptor_pool: vk::DescriptorPool,
    framebuffers: Vec<vk::Framebuffer>,
    view_scissor: VkViewScissor,
    render_pass: vk::RenderPass,
}

// Inputs
#[derive(Clone, Copy)]
struct Inputs {
    is_left_clicked: bool,
    cursor_position: (i32, i32),
    wheel_delta: f32,
    keyboard_forward: i32,
    keyboard_horizontal: i32,
    keyboard_vertical: i32,
    mod_shift: bool,
    mod_ctrl: bool,
    mod_alt: bool,
}

impl Default for Inputs {
    fn default() -> Inputs {
        Inputs {
            is_left_clicked: false,
            cursor_position: (0, 0),
            wheel_delta: 0.0,
            keyboard_forward: 0,
            keyboard_horizontal: 0,
            keyboard_vertical: 0,
            mod_shift: false,
            mod_ctrl: false,
            mod_alt: false,
        }
    }
}

#[derive(Debug)]
pub struct SpvFile {
    pub name: String,
    pub data: Vec<u32>,
}

pub fn compile_shaders() -> HashMap<String, SpvFile> {
    if let Ok(read_dir) = std::fs::read_dir("assets/shaders") {
        let ignored_shader_dirs = ["shared"];

        let shaders = read_dir.map(|dir_entry| match dir_entry {
            Ok(dir_entry) => (
                dir_entry.file_name().into_string().unwrap(),
                dir_entry.path(),
            ),
            Err(_) => (String::new(), PathBuf::default()),
        });

        let mut spv_files = HashMap::<String, SpvFile>::new();
        for (shader_name, path) in shaders {
            if shader_name.is_empty() || ignored_shader_dirs.contains(&shader_name.as_str()) {
                continue;
            }

            let result = SpirvBuilder::new(path, "spirv-unknown-spv1.3")
                .print_metadata(MetadataPrintout::None)
                .build();

            match result {
                Ok(result) => {
                    let path = result.module.unwrap_single().to_path_buf();

                    let data = read_spv(&mut File::open(path).unwrap()).unwrap();

                    spv_files.insert(
                        shader_name.clone(),
                        SpvFile {
                            name: shader_name,
                            data,
                        },
                    );
                }
                Err(err) => {
                    eprintln!("Failed to build shader `{}`, error: {}", shader_name, err);
                }
            }
        }
        spv_files
    } else {
        HashMap::new()
    }
}

impl Game {
    pub fn new(window: &crate::Window) -> Game {
        // Vulkan base initialization
        let base = VulkanBase::new(&window.window, window.width, window.height);

        // Render passes
        let render_pass_attachments = [
            vk::AttachmentDescription {
                format: base.surface_format.format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                ..Default::default()
            },
            vk::AttachmentDescription {
                format: vk::Format::D32_SFLOAT,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ..Default::default()
            },
        ];
        let color_attachment_refs = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        }];

        let subpasses = [vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .build()];

        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&render_pass_attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = unsafe {
            base.device
                .create_render_pass(&render_pass_create_info, None)
        }
        .unwrap();

        let framebuffers: Vec<vk::Framebuffer> = base
            .present_image_views
            .iter()
            .map(|&present_image_view| {
                let framebuffer_attachments = [present_image_view, base.depth_image_view];
                let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&framebuffer_attachments)
                    .width(base.surface_resolution.width)
                    .height(base.surface_resolution.height)
                    .layers(1);

                unsafe {
                    base.device
                        .create_framebuffer(&frame_buffer_create_info, None)
                }
                .unwrap()
            })
            .collect();

        let view_scissor = {
            let viewport = vk::Viewport {
                x: 0.0,
                y: base.surface_resolution.height as f32, // Flip viewport
                width: base.surface_resolution.width as f32,
                height: -(base.surface_resolution.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                extent: base.surface_resolution,
                ..Default::default()
            };
            VkViewScissor { viewport, scissor }
        };

        // Descriptor pool
        let descriptor_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: NUM_DESCRIPTORS_PER_TYPE,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: NUM_DESCRIPTORS_PER_TYPE,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: NUM_DESCRIPTORS_PER_TYPE,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: NUM_DESCRIPTORS_PER_TYPE,
            },
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&descriptor_sizes)
            .max_sets(NUM_DESCRIPTOR_SETS);

        let descriptor_pool = unsafe {
            base.device
                .create_descriptor_pool(&descriptor_pool_info, None)
        }
        .unwrap();

        let shader_files = compile_shaders();

        let static_meshes_to_load = ["spring-extended.glb"];
        let animated_meshes_to_load = ["skinned-box.glb"];

        let mut static_meshes = Vec::new();
        let mut animated_meshes = Vec::new();

        for mesh_path in static_meshes_to_load {
            let start = Instant::now();
            static_meshes.push(Rc::new(Mesh::load_gltf_mesh(mesh_path)));
            let duration = Instant::now() - start;
            println!("Mesh load completed in {:.2}s", duration.as_secs_f32());
        }
        
        for mesh_path in animated_meshes_to_load {
            let start = Instant::now();
            animated_meshes.push(Rc::new(Mesh::load_gltf_mesh(mesh_path)));
            let duration = Instant::now() - start;
            println!("Mesh load completed in {:.2}s", duration.as_secs_f32());
        }

        eprintln!("Num shader files loaded: {}", shader_files.len());

        let compute_skinning_pass = ComputeSkinningPass::new(
            &base.device,
            &base.allocator,
            &descriptor_pool,
            Rc::downgrade(&animated_meshes[0]),
            &shader_files["compute_skinning"],
        );

        let render_meshes = vec![RenderMeshPass::new(
            &base.device,
            &base.allocator,
            &descriptor_pool,
            &render_pass,
            &view_scissor,
            &static_meshes[0],
            &shader_files["pbr"],
        )];

        // Submit initialization command buffer before rendering starts
        base.record_submit_commandbuffer(
            0,
            base.present_queue,
            &[],
            &[],
            &[],
            |device, command_buffer| {
                compute_skinning_pass.gpu_setup(device, &command_buffer);

                for mesh in &render_meshes {
                    mesh.gpu_setup(device, &command_buffer);
                }
            },
        );

        Game {
            base,
            camera: Camera::new(Vec3::new(0.0, 6.0, -7.0)),
            settings_wrapper: SettingsWrapper::create_from_file(),
            static_meshes,
            animated_meshes,
            compute_skinning_pass,
            render_meshes,
            descriptor_pool,
            framebuffers,
            view_scissor,
            render_pass,
        }
    }

    pub fn run_update_loop(&mut self, window: &crate::Window, event_loop: &mut EventLoop<()>) {
        println!("Start window event loop");

        let mut inputs_prev: Inputs = Default::default();
        let mut inputs: Inputs = Default::default();

        let app_start_time = Instant::now();
        let mut last_title_update_time = app_start_time;
        let mut frame = 0u32;
        let mut frames_since_title_update = 0u32;
        let mut active_command_buffer = 0;

        event_loop.run_return(|event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::NewEvents(_) => {
                    inputs.wheel_delta = 0.0;
                }

                Event::MainEventsCleared => {
                    if self.settings_wrapper.is_out_of_date() {
                        self.settings_wrapper.reload();
                    }

                    let cursor_delta = (
                        inputs.cursor_position.0 - inputs_prev.cursor_position.0,
                        inputs.cursor_position.1 - inputs_prev.cursor_position.1,
                    );

                    let settings = &self.settings_wrapper.settings;

                    inputs_prev = inputs;

                    // Update camera based in inputs
                    let camera_forward = self.camera.rotation * Vec3::Z;
                    let camera_right = self.camera.rotation * Vec3::X;
                    let camera_up = self.camera.rotation * Vec3::Y;

                    let mut move_speed = settings.move_speed;
                    if inputs.mod_shift {
                        move_speed *= settings.mod_faster_speed;
                    }
                    if inputs.mod_ctrl {
                        move_speed *= settings.mod_slower_speed;
                    }

                    let forward_speed = inputs.wheel_delta * settings.dolly_speed
                        + inputs.keyboard_forward as f32 * move_speed;
                    let horizontal_speed = inputs.keyboard_horizontal as f32 * move_speed;
                    let vertical_speed = inputs.keyboard_vertical as f32 * move_speed;

                    self.camera.position += camera_forward * forward_speed
                        + camera_right * horizontal_speed
                        + camera_up * vertical_speed;

                    if inputs.is_left_clicked {
                        let delta_yaw = cursor_delta.0 as f32 * settings.mouse_sensitivity;
                        let delta_roll = cursor_delta.1 as f32 * settings.mouse_sensitivity;

                        if inputs.mod_alt {
                            // Orbit
                            let look_at_center = Vec3::ZERO;
                            let orbit_radius = (self.camera.position - look_at_center).length();
                            self.camera.position = look_at_center
                                + (self.camera.position - look_at_center
                                    + (Vec3::X * (delta_yaw * 10.0)))
                                    .normalize()
                                    * orbit_radius;
                            //let delta = (self.camera.position - look_at_center).normalize();
                            //let f = Vec3::Z.dot(-delta);
                            //self.camera.rotation = if (f - -1.0).abs() < 0.00001 {
                            //    Quat::from_xyzw(0.0, 1.0, 0.0, std::f32::consts::PI)
                            //} else if (f - 1.0).abs() < 0.00001 {
                            //    Quat::IDENTITY
                            //} else {
                            //    let target_angle = f.acos();
                            //    let rot_axis = Vec3::Z.cross(delta);
                            //    Quat::from_axis_angle(rot_axis, target_angle)
                            //}
                        } else {
                            self.camera.rotation = Quat::from_axis_angle(camera_right, delta_roll)
                                * Quat::from_axis_angle(Vec3::Y, delta_yaw)
                                * self.camera.rotation;
                        }
                    }

                    // Render
                    let (present_index, _) = unsafe {
                        self.base.swapchain_loader.acquire_next_image(
                            self.base.swapchain,
                            std::u64::MAX,
                            self.base.present_complete_semaphore,
                            vk::Fence::null(),
                        )
                    }
                    .unwrap();

                    // Update uniform buffer
                    let elapsed_sec = (Instant::now() - app_start_time).as_secs_f32();
                    let color = Vec4::new(
                        1.0 - (elapsed_sec + 0.9).sin(),
                        (elapsed_sec * 2.0).sin(),
                        (0.5 + elapsed_sec * 1.5).sin(),
                        0.0,
                    );

                    let v = Mat4::look_at_lh(
                        self.camera.position,
                        self.camera.position + self.camera.rotation * Vec3::Z,
                        Vec3::Y,
                    );
                    // Reverse Z
                    let z_near = 100000.0;
                    let z_far = 0.1;
                    let p = Mat4::perspective_lh(
                        std::f32::consts::FRAC_PI_2,
                        window.width as f32 / window.height as f32,
                        z_near,
                        z_far,
                    );
                    let world_to_screen = p * v;

                    self.compute_skinning_pass.update();

                    for mesh in &self.render_meshes {
                        mesh.update();
                    }

                    // Setup render passs
                    let clear_values = [
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.075, 0.1, 0.15, 0.0],
                            },
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 0.0,
                                stencil: 0,
                            },
                        },
                    ];

                    let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                        .render_pass(self.render_pass)
                        .framebuffer(self.framebuffers[present_index as usize])
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: self.base.surface_resolution,
                        })
                        .clear_values(&clear_values);

                    // Submit main command buffer
                    active_command_buffer = self.base.record_submit_commandbuffer(
                        active_command_buffer,
                        self.base.present_queue,
                        &[vk::PipelineStageFlags::BOTTOM_OF_PIPE],
                        &[self.base.present_complete_semaphore],
                        &[self.base.rendering_complete_semaphore],
                        |device, command_buffer| {
                            // Compute pass

                            let mut inv_bone_poses = [0.0; 16 * shared::MAX_NUM_JOINTS];
                            for (i, bone) in self.animated_meshes[0].bone_inv_bind_poses.iter().enumerate() {
                                let bone_mat: [f32; 16] = bone.to_cols_array();
                                for j in 0..16 {
                                    inv_bone_poses[i * 16 + j] = bone_mat[j];
                                }
                            }

                            // Update compute skinning constants
                            let skinning_constants = shared::SkinningConstants { t: 0.0 };

                            self.compute_skinning_pass.dispatch(
                                &skinning_constants,
                                device,
                                &command_buffer,
                            );

                            // Render pass
                            unsafe {
                                device.cmd_begin_render_pass(
                                    command_buffer,
                                    &render_pass_begin_info,
                                    vk::SubpassContents::INLINE,
                                );
                                device.cmd_set_viewport(
                                    command_buffer,
                                    0,
                                    &[self.view_scissor.viewport],
                                );
                                device.cmd_set_scissor(
                                    command_buffer,
                                    0,
                                    &[self.view_scissor.scissor],
                                );
                            }

                            // Draw meshes (main render pass)
                            for mesh in &self.render_meshes {
                                let shader_constants = shared::MeshShaderConstants {
                                    world_to_screen: world_to_screen.to_cols_array(),
                                    object_to_world: mesh.object_to_world.to_cols_array(),
                                    color: color.into(),
                                    time: elapsed_sec,
                                    // camera_position: Vec4::from((camera.position, 1.0)),
                                };

                                mesh.draw_main_render_pass(
                                    &shader_constants,
                                    device,
                                    &command_buffer,
                                );
                            }

                            unsafe {
                                device.cmd_end_render_pass(command_buffer);
                            }
                        },
                    );

                    // Present frame
                    let present_info = vk::PresentInfoKHR {
                        wait_semaphore_count: 1,
                        p_wait_semaphores: &self.base.rendering_complete_semaphore,
                        swapchain_count: 1,
                        p_swapchains: &self.base.swapchain,
                        p_image_indices: &present_index,
                        ..Default::default()
                    };

                    unsafe {
                        self.base
                            .swapchain_loader
                            .queue_present(self.base.present_queue, &present_info)
                    }
                    .unwrap();

                    // Output performance info every 60 frames
                    frame += 1;
                    frames_since_title_update += 1;
                    let time_now = Instant::now();
                    let time_since_title_update =
                        (time_now - last_title_update_time).as_millis() as f32;
                    if time_since_title_update > 500.0 {
                        window.window.set_title(
                            format!(
                                "{} - {:.2} ms",
                                crate::WINDOW_TITLE_PREFIX,
                                time_since_title_update / frames_since_title_update as f32
                            )
                            .as_str(),
                        );

                        last_title_update_time = time_now;
                        frames_since_title_update = 0;
                    }
                }

                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,

                    // TODO: Handle swapchain resize
                    WindowEvent::Resized { .. } => {}

                    // Keyboard
                    WindowEvent::KeyboardInput { input, .. } => {
                        let pressed = input.state == ElementState::Pressed;

                        if input.virtual_keycode == Some(VirtualKeyCode::LShift) {
                            inputs.mod_shift = pressed;
                        }

                        if input.virtual_keycode == Some(VirtualKeyCode::LControl) {
                            inputs.mod_ctrl = pressed;
                        }

                        if input.virtual_keycode == Some(VirtualKeyCode::LAlt) {
                            inputs.mod_alt = pressed;
                        }

                        if input.virtual_keycode == Some(VirtualKeyCode::R) {
                            self.camera.reset();
                        }

                        if input.virtual_keycode == Some(VirtualKeyCode::W) {
                            inputs.keyboard_forward = if pressed { 1 } else { 0 };
                        } else if input.virtual_keycode == Some(VirtualKeyCode::S) {
                            inputs.keyboard_forward = if pressed { -1 } else { 0 };
                        }

                        if input.virtual_keycode == Some(VirtualKeyCode::D) {
                            inputs.keyboard_horizontal = if pressed { 1 } else { 0 };
                        } else if input.virtual_keycode == Some(VirtualKeyCode::A) {
                            inputs.keyboard_horizontal = if pressed { -1 } else { 0 };
                        }

                        if input.virtual_keycode == Some(VirtualKeyCode::E) {
                            inputs.keyboard_vertical = if pressed { 1 } else { 0 };
                        } else if input.virtual_keycode == Some(VirtualKeyCode::Q) {
                            inputs.keyboard_vertical = if pressed { -1 } else { 0 };
                        }
                        if pressed
                            && inputs.mod_ctrl
                            && input.virtual_keycode == Some(VirtualKeyCode::S)
                        {
                            let result = self.settings_wrapper.serialize();
                            if result.is_ok() {
                                println!("saved");
                            }
                        }
                    }

                    // Mouse
                    WindowEvent::MouseInput {
                        button: MouseButton::Left,
                        state,
                        ..
                    } => {
                        inputs.is_left_clicked = state == ElementState::Pressed;
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let position: (i32, i32) = position.into();
                        inputs.cursor_position = position;
                    }
                    WindowEvent::MouseWheel {
                        delta: MouseScrollDelta::LineDelta(_, v_lines),
                        ..
                    } => {
                        inputs.wheel_delta += v_lines;
                    }
                    _ => (),
                },

                Event::LoopDestroyed => unsafe { self.base.device.device_wait_idle() }.unwrap(),
                _ => (),
            }
        });

        self.cleanup();
    }

    fn cleanup(&self) {
        self.compute_skinning_pass
            .destroy(&self.base.device, &self.base.allocator);

        for mesh in &self.render_meshes {
            mesh.destroy(&self.base.device, &self.base.allocator);
        }

        unsafe {
            self.base
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for framebuffer in &self.framebuffers {
                self.base.device.destroy_framebuffer(*framebuffer, None);
            }
            self.base.device.destroy_render_pass(self.render_pass, None);
        }
    }
}
