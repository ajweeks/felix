use glam::{Mat4, Quat, Vec3, Vec4};
use gltf::Primitive;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::Cursor;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::{fs, io, slice};

use byteorder::ByteOrder;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use glam::*;
use gltf::animation::Property;
use gltf::Glb;
use gltf::Gltf;

use crate::anim::*;

pub trait IVertex: Default + Clone {
    const ATTRIBUTES: u32;

    fn set_from_buffer(
        &mut self,
        vertex_index: usize,
        positions: &Option<Vec<[f32; 3]>>,
        normals: &Option<Vec<[f32; 3]>>,
        tex_coords: &Option<Vec<[f32; 2]>>,
        colours: &Option<Vec<[f32; 4]>>,
        joint_indices: &Option<Vec<[u32; 4]>>,
        joint_weights: &Option<Vec<[f32; 4]>>,
    );
}

#[derive(Clone, Copy)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub colour: [f32; 4],
}

impl Default for Vertex {
    fn default() -> Vertex {
        Vertex {
            pos: [0.0; 3],
            normal: [0.0; 3],
            uv: [0.0; 2],
            colour: [0.0; 4],
        }
    }
}

impl Debug for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{},{},{} - ",
            self.pos[0], self.pos[1], self.pos[2]
        ))?;
        f.write_fmt(format_args!(
            "{},{},{} - ",
            self.normal[0], self.normal[1], self.normal[2]
        ))?;
        f.write_fmt(format_args!("{},{},", self.uv[0], self.uv[1]))?;
        f.write_fmt(format_args!(
            "{},{},{},{} - ",
            self.colour[0], self.colour[1], self.colour[2], self.colour[3]
        ))?;

        Ok(())
    }
}

impl IVertex for Vertex {
    const ATTRIBUTES: u32 = VertexAttribute::Position as u32
        | VertexAttribute::Normal as u32
        | VertexAttribute::TexCoord0 as u32
        | VertexAttribute::Color0 as u32;

    fn set_from_buffer(
        &mut self,
        vertex_index: usize,
        positions: &Option<Vec<[f32; 3]>>,
        normals: &Option<Vec<[f32; 3]>>,
        tex_coords: &Option<Vec<[f32; 2]>>,
        colours: &Option<Vec<[f32; 4]>>,
        _joint_indices: &Option<Vec<[u32; 4]>>,
        _joint_weights: &Option<Vec<[f32; 4]>>,
    ) {
        self.pos = if let Some(positions) = positions.as_ref() {
            positions[vertex_index]
        } else {
            [0.0; 3]
        };
        self.normal = if let Some(normals) = normals.as_ref() {
            normals[vertex_index]
        } else {
            [0.0; 3]
        };
        self.uv = if let Some(tex_coords) = tex_coords.as_ref() {
            tex_coords[vertex_index]
        } else {
            [0.0; 2]
        };
        self.colour = if let Some(colours) = colours.as_ref() {
            colours[vertex_index]
        } else {
            [0.0; 4]
        };
    }
}

#[derive(Clone, Copy)]
pub struct AnimatedVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub colour: [f32; 4],
    pub joint_indices: [u32; 4],
    pub joint_weights: [f32; 4],
}

impl Default for AnimatedVertex {
    fn default() -> AnimatedVertex {
        AnimatedVertex {
            pos: [0.0; 3],
            normal: [0.0; 3],
            uv: [0.0; 2],
            colour: [0.0; 4],
            joint_indices: [0; 4],
            joint_weights: [0.0; 4],
        }
    }
}

impl Debug for AnimatedVertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{},{},{} - ",
            self.pos[0], self.pos[1], self.pos[2]
        ))?;
        f.write_fmt(format_args!(
            "{},{},{} - ",
            self.normal[0], self.normal[1], self.normal[2]
        ))?;
        f.write_fmt(format_args!("{},{}, - ", self.uv[0], self.uv[1]))?;
        f.write_fmt(format_args!(
            "{},{},{},{} - ",
            self.colour[0], self.colour[1], self.colour[2], self.colour[3]
        ))?;
        f.write_fmt(format_args!(
            "{},{},{},{} - ",
            self.joint_indices[0],
            self.joint_indices[1],
            self.joint_indices[2],
            self.joint_indices[3]
        ))?;
        f.write_fmt(format_args!(
            "{},{},{},{}",
            self.joint_weights[0],
            self.joint_weights[1],
            self.joint_weights[2],
            self.joint_weights[3]
        ))?;

        Ok(())
    }
}

impl IVertex for AnimatedVertex {
    const ATTRIBUTES: u32 = VertexAttribute::Position as u32
        | VertexAttribute::Normal as u32
        | VertexAttribute::TexCoord0 as u32
        | VertexAttribute::Color0 as u32
        | VertexAttribute::JointIndices as u32
        | VertexAttribute::JointWeights as u32;

    fn set_from_buffer(
        &mut self,
        vertex_index: usize,
        positions: &Option<Vec<[f32; 3]>>,
        normals: &Option<Vec<[f32; 3]>>,
        tex_coords: &Option<Vec<[f32; 2]>>,
        colours: &Option<Vec<[f32; 4]>>,
        joint_indices: &Option<Vec<[u32; 4]>>,
        joint_weights: &Option<Vec<[f32; 4]>>,
    ) {
        self.pos = if let Some(positions) = positions.as_ref() {
            positions[vertex_index]
        } else {
            [0.0; 3]
        };
        self.normal = if let Some(normals) = normals.as_ref() {
            normals[vertex_index]
        } else {
            [0.0; 3]
        };
        self.uv = if let Some(tex_coords) = tex_coords.as_ref() {
            tex_coords[vertex_index]
        } else {
            [0.0; 2]
        };
        self.colour = if let Some(colours) = colours.as_ref() {
            colours[vertex_index]
        } else {
            [0.0; 4]
        };
        self.joint_indices = if let Some(joint_indices) = joint_indices.as_ref() {
            joint_indices[vertex_index]
        } else {
            [0; 4]
        };
        self.joint_weights = if let Some(joint_weights) = joint_weights.as_ref() {
            joint_weights[vertex_index]
        } else {
            [0.0; 4]
        };
    }
}

#[repr(u32)]
pub enum VertexAttribute {
    Position = 1 << 0,
    Normal = 1 << 1,
    TexCoord0 = 1 << 2,
    Color0 = 1 << 3,
    JointIndices = 1 << 4,
    JointWeights = 1 << 5,
}

#[derive(Debug)]
pub struct VertexBuffer<VertexType: IVertex + Clone + Default> {
    pub vertices: Vec<VertexType>,
    pub vertex_count: u32,
}

impl<VertexType: IVertex + Clone + Default + 'static> VertexBuffer<VertexType> {
    fn new(
        positions: Option<Vec<[f32; 3]>>,
        normals: Option<Vec<[f32; 3]>>,
        tex_coords: Option<Vec<[f32; 2]>>,
        colours: Option<Vec<[f32; 4]>>,
        joint_indices: Option<Vec<[u32; 4]>>,
        joint_weights: Option<Vec<[f32; 4]>>,
    ) -> VertexBuffer<VertexType> {
        let mut vertex_count: usize = 0;
        // let mut attributes = 0;
        if let Some(positions) = &positions {
            // attributes |= VertexAttribute::Position as u32;
            vertex_count = positions.len();
        }
        // if normals.is_some() {
        //     attributes |= VertexAttribute::Normal as u32;
        // }
        // if tex_coords.is_some() {
        //     attributes |= VertexAttribute::TexCoord0 as u32;
        // }
        // if colours.is_some() {
        //     attributes |= VertexAttribute::Color0 as u32;
        // }
        // if joint_indices.is_some() {
        //     attributes |= VertexAttribute::JointIndices as u32;
        // }
        // if joint_weights.is_some() {
        //     attributes |= VertexAttribute::JointWeights as u32;
        // }

        // println!("{}, {}", attributes , VertexType::ATTRIBUTES);
        // assert!(attributes == VertexType::ATTRIBUTES);

        // let stride = calculate_stride(VertexType::ATTRIBUTES);

        let mut buffer = VertexBuffer::<VertexType> {
            vertices: Vec::new(),
            vertex_count: vertex_count as u32,
        };

        buffer
            .vertices
            .resize(vertex_count as usize, Default::default());

        for i in 0..vertex_count {
            VertexType::set_from_buffer(
                &mut buffer.vertices[i],
                i,
                &positions,
                &normals,
                &tex_coords,
                &colours,
                &joint_indices,
                &joint_weights,
            );
        }

        //let size_of_f32 = std::mem::size_of::<f32>();
        //let size_of_u32 = std::mem::size_of::<u32>();

        //let mut offset: usize = 0;
        // for i in 0..vertex_count {
        //     if let Some(positions) = &positions {
        //         //let buffer_ptr: *mut u32 = &mut buffer.buffer[buffer_offset];
        //         let stride = size_of_f32 * 3;
        //         unsafe {
        //             buffer.copy_from_slice(offset, &positions[i]);
        //         }
        //         offset += stride;
        //     }
        //     if let Some(normals) = &normals {
        //         let stride = size_of_f32 * 3;
        //         unsafe {
        //             buffer.copy_from_slice(offset, &normals[i]);
        //         }
        //         offset += stride;
        //     }
        //     if let Some(tex_coords) = &tex_coords {
        //         let stride = size_of_f32 * 2;
        //         unsafe {
        //             buffer.copy_from_slice(offset, &tex_coords[i]);
        //         }
        //         offset += stride;
        //     }
        //     if let Some(colours) = &colours {
        //         let stride = size_of_f32 * 4;
        //         unsafe {
        //             buffer.copy_from_slice(offset, &colours[i]);
        //         }
        //         offset += stride;
        //     }
        //     if let Some(joint_indices) = &joint_indices {
        //         let stride = size_of_u32 * 4;
        //         unsafe {
        //             buffer.copy_from_slice(offset, &joint_indices[i]);
        //         }
        //         offset += stride;
        //     }
        //     if let Some(joint_weights) = &joint_weights {
        //         let stride = size_of_f32 * 4;
        //         unsafe {
        //             buffer.copy_from_slice(offset, &joint_weights[i]);
        //         }
        //         offset += stride;
        //     }
        // }

        buffer
    }

    // unsafe fn copy_from_slice<T>(&mut self, offset: usize, slice: &[T]) {
    //     std::ptr::copy_nonoverlapping(
    //         slice.as_ptr() as *const u8,
    //         self.buffer[offset] as *mut u8,
    //         slice.len(),
    //     );
    // }
}

fn calculate_stride(attributes: u32) -> u32 {
    let mut result: u32 = 0;

    let size_of_f32 = std::mem::size_of::<f32>() as u32;

    if attributes & VertexAttribute::Position as u32 != 0 {
        result += size_of_f32 * 3;
    }
    if attributes & VertexAttribute::Normal as u32 != 0 {
        result += size_of_f32 * 3;
    }
    if attributes & VertexAttribute::TexCoord0 as u32 != 0 {
        result += size_of_f32 * 2;
    }
    if attributes & VertexAttribute::Color0 as u32 != 0 {
        result += size_of_f32 * 4;
    }

    result
}

pub struct SubMesh<VertexType: IVertex + Clone + Default + 'static> {
    pub vertex_buffer: VertexBuffer<VertexType>,
    pub index_buffer: Option<Vec<u32>>,
}

pub struct Mesh<VertexType: IVertex + Clone + Default + 'static> {
    pub submeshes: Vec<SubMesh<VertexType>>,
    pub bone_inv_bind_poses: Vec<Mat4>,
    pub anim_clips: Vec<AnimClipCPU>,
    pub active_clip: i32,
}

impl<VertexType: IVertex + Clone + Default> Mesh<VertexType> {
    pub fn load_gltf_mesh(mesh_path: &str) -> Mesh<VertexType>
    where
        VertexType: Debug,
    {
        let path = format!(
            "{}/assets/meshes/{}",
            std::env::current_dir().unwrap().to_str().unwrap(),
            mesh_path
        );
        println!("Loading mesh {}...", path);

        let gltf = Gltf::open(&path).expect("Failed to load mesh");
        let glb = Glb::from_reader(fs::File::open(path).expect("Failed to load mesh"))
            .expect("Failed to load mesh");

        // println!("Mesh has {} scene(s)", gltf.scenes().count());

        let mut submeshes = Vec::new();

        let mut bone_inv_bind_poses = Vec::new();

        for scene in gltf.scenes() {
            //println!(
            //    "Scene {} has {} node(s)",
            //    scene.index(),
            //    scene.nodes().count()
            //);
            for node in scene.nodes() {
                // TODO: unmut
                let mut bytes = Cursor::new(glb.bin.as_ref().unwrap());

                println!("Skin: {}", node.skin().is_some());
                if let Some(skin) = node.skin() {
                    // if let Some(accessor) = skin.inverse_bind_matrices() {
                    //     let view = accessor.view().unwrap();
                    //     const STRIDE: usize = 16 * 4;
                    //     bytes.seek(SeekFrom::Start((view.offset()) as u64)).unwrap();
                    //     let mut element_bytes = [0; STRIDE];
                    //     for _ in 0..accessor.count() {
                    //         bytes.read_exact(&mut element_bytes).unwrap();
                    //         let mut mat_values = [[0.0; 4]; 4];
                    //         let mut j = 0;
                    //         for i in 0..4 {
                    //             let x = LittleEndian::read_f32(&element_bytes[(j + 0)..(j + 4)]);
                    //             let y = LittleEndian::read_f32(&element_bytes[(j + 4)..(j + 8)]);
                    //             let z = LittleEndian::read_f32(&element_bytes[(j + 8)..(j + 12)]);
                    //             let w = LittleEndian::read_f32(&element_bytes[(j + 12)..(j + 16)]);
                    //             mat_values[i] = [x, y, z, w];
                    //             j += 16;
                    //         }
                    //         bone_poses.push(Bone::from(Mat4::from_cols(
                    //             Vec4::from(mat_values[0]),
                    //             Vec4::from(mat_values[1]),
                    //             Vec4::from(mat_values[2]),
                    //             Vec4::from(mat_values[3]),
                    //         )));
                    //         println!(
                    //             "bone poses: {}",
                    //             &bone_poses[bone_poses.len() - 1].transform
                    //         );
                    //     }
                    // }

                    println!("Bone count: {}\nBones:", skin.joints().count());
                    for joint in skin.joints() {
                        let matrix = joint.transform().matrix();

                        println!(
                            "\t{}:\n\t{:?}\n\t{:?}\n\t{:?}\n\t{:?}",
                            joint.name().unwrap(),
                            matrix[0],
                            matrix[1],
                            matrix[2],
                            matrix[3]
                        );

                        bone_inv_bind_poses.push(Mat4::from_cols(
                            Vec4::from(matrix[0]),
                            Vec4::from(matrix[1]),
                            Vec4::from(matrix[2]),
                            Vec4::from(matrix[3]),
                        ));
                    }
                }

                if let Some(mesh) = node.mesh() {
                    //println!(
                    //    "Mesh: {} has {} primitive(s)",
                    //    mesh.name().unwrap_or("Empty"),
                    //    mesh.primitives().count()
                    //);

                    for primitive in mesh.primitives() {
                        //let indices = primitive.indices().unwrap();
                        //let bb = primitive.bounding_box();
                        //println!("Index count: {}", indices.count());
                        //println!("Bounding box {:?}", bb);
                        // println!("Mesh: has {} attributes", primitive.attributes().count());

                        let vertex_count =
                            primitive.get(&gltf::Semantic::Positions).unwrap().count();

                        // println!("Mesh has {} verts", vertex_count);

                        let positions = parse_float3_attribute(
                            &mut bytes,
                            &gltf::Semantic::Positions,
                            vertex_count,
                            &primitive,
                        );
                        let normals = parse_float3_attribute(
                            &mut bytes,
                            &gltf::Semantic::Normals,
                            vertex_count,
                            &primitive,
                        );
                        let texcoords = parse_float2_attribute(
                            &mut bytes,
                            &gltf::Semantic::TexCoords(0),
                            vertex_count,
                            &primitive,
                        );
                        let colours = parse_float4_attribute(
                            &mut bytes,
                            &gltf::Semantic::Colors(0),
                            vertex_count,
                            &primitive,
                        );
                        let index_buffer = parse_indices(&mut bytes, &primitive);

                        let joint_indices =
                            parse_joint_indices(&mut bytes, vertex_count, &primitive);
                        let joint_weights = parse_float4_attribute(
                            &mut bytes,
                            &gltf::Semantic::Weights(0),
                            vertex_count,
                            &primitive,
                        );

                        let vertex_buffer = VertexBuffer::new(
                            positions,
                            normals,
                            texcoords,
                            colours,
                            joint_indices,
                            joint_weights,
                        );



                        std::fs::write("buffer.txt", format!("{:#?}", vertex_buffer)).unwrap();


                        
                        submeshes.push(SubMesh {
                            vertex_buffer,
                            index_buffer,
                        })
                    }
                }
            }
        }

        let pause = || {
            println!("Enter char to continue...");
            let mut s = String::new();
            std::io::stdin().read_line(&mut s).expect("");
        };

        let mut bytes = Cursor::new(glb.bin.as_ref().unwrap());
        let mut anim_clips = Vec::new();

        println!("anim count: {:?}", gltf.animations().len());

        pause();

        for anim in gltf.animations() {
            let mut anim_clip = AnimClipCPU::default();
            let mut anim_channels: HashMap<String, AnimClipCPU> = HashMap::new();

            let mut channel_lookup = Vec::new();

            let channel_count = anim.channels().count();
            println!("anim channel count: {}", channel_count);
            anim_clip.channel_count = channel_count;

            for channel in anim.channels() {
                let target = channel.target();

                let channel_name = target.node().name().unwrap().to_owned();

                channel_lookup.push((target.property(), channel_name.clone()));

                let anim_clip = anim_channels.entry(channel_name).or_default();
                match target.property() {
                    Property::Translation => {
                        if anim_clip.positions.is_none() {
                            anim_clip.positions = Some(Vec::new());
                        }
                    }
                    Property::Rotation => {
                        if anim_clip.rotations.is_none() {
                            anim_clip.rotations = Some(Vec::new());
                        }
                    }
                    Property::Scale => {
                        if anim_clip.scales.is_none() {
                            anim_clip.scales = Some(Vec::new());
                        }
                    }
                    _ => {}
                }
            }

            let frame_count = anim.samplers().next().unwrap().output().count();
            println!("anim frame count: {}", frame_count);
            anim_clip.frame_count = frame_count;

            for (sampler_index, sampler) in anim.samplers().enumerate() {
                const VEC3_STRIDE: usize = 3 * 4;
                const VEC4_STRIDE: usize = 4 * 4;

                let accessor = sampler.output();
                let view = accessor.view().unwrap();
                bytes.seek(SeekFrom::Start((view.offset()) as u64)).unwrap();
                let mut vec3_element_bytes = [0; VEC3_STRIDE];
                let mut vec4_element_bytes = [0; VEC4_STRIDE];

                let (sample_property, sample_bone_name) =
                    channel_lookup.get(sampler_index).unwrap();

                if let Some(anim_clip) = anim_channels.get_mut(sample_bone_name) {
                    for _ in 0..accessor.count() {
                        match *sample_property {
                            Property::Translation | Property::Scale => {
                                bytes.read_exact(&mut vec3_element_bytes).unwrap();
                                let x = LittleEndian::read_f32(&vec3_element_bytes[0..4]);
                                let y = LittleEndian::read_f32(&vec3_element_bytes[4..8]);
                                let z = LittleEndian::read_f32(&vec3_element_bytes[8..12]);
                                anim_clip.add_vec3(Vec3::new(x, y, z), *sample_property);
                                println!("{:?}, {:?}", *sample_property, [x, y, z]);
                            }
                            Property::Rotation => {
                                bytes.read_exact(&mut vec4_element_bytes).unwrap();
                                let x = LittleEndian::read_f32(&vec4_element_bytes[0..4]);
                                let y = LittleEndian::read_f32(&vec4_element_bytes[4..8]);
                                let z = LittleEndian::read_f32(&vec4_element_bytes[8..12]);
                                let w = LittleEndian::read_f32(&vec4_element_bytes[12..16]);
                                anim_clip.add_quat(Quat::from_xyzw(x, y, z, w), *sample_property);
                                println!("{:?}, {:?}", *sample_property, [x, y, z, w]);
                            }
                            _ => {}
                        }
                    }
                }
            }

            anim_clips.push(anim_clip);
        }

        Mesh {
            submeshes,
            bone_inv_bind_poses,
            anim_clips,
            active_clip: 0,
        }
    }
}

pub fn read_spv<R: io::Read + io::Seek>(x: &mut R) -> io::Result<Vec<u32>> {
    let size = x.seek(io::SeekFrom::End(0))?;
    if size % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input length not divisible by 4",
        ));
    }
    if size > usize::max_value() as u64 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "input too long"));
    }
    let words = (size / 4) as usize;
    let mut result = Vec::<u32>::with_capacity(words);
    x.seek(io::SeekFrom::Start(0))?;
    unsafe {
        x.read_exact(slice::from_raw_parts_mut(
            result.as_mut_ptr() as *mut u8,
            words * 4,
        ))?;
        result.set_len(words);
    }
    const MAGIC_NUMBER: u32 = 0x07230203;
    if !result.is_empty() && result[0] == MAGIC_NUMBER.swap_bytes() {
        for word in &mut result {
            *word = word.swap_bytes();
        }
    }
    if result.is_empty() || result[0] != MAGIC_NUMBER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input missing SPIR-V magic number",
        ));
    }
    Ok(result)
}

fn parse_float4_attribute(
    bytes: &mut Cursor<&Cow<[u8]>>,
    semantic: &gltf::Semantic,
    vertex_count: usize,
    primitive: &Primitive,
) -> Option<Vec<[f32; 4]>> {
    let mut output = Vec::new();
    output.reserve(vertex_count);

    if let Some(accessor) = primitive.get(semantic) {
        let view = accessor.view().unwrap();
        const STRIDE: usize = 4 * 4;
        bytes.seek(SeekFrom::Start((view.offset()) as u64)).unwrap();
        let mut element_bytes = [0; STRIDE];
        for _ in 0..accessor.count() {
            bytes.read_exact(&mut element_bytes).unwrap();
            let x = LittleEndian::read_f32(&element_bytes[0..4]);
            let y = LittleEndian::read_f32(&element_bytes[4..8]);
            let z = LittleEndian::read_f32(&element_bytes[8..12]);
            let w = LittleEndian::read_f32(&element_bytes[12..16]);
            output.push([x, y, z, w]);
            //println!("{}: {:?}, len: {}", i, output[output.len() - 1], (x * x + y * y + z * z).sqrt());
        }
    } else {
        return None;
    }

    Some(output)
}

fn parse_float3_attribute(
    bytes: &mut Cursor<&Cow<[u8]>>,
    semantic: &gltf::Semantic,
    vertex_count: usize,
    primitive: &Primitive,
) -> Option<Vec<[f32; 3]>> {
    let mut output = Vec::new();
    output.reserve(vertex_count);

    if let Some(accessor) = primitive.get(semantic) {
        let view = accessor.view().unwrap();
        const STRIDE: usize = 3 * 4;
        bytes.seek(SeekFrom::Start((view.offset()) as u64)).unwrap();
        let mut element_bytes = [0; STRIDE];
        for _ in 0..accessor.count() {
            bytes.read_exact(&mut element_bytes).unwrap();
            let x = LittleEndian::read_f32(&element_bytes[0..4]);
            let y = LittleEndian::read_f32(&element_bytes[4..8]);
            let z = LittleEndian::read_f32(&element_bytes[8..12]);
            output.push([x, y, z]);
            //println!("{}: {:?}, len: {}", i, output[output.len() - 1], (x * x + y * y + z * z).sqrt());
        }
    } else {
        return None;
    }

    Some(output)
}

fn parse_float2_attribute(
    bytes: &mut Cursor<&Cow<[u8]>>,
    semantic: &gltf::Semantic,
    vertex_count: usize,
    primitive: &Primitive,
) -> Option<Vec<[f32; 2]>> {
    let mut output = Vec::new();
    output.reserve(vertex_count);
    if let Some(accessor) = primitive.get(semantic) {
        let view = accessor.view().unwrap();
        const STRIDE: usize = 2 * 4;
        bytes.seek(SeekFrom::Start((view.offset()) as u64)).unwrap();
        let mut element_bytes = [0; STRIDE];
        for _ in 0..accessor.count() {
            bytes.read_exact(&mut element_bytes).unwrap();
            let x = LittleEndian::read_f32(&element_bytes[0..4]);
            let y = LittleEndian::read_f32(&element_bytes[4..8]);
            output.push([x, y]);
            //println!("{}: {:?}", i, output[output.len() - 1]);
        }
    } else {
        return None;
    }

    Some(output)
}

fn parse_indices(bytes: &mut Cursor<&Cow<[u8]>>, primitive: &Primitive) -> Option<Vec<u32>> {
    let mut index_buffer: Vec<u32> = Vec::new();
    let indices_iter = primitive.indices().unwrap();
    let index_count = indices_iter.count();
    index_buffer.reserve(index_count);
    let index_stride = indices_iter.size();
    for _ in 0..index_count {
        let index: u32 = match index_stride {
            1 => bytes.read_u8().unwrap() as u32,
            2 => bytes.read_u16::<LittleEndian>().unwrap() as u32,
            4 => bytes.read_u32::<LittleEndian>().unwrap() as u32,
            _ => 0,
        };
        index_buffer.push(index);

        // println!("{}: {:?}", i, indices[indices.len() - 1]);
    }
    Some(index_buffer)
}

fn parse_joint_indices(
    bytes: &mut Cursor<&Cow<[u8]>>,
    vertex_count: usize,
    primitive: &Primitive,
) -> Option<Vec<[u32; 4]>> {
    let mut output = Vec::new();
    println!(
        "Joint indices: {}",
        primitive.get(&gltf::Semantic::Joints(0)).is_some()
    );

    if let Some(accessor) = primitive.get(&gltf::Semantic::Joints(0)) {
        output.reserve(vertex_count);
        let view = accessor.view().unwrap();
        const STRIDE: usize = 16;
        bytes.seek(SeekFrom::Start((view.offset()) as u64)).unwrap();

        println!("joint index count: {}", accessor.count());

        let mut element_bytes = [0; STRIDE];
        for _ in 0..accessor.count() {
            bytes.read_exact(&mut element_bytes).unwrap();
            let index_0 = LittleEndian::read_u32(&element_bytes[0..4]);
            let index_1 = LittleEndian::read_u32(&element_bytes[4..8]);
            let index_2 = LittleEndian::read_u32(&element_bytes[8..12]);
            let index_3 = LittleEndian::read_u32(&element_bytes[12..16]);
            output.push([index_0, index_1, index_2, index_3]);
            println!(
                "weight indices: {},{},{},{}",
                index_0, index_1, index_2, index_3
            );
        }
    } else {
        return None;
    }

    Some(output)
}
