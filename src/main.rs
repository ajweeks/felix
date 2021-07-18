extern crate winit;

#[macro_use]
extern crate lazy_static; // For filewatcher::FILE_WATCHER

mod compute_skinning_pass;
mod game;
mod math_helpers;
mod mesh;
mod render_mesh_pass;
mod render_pass_common;
mod vulkan_base;
mod vulkan_helpers;

mod filewatcher;
mod settings;

use game::*;
use winit::{dpi, event_loop::EventLoop, window::WindowBuilder};

pub const WINDOW_TITLE_PREFIX: &str = "Felix";

pub struct Window {
    window: winit::window::Window,
    width: u32,
    height: u32,
}

impl Window {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let (width, height) = (1280, 720);

        let window = WindowBuilder::new()
            .with_title(crate::WINDOW_TITLE_PREFIX)
            .with_inner_size(dpi::PhysicalSize::new(width, height))
            .build(event_loop)
            .unwrap();

        Window {
            window,
            width,
            height,
        }
    }
}

fn main() {
    let mut event_loop = EventLoop::new();
    let window = Window::new(&event_loop);

    let mut game = Game::new(&window);
    game.run_update_loop(&window, &mut event_loop);
}
