use serde::{Deserialize, Serialize};

use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use crate::filewatcher;

pub const SETTINGS_LOCATION: &str = "config/settings.json";

pub struct SettingsWrapper {
    pub settings: Settings,
    out_of_date: Arc<AtomicBool>,
}

impl Default for SettingsWrapper {
    fn default() -> Self {
        SettingsWrapper {
            settings: Settings::default(),
            out_of_date: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl SettingsWrapper {
    pub fn is_out_of_date(&self) -> bool {
        self.out_of_date.load(Ordering::Acquire)
    }

    pub fn reload(&mut self) {
        self.settings = Settings::deserialize();
        self.out_of_date.store(false, Ordering::Release);
    }

    pub fn serialize(&self) -> std::io::Result<()> {
        Settings::serialize(&self.settings)
    }

    pub fn create_from_file() -> SettingsWrapper {
        let settings_out_of_date = Arc::new(AtomicBool::new(false));
        let settings_out_of_date2 = settings_out_of_date.clone();
        filewatcher::watch_file(SETTINGS_LOCATION, move || {
            settings_out_of_date2.store(true, Ordering::Release);
        });

        SettingsWrapper {
            settings: Settings::deserialize(),
            out_of_date: settings_out_of_date,
        }
    }
}

#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct Settings {
    pub mouse_sensitivity: f32,
    pub move_speed: f32,
    pub dolly_speed: f32,
    pub mod_faster_speed: f32,
    pub mod_slower_speed: f32,
}

impl Settings {
    fn default() -> Self {
        Settings {
            mouse_sensitivity: 0.0015,
            move_speed: 1.5,
            dolly_speed: 5.0,
            mod_faster_speed: 2.0,
            mod_slower_speed: 0.5,
        }
    }

    pub fn serialize(settings: &Settings) -> std::io::Result<()> {
        return {
            match serde_json::to_string_pretty(settings) {
                Ok(json_string) => {
                    let path = Path::new(SETTINGS_LOCATION);
                    let path_buf = PathBuf::from(&path);
                    let dir_str = path_buf.parent().unwrap().to_str().unwrap();
                    let path = Path::new(path.to_str().unwrap()).to_str().unwrap();
                    let dir_result = fs::create_dir(&dir_str);
                    if let Err(err) = dir_result {
                        if err.kind() != ErrorKind::AlreadyExists {
                            eprintln!("Failed to create directory {}, error: {}", dir_str, err);
                            return Err(err);
                        }
                    }
                    if let Err(err) = fs::write(&path, json_string) {
                        eprintln!("Failed to serialize settings to {}, error: {}", path, err);
                        return Err(err);
                    }

                    Ok(())
                }
                Err(err) => Err(err.into()),
            }
        };
    }

    pub fn deserialize() -> Settings {
        if let Ok(file_contents) = fs::read_to_string(SETTINGS_LOCATION) {
            serde_json::from_str(&file_contents).unwrap()
        } else {
            Settings::default()
        }
    }
}
