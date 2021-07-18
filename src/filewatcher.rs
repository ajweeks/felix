use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Mutex,
    thread,
    time::Duration,
};

use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};

struct FileWatcher {
    watcher: RecommendedWatcher,
    callbacks: HashMap<PathBuf, Box<dyn Fn() + Sync + Send>>,
}

impl FileWatcher {
    fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();

        let watcher: RecommendedWatcher = Watcher::new(tx, Duration::from_millis(50)).unwrap();

        thread::spawn(move || loop {
            match rx.recv() {
                Ok(DebouncedEvent::Write(path)) => {
                    if let Some(ref callback) = FILE_WATCHER.lock().unwrap().callbacks.get(&path) {
                        callback();
                    }
                }
                Err(e) => {
                    eprintln!("File watch error: {:?}", e);
                }
                _ => (),
            }
        });

        FileWatcher {
            watcher,
            callbacks: HashMap::new(),
        }
    }

    fn watch<F: Fn() + Sync + Send + 'static>(&mut self, path: &str, callback: F) {
        let path = Path::new(path).canonicalize().unwrap();
        if !self.callbacks.contains_key(&path) {
            self.watcher
                .watch(path.clone(), RecursiveMode::NonRecursive)
                .unwrap();
        }

        self.callbacks.insert(path, Box::new(callback));
    }
}

lazy_static! {
    static ref FILE_WATCHER: Mutex<FileWatcher> = Mutex::new(FileWatcher::new());
}

pub fn watch_file<F: Fn() + Sync + Send + 'static>(path: &str, callback: F) {
    FILE_WATCHER.lock().unwrap().watch(path, callback);
}
