use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Mutex,
};

use notify::{Error, Event, RecommendedWatcher, RecursiveMode, Watcher};

struct FileWatcher {
    watcher: RecommendedWatcher,
    callbacks: HashMap<PathBuf, Box<dyn Fn() + Sync + Send>>,
}

impl FileWatcher {
    fn new() -> Self {
        // let (tx, rx) = std::sync::mpsc::channel();

        let mut watcher = notify::recommended_watcher(|res: Result<Event, Error>| match res {
            Ok(event) => {
                for path in event.paths {
                    if let Some(ref callback) = FILE_WATCHER.lock().unwrap().callbacks.get(&path) {
                        callback();
                    }
                }
            }
            Err(e) => println!("File watch error: {:?}", e),
        })
        .unwrap();

        FileWatcher {
            watcher,
            callbacks: HashMap::new(),
        }
    }

    fn watch<F: Fn() + Sync + Send + 'static>(
        &mut self,
        path: &str,
        callback: F,
    ) -> anyhow::Result<()> {
        let path = Path::new(path).canonicalize().unwrap();

        if !self.callbacks.contains_key(&path) {
            self.watcher
                .watch(path.as_path(), RecursiveMode::NonRecursive)?;
        }

        self.callbacks.insert(path, Box::new(callback));

        Ok(())
    }
}

lazy_static! {
    static ref FILE_WATCHER: Mutex<FileWatcher> = Mutex::new(FileWatcher::new());
}

pub fn watch_file<F: Fn() + Sync + Send + 'static>(path: &str, callback: F) {
    FILE_WATCHER.lock().unwrap().watch(path, callback);
}
