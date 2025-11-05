use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::io::{BufRead, BufReader};
use tauri::{State, Emitter};

struct ServerState {
    process: Mutex<Option<Child>>,
    logs: Arc<Mutex<Vec<String>>>,
}

#[tauri::command]
async fn start_server(state: State<'_, ServerState>, app: tauri::AppHandle) -> Result<String, String> {
    // Check if server is already running
    {
        let process_guard = state.process.lock().unwrap();
        if process_guard.is_some() {
            return Err("Server is already running".to_string());
        }
    } // MutexGuard is dropped here

    // Clear previous logs
    {
        let mut logs = state.logs.lock().unwrap();
        logs.clear();
    }

    // Determine the RNN server path
    let rnn_server_path = if cfg!(debug_assertions) {
        // In development, use path relative to workspace root
        // The executable is in target/debug/, so we go up 3 levels to workspace root
        let exe_path = std::env::current_exe()
            .map_err(|e| format!("Failed to get executable path: {}", e))?;

        exe_path
            .parent() // target/debug
            .and_then(|p| p.parent()) // target
            .and_then(|p| p.parent()) // src-tauri
            .and_then(|p| p.parent()) // tauri-app
            .and_then(|p| p.parent()) // project root
            .ok_or("Failed to resolve project root")?
            .join("rnn-server")
    } else {
        // In production, rnn-server is bundled next to the executable
        let app_dir = std::env::current_exe()
            .map_err(|e| format!("Failed to get executable path: {}", e))?
            .parent()
            .ok_or("Failed to get parent directory")?
            .to_path_buf();
        app_dir.join("rnn-server")
    };

    if !rnn_server_path.exists() {
        return Err(format!("RNN server directory not found at: {:?}", rnn_server_path));
    }

    // Spawn the process with merged stdout/stderr and unbuffered output
    // Redirect stderr to stdout to capture all output in one stream
    #[cfg(target_os = "windows")]
    let mut child = Command::new("cmd")
        .args(&["/C", "uv", "run", "fastapi", "dev", "main.py", "--host", "127.0.0.1", "--port", "8000", "2>&1"])
        .current_dir(&rnn_server_path)
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to start server: {}. Make sure 'uv' is installed and in PATH.", e))?;

    #[cfg(not(target_os = "windows"))]
    let mut child = Command::new("sh")
        .args(&["-c", "uv run fastapi dev main.py --host 127.0.0.1 --port 8000 2>&1"])
        .current_dir(&rnn_server_path)
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to start server: {}. Make sure 'uv' is installed and in PATH.", e))?;

    // Capture merged stdout/stderr
    if let Some(stdout) = child.stdout.take() {
        let logs = state.logs.clone();
        let app_handle = app.clone();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            println!("[RUST] Started output capture thread");
            for line in reader.lines() {
                if let Ok(line) = line {
                    println!("[RUST OUTPUT] {}", line);
                    // Store in logs
                    {
                        let mut logs = logs.lock().unwrap();
                        logs.push(line.clone());
                        // Keep only last 1000 lines
                        if logs.len() > 1000 {
                            logs.remove(0);
                        }
                    }
                    // Emit to frontend
                    if let Err(e) = app_handle.emit("server-log", &line) {
                        eprintln!("[RUST] Failed to emit server-log: {}", e);
                    }
                }
            }
            println!("[RUST] Output capture thread ended");
        });
    }

    // Store the process
    {
        let mut process_guard = state.process.lock().unwrap();
        *process_guard = Some(child);
    } // MutexGuard is dropped here before await

    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    Ok("Server started successfully on http://127.0.0.1:8000".to_string())
}

#[tauri::command]
async fn stop_server(state: State<'_, ServerState>) -> Result<String, String> {
    let mut process_guard = state.process.lock().unwrap();
    
    if let Some(mut child) = process_guard.take() {
        child.kill()
            .map_err(|e| format!("Failed to stop server: {}", e))?;
        Ok("Server stopped successfully".to_string())
    } else {
        Err("Server is not running".to_string())
    }
}

#[tauri::command]
async fn check_server_status(state: State<'_, ServerState>) -> Result<bool, String> {
    let process_guard = state.process.lock().unwrap();
    Ok(process_guard.is_some())
}

#[tauri::command]
async fn test_server() -> Result<String, String> {
    let client = reqwest::Client::new();
    
    match client.get("http://127.0.0.1:8000/health-check")
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                match response.json::<serde_json::Value>().await {
                    Ok(json) => Ok(format!("✓ Server is healthy: {}", json)),
                    Err(_) => Ok("✓ Server responded successfully".to_string()),
                }
            } else {
                Err(format!("Server returned status: {}", response.status()))
            }
        }
        Err(e) => Err(format!("Connection failed: {}", e)),
    }
}

#[tauri::command]
fn get_server_logs(state: State<'_, ServerState>) -> Vec<String> {
    let logs = state.logs.lock().unwrap();
    logs.clone()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(ServerState {
            process: Mutex::new(None),
            logs: Arc::new(Mutex::new(Vec::new())),
        })
        .invoke_handler(tauri::generate_handler![
            start_server,
            stop_server,
            check_server_status,
            test_server,
            get_server_logs
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() {
    run();
}
