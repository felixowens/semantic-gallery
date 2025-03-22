use crate::core::state::AppState;
use std::error::Error;

pub async fn run_server(
    host: String,
    port: u16,
    app_state: AppState,
) -> Result<(), Box<dyn Error>> {
    // TODO: Implement API server
    todo!("Implement API server")
}
