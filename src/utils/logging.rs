use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

/// Initialise logging
///
/// This function sets up the logging system using the `tracing` crate.
/// It creates a subscriber that formats log messages using the `fmt`
/// formatter and sets the global default subscriber.
///
/// # Example
///
/// ```rust
/// use semantic_gallery::utils::logging;
///
/// fn main() {
///     logging::init();
/// }
/// ```
pub fn init() {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");

    info!("Logging initialised");
}
