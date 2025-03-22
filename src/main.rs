mod api;
mod cli;
mod core;
mod utils;

use clap::{Parser, Subcommand};
use std::error::Error;
use tracing::info;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the API server
    Serve {
        #[arg(short = 'H', long, default_value = "127.0.0.1")]
        host: String,

        #[arg(short, long, default_value_t = 8080)]
        port: u16,
    },

    /// Ingest media files
    Ingest {
        /// Path to file or directory
        path: std::path::PathBuf,

        #[arg(short, long)]
        recursive: bool,

        #[arg(
            short,
            long,
            help = "Maximum recursion depth when scanning directories"
        )]
        max_depth: Option<usize>,
    },

    /// Search for media
    Search {
        /// Search query
        query: String,

        #[arg(short, long, default_value_t = 10)]
        limit: usize,
    },

    /// Manage tags
    Tag {
        /// Media ID to tag
        media_id: String,

        #[arg(long, value_delimiter = ',')]
        add: Vec<String>,

        #[arg(long, value_delimiter = ',')]
        remove: Vec<String>,
    },

    /// List all tags
    ListTags,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    utils::logging::init();

    // Load configuration
    let config = core::config::load_config()?;

    // Initialize application state
    let app_state = core::state::AppState::new(config).await?;

    // Parse command line arguments
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { host, port } => {
            info!("Starting API server at {}:{}", host, port);
            api::run_server(host, port, app_state).await?;
        }
        Commands::Ingest {
            path,
            recursive,
            max_depth,
        } => {
            info!(
                "Ingesting media from {:?} (recursive: {}, max_depth: {})",
                path,
                recursive,
                max_depth.unwrap_or(5)
            );
            cli::commands::ingest(path, recursive, &app_state, max_depth).await?;
        }
        Commands::Search { query, limit } => {
            info!("Searching for: {}", query);
            cli::commands::search(query, limit, &app_state).await?;
        }
        Commands::Tag {
            media_id,
            add,
            remove,
        } => {
            info!("Managing tags for media: {}", media_id);
            cli::commands::tag(media_id, add, remove, &app_state).await?;
        }
        Commands::ListTags => {
            info!("Listing all tags");
            cli::commands::list_tags(&app_state).await?;
        }
    }

    Ok(())
}
