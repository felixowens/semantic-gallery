mod api;
mod cli;
mod core;
mod utils;

use clap::{Parser, Subcommand};
use std::error::Error;

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
        #[arg(short, long, default_value = "127.0.0.1")]
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

    // Parse command line arguments
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { host, port } => {
            println!("Starting API server at {}:{}", host, port);
            api::run_server(host, port, config).await?;
        }
        Commands::Ingest { path, recursive } => {
            println!("Ingesting media from {:?} (recursive: {})", path, recursive);
            cli::commands::ingest(path, recursive, &config).await?;
        }
        Commands::Search { query, limit } => {
            println!("Searching for: {}", query);
            cli::commands::search(query, limit, &config).await?;
        }
        Commands::Tag {
            media_id,
            add,
            remove,
        } => {
            println!("Managing tags for media: {}", media_id);
            cli::commands::tag(media_id, add, remove, &config).await?;
        }
        Commands::ListTags => {
            println!("Listing all tags");
            cli::commands::list_tags(&config).await?;
        }
    }

    Ok(())
}
