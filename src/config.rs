use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub enum SdapiBackend {
    Webui,
    // Naifu,
}

#[derive(Deserialize, Serialize)]
pub struct Config {
    pub token: String,
    pub app_id: u64,
    pub target_guild: Option<u64>,
    pub prefix: String,
    pub sdapi_backend: SdapiBackend, 
    pub admins: Vec<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            token: "PASTE_TOKEN_HERE".into(),
            app_id: 0,
            target_guild: Some(0),
            prefix: "$$".into(),
            sdapi_backend: SdapiBackend::Webui,
            admins: vec![]
        }
    }
}

impl Config {
    /// Load config from file. If file not exists, load default
    pub fn load(path: &str) -> anyhow::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(s) => Ok(toml::from_str(&s)?), 
            Err(e) => match e.kind() {
                std::io::ErrorKind::NotFound => {
                    let cfg = Self::default();
                    let toml_str = toml::to_string_pretty(&cfg)?;
                    std::fs::write(path, toml_str)?;
                    Ok(cfg)
                },
                e => anyhow::bail!(e),
            },
        }
    }
}