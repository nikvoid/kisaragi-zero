use std::collections::HashMap;

use serde::Deserialize;

#[derive(Deserialize)]
pub enum SdapiBackend {
    Mock,
    Webui,
    Naifu
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Access {
    Public,
    Admin,
    Nobody
}

#[derive(Deserialize)]
pub struct Config {
    pub token: String,
    pub app_id: u64,
    pub target_guilds: Vec<u64>,
    pub prefix: String,
    pub sdapi_backend: SdapiBackend, 
    pub admins: Vec<u64>,
    pub acl: HashMap<String, Access>
}

impl Config {
    /// Load config from file.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let cfg = toml::from_str(&s)?;
        Ok(cfg)
    }

    /// Return true if user has rights to use command
    pub fn has_rights(&self, user_id: impl Into<u64>, cmd: &str) -> bool {
        match self.acl.get(cmd).copied().unwrap_or(Access::Public) {
            Access::Public => true,
            Access::Admin => self.admins.contains(&user_id.into()),
            Access::Nobody => false,
        }
    }
}