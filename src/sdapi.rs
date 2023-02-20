use once_cell::sync::Lazy;
use serde::{Serialize, Deserialize};
use serenity::async_trait;
use reqwest::{Client, StatusCode};
use base64::{Engine, engine::general_purpose, DecodeError};
use rand::Rng;

/// Webui singleton
pub static WEBUI: Lazy<Webui> = Lazy::new(Webui::new); 

pub type ImageVec = Vec<Vec<u8>>;

/// Backend-agnostic generation request
/// Generation type based on specified parameters
#[derive(Clone)]
pub struct GenerationRequest {
    pub prompt: String,
    pub neg_prompt: Option<String>,
    pub seed: Option<i64>,
    pub sampler: Option<String>,
    pub steps: Option<u32>,
    pub scale: Option<u32>,
    pub strength: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub batch: Option<u32>,
    pub no_default_prompt: bool,
    pub no_default_neg_prompt: bool,
    pub init_images: Option<ImageVec>,
}

enum GenerationType {
    Txt2Img,
    Img2Img,
}

impl GenerationRequest {
    fn get_type(&self) -> GenerationType {
        match self.init_images {
            Some(_) => GenerationType::Img2Img,
            None => GenerationType::Txt2Img
        }
    }
}

#[async_trait]
pub trait SdApi {
    // Default values
    const PROMPT: &'static str;
    const NEG_PROMPT: &'static str;
    const SAMPLER: &'static str;
    const STEPS: u32;
    const CFG_SCALE: u32;
    const WIDTH: u32;
    const HEIGHT: u32;
    const STRENGTH: f64;

    /// Fill unspecified fields with default parameters
    fn fill_request(req: &mut GenerationRequest) {
        // Append default prompt
        if !req.no_default_prompt {
            req.prompt.push_str(", ");
            req.prompt.push_str(Self::PROMPT);
        }

        // Append default negative prompt
        if !req.no_default_neg_prompt {
            if let Some(ref mut neg) = req.neg_prompt {
                neg.push_str(", ");
                neg.push_str(Self::NEG_PROMPT);   
            }
        }

        // Generate random seed
        req.seed.get_or_insert(rand::thread_rng().gen());
        
        req.neg_prompt.get_or_insert(Self::NEG_PROMPT.to_string());
        req.sampler.get_or_insert(Self::SAMPLER.to_string());
        req.steps.get_or_insert(Self::STEPS);
        req.scale.get_or_insert(Self::CFG_SCALE);
        req.strength.get_or_insert(Self::STRENGTH);
        req.width.get_or_insert(Self::WIDTH);
        req.height.get_or_insert(Self::HEIGHT);
        req.batch.get_or_insert(1);
    }
    
    /// Generate txt2img or img2img based on request
    async fn generate(&self, req: GenerationRequest) -> anyhow::Result<(ImageVec, GenerationRequest)>;
    
    /// Get progress `(current, full)`
    async fn progress(&self) -> Option<(u32, u32)>;
}

#[async_trait]
impl SdApi for Webui {
    const PROMPT: &'static str = "masterpiece, best quality";
    const NEG_PROMPT: &'static str = "(worst quality, low quality:1.4), bad anatomy, hands";
    const SAMPLER: &'static str = "Euler a";
    const STEPS: u32 = 28;
    const CFG_SCALE: u32 = 7;
    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 512;
    const STRENGTH: f64 = 0.75;

    async fn generate(&self, mut req: GenerationRequest) -> anyhow::Result<(ImageVec, GenerationRequest)> {
        Self::fill_request(&mut req);
        
        let builder = match req.get_type() {
            GenerationType::Txt2Img => self.client.post("http://127.0.0.1:7860/sdapi/v1/txt2img"),
            GenerationType::Img2Img => self.client.post("http://127.0.0.1:7860/sdapi/v1/img2img"),
        };

        let out_req = req.clone();
        let resp = match req.get_type() {
            GenerationType::Txt2Img => {
                let req: WebuiRequestTxt2Img = req.into();
                builder.json(&req)
            }
            GenerationType::Img2Img => {
                let req: WebuiRequestImg2Img = req.into();
                builder.json(&req)
            }
        }
        .send()
        .await?;

        match resp.status() {
            StatusCode::OK => {
                let resp: WebuiResponse = resp.json().await?;
                let res: Result<Vec<_>, DecodeError> = resp.images
                    .into_iter()
                    .map(|b64| general_purpose::STANDARD.decode(b64))
                    .collect();

                Ok((res?, out_req))
            }
            _ => anyhow::bail!(resp.text().await?)
        }
        
    }
    
    async fn progress(&self) -> Option<(u32, u32)> {
        None
    }
}

impl From<GenerationRequest> for WebuiRequestTxt2Img {
    /// Assume that request fully filled and unwrap it
    fn from(value: GenerationRequest) -> Self {
        Self {
            prompt: value.prompt,
            seed: value.seed.unwrap(),
            sampler_name: value.sampler.unwrap(),
            batch_size: value.batch.unwrap(),
            steps: value.steps.unwrap(),
            cfg_scale: value.scale.unwrap(),
            width: value.width.unwrap(),
            height: value.height.unwrap(),
            denoising_strength: value.strength.unwrap(),
            negative_prompt: value.neg_prompt.unwrap(),
        }
    }
}

impl From<GenerationRequest> for WebuiRequestImg2Img {
    /// Assume that request filled and has init_images
    fn from(mut value: GenerationRequest) -> Self {
        Self {
            init_images: value.init_images
                .take()
                .unwrap()
                .into_iter()
                .map(|img|
                    general_purpose::STANDARD.encode(img.as_slice())
                )
                .collect(),
            txt2img: value.into(),
        }
    }
}

#[derive(Serialize)]
struct WebuiRequestTxt2Img {
    pub prompt: String,
    pub seed: i64,
    pub sampler_name: String,
    pub batch_size: u32,
    pub steps: u32,
    pub cfg_scale: u32,
    pub width: u32,
    pub height: u32,
    pub denoising_strength: f64,
    pub negative_prompt: String,
}

#[derive(Serialize)]
struct WebuiRequestImg2Img {
    #[serde(flatten)]
    pub txt2img: WebuiRequestTxt2Img,
    pub init_images: Vec<String>,
}

#[derive(Deserialize)]
struct WebuiResponse {
    pub images: Vec<String>,
}

pub struct Webui {
    client: Client
}

impl Webui {
    pub fn new() -> Self {
        Self { client: Client::new() }
    }
}