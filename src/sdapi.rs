use std::{time::Duration, io::Cursor};

use image::{imageops::FilterType, ImageFormat};
use once_cell::sync::Lazy;
use serde::{Serialize, Deserialize};
use reqwest::{Client, StatusCode};
use base64::{Engine, engine::general_purpose, DecodeError};
use rand::Rng;
use tokio::sync::RwLock;

use crate::{CONFIG, config::SdapiBackend};

/// Enum for trait static dispatch
pub enum SdBackend {
    Webui(Webui),
    Naifu(Naifu),
    Mock(MockSdApi)
}

impl SdApi for SdBackend {
    fn fill_request(&self, req: &mut GenerationRequest) {
        match self {
            SdBackend::Webui(w) => w.fill_request(req),
            SdBackend::Naifu(n) => n.fill_request(req),
            SdBackend::Mock(m) => m.fill_request(req),
        }
    }
    
    async fn generate(&self, req: GenerationRequest) -> anyhow::Result<(ImageVec, GenerationRequest)> {
        match self {
            SdBackend::Webui(w) => w.generate(req).await,
            SdBackend::Naifu(n) => n.generate(req).await,
            SdBackend::Mock(m) => m.generate(req).await,
        }
    }

    async fn progress(&self) -> anyhow::Result<Progress> {
        match self {
            SdBackend::Webui(w) => w.progress().await,
            SdBackend::Naifu(n) => n.progress().await,
            SdBackend::Mock(m) => m.progress().await,
        }
    }
}

/// Stable Diffusion backend singleton
pub static SDAPI: Lazy<SdBackend> = Lazy::new(|| match &CONFIG.sdapi_backend {
    SdapiBackend::Mock => SdBackend::Mock(MockSdApi { 
        progress: RwLock::new((0, 0)) 
    }),
    SdapiBackend::Webui { url } => SdBackend::Webui(Webui { 
        client: Client::new(), 
        url: url.clone()  
    }),
    SdapiBackend::Naifu { url } => SdBackend::Naifu(Naifu { 
        client: Client::new(), 
        url: url.clone() 
    }),
});

pub type ImageVec = Vec<Vec<u8>>;
pub type Progress = (u32, u32);

/// Backend-agnostic generation request
/// Generation type based on specified parameters
#[derive(Clone)]
pub struct GenerationRequest {
    pub prompt: String,
    pub neg_prompt: Option<String>,
    pub seed: Option<u32>,
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

pub trait SdApi {
    // Default values
    const PROMPT: &'static str = "";
    const NEG_PROMPT: &'static str = "";
    const SAMPLER: &'static str = "";
    const STEPS: u32 = 0;
    const CFG_SCALE: u32 = 0;
    const WIDTH: u32 = 0;
    const HEIGHT: u32 = 0;
    const STRENGTH: f64 = 0.;

    /// Fill unspecified fields with default parameters
    fn fill_request(&self, req: &mut GenerationRequest) {
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
        let seed = rand::thread_rng().gen_range(0..u32::MAX);
        req.seed.get_or_insert(seed);
        
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
    async fn progress(&self) -> anyhow::Result<Progress>;
}

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
        self.fill_request(&mut req);
        
        let builder = match req.get_type() {
            GenerationType::Txt2Img => self.client
                .post(format!("{}/sdapi/v1/txt2img", self.url)),
            GenerationType::Img2Img => self.client
                .post(format!("{}/sdapi/v1/img2img", self.url)),
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
    
    async fn progress(&self) -> anyhow::Result<Progress> {
        let resp: WebuiProgress = self.client
            .get("http://127.0.0.1:7860/sdapi/v1/progress")
            .send()
            .await?
            .json()
            .await?;

        Ok((resp.state.sampling_step, resp.state.sampling_steps))
    }
}

impl SdApi for Naifu {
    const PROMPT: &'static str = "masterpiece, best quality";
    const NEG_PROMPT: &'static str = 
        "lowres, bad anatomy, bad hands, text, error,
        missing fingers, extra digit, fewer digits, cropped, 
        worst quality, low quality, normal quality, jpeg artifacts, 
        signature, watermark, username, blurry";
    const SAMPLER: &'static str = "Euler a";
    const STEPS: u32 = 28;
    const CFG_SCALE: u32 = 12;
    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 512;
    const STRENGTH: f64 = 0.69;

    
    async fn generate(&self, mut req: GenerationRequest) -> anyhow::Result<(ImageVec, GenerationRequest)> {
        self.fill_request(&mut req);

        let request: NaifuGenerationRequest = req.clone().into();
        let out: NaifuGenerationOutput = self.client
            .post(format!("{}/generate", self.url))
            .json(&request)
            .send()
            .await?
            .json()
            .await?;

        match out {
            NaifuGenerationOutput { output: Some(output), .. } => {
                let res: Result<Vec<_>, DecodeError> = output
                    .into_iter()
                    .map(|b64| general_purpose::STANDARD.decode(b64))
                    .collect();
                Ok((res?, req))
            },
            NaifuGenerationOutput { error: Some(error), .. } => anyhow::bail!(error), 
            NaifuGenerationOutput { .. } => anyhow::bail!("unknown naifu error"), 
        }
    }
    
    async fn progress(&self) -> anyhow::Result<Progress> {
        anyhow::bail!("not supported")
    }
}

impl SdApi for MockSdApi {
    const PROMPT: &'static str = "masterpiece, best quality";
    const NEG_PROMPT: &'static str = "(worst quality, low quality:1.4), bad anatomy, hands";
    const SAMPLER: &'static str = "Euler a";
    const STEPS: u32 = 28;
    const CFG_SCALE: u32 = 7;
    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 512;
    const STRENGTH: f64 = 0.75;
    
    async fn generate(&self, mut req: GenerationRequest) -> anyhow::Result<(ImageVec, GenerationRequest)> {
        self.fill_request(&mut req);
        
        let steps = req.steps.unwrap();
        self.progress.write().await.1 = steps;
        for step in 0..steps {
            tokio::time::sleep(Duration::from_millis(10)).await;
            self.progress.write().await.0 = step;
        }
        *self.progress.write().await = (0, 0);
        
        let img = image::load_from_memory(include_bytes!("../res/hanyuu.png"))?
            .resize_exact(req.width.unwrap(), req.height.unwrap(), FilterType::Triangle);
        
        let mut out = Cursor::new(vec![]);
        img.write_to(&mut out, ImageFormat::Png)?;
        Ok((vec![out.into_inner(); req.batch.unwrap() as usize], req))
    }
    
    async fn progress(&self) -> anyhow::Result<Progress> {
        Ok(*self.progress.read().await)
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

impl From<GenerationRequest> for NaifuGenerationRequest {
    fn from(value: GenerationRequest) -> Self {
        Self {
            sampler: match value.sampler.unwrap().as_str() {
                "Euler a" => "k_euler_ancestral",
                "Euler" => "k_euler",
                "LMS" => "k_lms",
                "DDIM" => "ddim",
                _ => "non-existent sampler"
            }.to_string(),
            height: value.height.unwrap(),
            width: value.width.unwrap(),
            n_samples: value.batch.unwrap(),
            prompt: value.prompt,
            scale: value.scale.unwrap() as f64,
            seed: value.seed.unwrap(),
            steps: value.steps.unwrap(),
            strength: value.strength.unwrap(),
            uc: value.neg_prompt.unwrap(),
            image: value.init_images
                .and_then(|v| v.into_iter()
                    .next()
                    .map(|img| general_purpose::STANDARD.encode(img.as_slice()))
                ),
        }
    }
}

#[derive(Serialize)]
struct WebuiRequestTxt2Img {
    pub prompt: String,
    pub seed: u32,
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

#[derive(Deserialize)]
struct WebuiProgress {
    // progress: f64,
    // eta_relative: f64,
    state: WebuiProgressState
}

#[derive(Deserialize)]
struct WebuiProgressState {
    sampling_step: u32,
    sampling_steps: u32,
}

#[derive(Serialize)]
struct NaifuGenerationRequest {
    height: u32,
    width: u32,
    sampler: String,
    n_samples: u32,
    prompt: String,
    scale: f64,
    seed: u32,
    steps: u32,
    strength: f64,
    uc: String,
    image: Option<String>,
}

#[derive(Deserialize)]
struct NaifuGenerationOutput {
    output: Option<Vec<String>>,
    error: Option<String>,
}

/// https://github.com/AUTOMATIC1111/stable-diffusion-webui
pub struct Webui {
    client: Client,
    url: String,
}

/// Api used for testing bot
pub struct MockSdApi {
    progress: RwLock<Progress>
}

/// NovelAI leaked frontend
pub struct Naifu {
    client: Client,
    url: String
}