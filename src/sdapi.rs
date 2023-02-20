use once_cell::sync::Lazy;
use serde::{Serialize, Deserialize};
use serenity::async_trait;
use reqwest::Client;

/// Webui singleton
pub static WEBUI: Lazy<Webui> = Lazy::new(Webui::new); 

/// Trait for stable diffusion backends
#[async_trait]
pub trait SdApi {    
    type Txt2ImgReq;
    type Img2ImgReq;
    type Txt2ImgResp;
    type Img2ImgResp;

    async fn txt2img(&self, req: Self::Txt2ImgReq) -> reqwest::Result<Self::Txt2ImgResp>;
    async fn img2img(&self, req: Self::Img2ImgReq) -> reqwest::Result<Self::Img2ImgResp>;
}

#[derive(Serialize)]
pub struct WebuiRequestTxt2Img {
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
pub struct WebuiRequestImg2Img {
    #[serde(flatten)]
    pub txt2img: WebuiRequestTxt2Img,
    pub init_images: Vec<String>,
}

#[derive(Deserialize)]
pub struct WebuiResponse {
    pub images: Vec<String>,
    pub info: String,
}

#[derive(Deserialize)]
pub struct WebuiInfo {
    // ...
    pub infotexts: Vec<String>,
}

pub struct Webui {
    client: Client
}

impl Webui {
    pub fn new() -> Self {
        Self { client: Client::new() }
    }
}

impl Default for WebuiRequestTxt2Img {
    fn default() -> Self {
        Self {
            prompt: "masterpiece, best quality,".into(),
            seed: -1,
            sampler_name: "Euler a".into(),
            batch_size: 1,
            steps: 28,
            cfg_scale: 7,
            width: 512,
            height: 512,
            denoising_strength: 0.75,
            negative_prompt: "(worst quality, low quality:1.4), bad anatomy, hands, ".into(),
        }
    }
}

#[async_trait]
impl SdApi for Webui {
    type Txt2ImgReq = WebuiRequestTxt2Img;
    type Img2ImgReq = WebuiRequestImg2Img;
    type Txt2ImgResp = WebuiResponse;
    type Img2ImgResp = WebuiResponse;

    async fn txt2img(&self, req: Self::Txt2ImgReq) -> reqwest::Result<Self::Txt2ImgResp> {
        self.client
            .post("http://127.0.0.1:7860/sdapi/v1/txt2img")
            .json(&req)
            .send()
            .await?
            .json()
            .await
    }

    async fn img2img(&self, req: Self::Img2ImgReq) -> reqwest::Result<Self::Img2ImgResp> {
        self.client
            .post("http://127.0.0.1:7860/sdapi/v1/img2img")
            .json(&req)
            .send()
            .await?
            .json()
            .await
    }
}
