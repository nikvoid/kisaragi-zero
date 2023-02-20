use once_cell::sync::Lazy;
use serenity::model::prelude::Attachment;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;
use crate::config::SdapiBackend;
use crate::sdapi::{WebuiRequestTxt2Img, SdApi, WebuiRequestImg2Img, WebuiInfo};
use super::prelude::*;


enum DreamTask {
    Dream(DreamCommand, oneshot::Sender<anyhow::Result<(Vec<Vec<u8>>, String)>>),
}

/// Dream command executor
static DREAM_POOL: Lazy<Sender<DreamTask>> = Lazy::new(|| { 
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);
    tokio::spawn(async move { 
        while let Some(msg) = rx.recv().await {
            match crate::CONFIG.sdapi_backend {
                // Swith on backend
                SdapiBackend::Webui => match msg {
                    // /dream
                    DreamTask::Dream(cmd, back_tx) => {
                        // Send request
                        let resp: anyhow::Result<_> = try {
                            match &cmd.init_image {
                                Some(img) => {
                                    let req = WebuiRequestImg2Img {
                                        init_images: {
                                            let raw = img.download().await?;
                                            let b64 = base64::encode(raw);
                                            vec![b64] 
                                        },
                                        txt2img: cmd.into(),
                                    };
                                    crate::sdapi::WEBUI.img2img(req)
                                },
                                None => crate::sdapi::WEBUI.txt2img(cmd.into())
                            }
                            .await
                            .map(|r| {
                                // decode base64
                                let imgs: Vec<_> = r.images.into_iter()
                                    .filter_map(|s| match base64::decode(s) {
                                        Ok(bytes) => Some(bytes),
                                        Err(e) => {
                                            error!(?e, "Failed to decode base64");
                                            None
                                        },
                                    })
                                    .collect();
                                (imgs, r.info)
                            })?
                        };
                        back_tx.send(resp).ok();
                    },
                },
            }
        }
    }); 
    tx
});

/// Request image generation
#[derive(Command, Clone)]
#[name = "dream_ng"]
pub struct DreamCommand {
    /// Prompt. Quality improving keeeeeee will be prepended
    prompt: String,
    /// Steps
    #[min = 1]
    #[max = 166]
    steps: Option<i64>,
    /// Guidance scale
    #[min = 0]
    #[max = 100]
    cfg_scale: Option<i64>,
    /// Resulting width
    #[min = 1]
    #[max = 2048]
    width: Option<i64>,
    /// Resulting height
    #[min = 1]
    #[max = 2048]
    height: Option<i64>,
    /// Negative prompt. Will be appended to default keywords
    negative_prompt: Option<String>,
    /// Sampler name
    #[choice("Euler")]
    #[choice("Euler a")]
    #[choice("LMS")]
    #[choice("Heun")]
    #[choice("DPM2")]
    #[choice("DPM++ SDE Karras")]
    #[choice("DDIM")]
    #[choice("DPM2 a Karras")]
    #[choice("DPM adaptive")]
    #[choice("DPM++ SDE")]
    #[choice("DPM fast")]
    sampler_name: Option<String>,
    /// Seed for generation
    seed: Option<i64>,
    /// Initial image to use in img2img
    init_image: Option<Attachment>,
    /// Denoising strength
    #[min = 0.0]
    #[max = 1.0]
    denoising_strength: Option<f64>,
    /// Do not add default keywords to prompts
    no_default_keywords: Option<bool>,
    /// LoRa that will be appended to prompt
    #[choice("ryukishi")]
    lora: Option<String>,
}

impl From<DreamCommand> for WebuiRequestTxt2Img {    
    fn from(mut value: DreamCommand) -> Self {
        let lora = value.lora.map(|lora| match lora.as_str() {
            "ryukishi" => ", <lora:ryukishi07Higurashi_v1:1>",
            _ => panic!("unknown lora"),
        });

        if let Some(lora) = lora {
            value.prompt.push_str(lora)
        }
        
        let mut def = Self::default();
        Self {
            prompt: if matches!(value.no_default_keywords, Some(true)) {
                value.prompt
            } else {
                def.prompt.push_str(&value.prompt);
                def.prompt
            },
            seed: value.seed.unwrap_or(def.seed),
            sampler_name: value.sampler_name.unwrap_or(def.sampler_name),
            steps: value.steps.unwrap_or(def.steps as _) as _,
            cfg_scale: value.cfg_scale.unwrap_or(def.cfg_scale as _) as _,
            width: value.width.unwrap_or(def.width as _) as _,
            height: value.height.unwrap_or(def.height as _) as _,
            denoising_strength: value.denoising_strength.unwrap_or(def.denoising_strength),
            negative_prompt: match (value.no_default_keywords, value.negative_prompt) {
                (Some(true), Some(p)) => p,
                (Some(true), None) => String::new(),
                (Some(false) | None, None) => def.negative_prompt, 
                (None | Some(false), Some(p)) => {
                    def.negative_prompt.push_str(&p);
                    def.negative_prompt
                },
            },
            ..def
        }
    }
}

#[async_trait]
impl ApplicationCommandInteractionHandler for DreamCommand {
    async fn invoke(
        &self,
        ctx: &Context,
        command: &ApplicationCommandInteraction,
    ) -> Result<(), InvocationError> {
        // Send task to executor
        let queue_pos = DREAM_POOL.max_capacity() - DREAM_POOL.capacity();
        command
            .create_interaction_response(&ctx.http, |response| response
                .kind(InteractionResponseType::ChannelMessageWithSource)
                .interaction_response_data(|message| message
                    .content(format!("Queue position: #{queue_pos}")))
            )
            .await
            .map_err(|_| InvocationError)?;

        // Send command to executor with oneshot channel
        let (tx, rx) = oneshot::channel();
        DREAM_POOL.send(DreamTask::Dream(self.clone(), tx))
            .await
            .map_err(|_| InvocationError)?;

        // Wait on response
        let result = rx.await.map_err(|_| InvocationError)?;
        command
            .create_followup_message(&ctx.http, |response| match &result {
                Ok((imgs, info)) => {
                    let info = {
                        let slice = info.as_bytes().get(0..info.len() - 0);
                        if let Some(info) = slice {
                            let inf: WebuiInfo = serde_json::from_slice(info)
                                .inspect_err(|e| error!(e = ?e, full = std::str::from_utf8(info).unwrap(), "failed to get info"))
                                .unwrap_or(WebuiInfo {
                                infotexts: vec!["error".to_string()]
                            });
                            inf.infotexts[0].clone()
                        } else {
                            "nothing".to_string()
                        }
                    };
                    // info!("{info}");
                    let resp = response                       
                        .content(&command.user)
                        .embed(|embed| { 
                            let mut embed = embed 
                                .title("Generation result")
                                .attachment("result.png")
                                .field("Parameters", info, false);

                            if let Some(ref att) = self.init_image {
                                // embed = embed.field("Original image", "", false);
                                embed = embed.thumbnail(&att.url)
                            }

                            embed
                            // TODO: Original img for img2img
                        });
                        // FIXME: Actually only one image processed
                        resp.add_files(imgs.iter().map(|img| (img.as_slice(), "result.png")))
                }
                Err(err) => {
                    response
                        .content(&command.user)
                        .embed(|embed| embed
                            .title("Error generating response")
                            .description(err)
                        )
                }
            })
            .await
            .map_err(|e| {error!(?e, "dream failed"); InvocationError })?;
        Ok(())
    }
}
