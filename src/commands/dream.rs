use std::{sync::atomic::{AtomicBool, Ordering}, time::Duration};

use once_cell::sync::Lazy;
use serenity::builder::{CreateEmbed, CreateInteractionResponseFollowup};
use serenity::model::prelude::Attachment;
use tokio::sync::oneshot::error::TryRecvError;
use tokio::sync::{watch, oneshot, mpsc};
use crate::sdapi::{GenerationRequest, SdApi, ImageVec, Progress};
use super::prelude::*;

/// Wrapper for choosing backend
macro_rules! sd_generate {
    ($req:expr) => { 
        match $crate::CONFIG.sdapi_backend {
            $crate::config::SdapiBackend::Webui => $crate::sdapi::WEBUI.generate($req)
        }    
    } 
}

/// Wrapper for choosing backend
macro_rules! sd_progress {
    () => { 
        match $crate::CONFIG.sdapi_backend {
            $crate::config::SdapiBackend::Webui => $crate::sdapi::WEBUI.progress()
        }    
    } 
}

/// Progress watcher update interval
const UPDATE_INTERVAL: Duration = Duration::from_secs(5);

/// Is currently generating
static GENERATING: AtomicBool = AtomicBool::new(false);

/// Dream command executor
static DREAM_POOL: Lazy<mpsc::Sender<DreamTask>> = Lazy::new(|| { 
    let (tx, mut rx) = mpsc::channel::<DreamTask>(32);
    tokio::spawn(async move { 
        // Fetch commands
        while let Some(msg) = rx.recv().await {
            // Indicate state
            GENERATING.store(true, Ordering::SeqCst);
            // Capture time
            let start_time = tokio::time::Instant::now();        
            match msg.kind {
                DreamType::Dream(cmd) => {
                    // Spawn watcher task
                    tokio::spawn(async move {
                        loop {
                            if let Ok(progress) = sd_progress!().await {
                                match msg.progress_tx.send(progress) {
                                    Ok(_) => tokio::time::sleep(UPDATE_INTERVAL).await,
                                    Err(_) => break,
                            }}
                        }
                    });
                    
                    let res = cmd.into_request()
                        .await
                        .map(|req| sd_generate!(req));
                    match res {
                        Ok(r) => {
                            let res = r.await.map(|r| (r.0, r.1, start_time.elapsed()));
                            msg.back_tx.send(res).ok()
                        }
                        Err(e) => msg.back_tx.send(Err(e)).ok(),
                    };
                },
            }
            GENERATING.store(false, Ordering::SeqCst);
        }
    }); 
    tx
});


type DreamResult = anyhow::Result<(ImageVec, GenerationRequest, Duration)>;

struct DreamTask {
    back_tx: oneshot::Sender<DreamResult>,
    progress_tx: watch::Sender<Progress>,
    kind: DreamType
}

enum DreamType {
    Dream(DreamCommand)
}

/// Request image generation
#[derive(Command, Clone)]
#[name = "dream_ng"]
pub struct DreamCommand {
    /// Prompt. Quality improving keywords will be appended
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
    /// Do not add default keywords to prompt
    no_default_prompt: Option<bool>,
    /// Do not add default keywords to negative prompt
    no_default_neg_prompt: Option<bool>,
    /// Count of images to generate. Will be ignored if not from admin
    #[min = 1]
    #[max = 16]
    batch_size: Option<i64>,
    /// LoRa that will be appended to prompt.
    /// Only for webui backend
    #[choice("ryukishi")]
    lora: Option<String>,
}

impl DreamCommand {
    /// Convert dream command input into request, optionally downloading image
    async fn into_request(mut self) -> anyhow::Result<GenerationRequest> {
        let lora = self.lora.map(|lora| match lora.as_str() {
            "ryukishi" => ", <lora:ryukishi07Higurashi_v1:1>",
            _ => "",
        });

        if let Some(lora) = lora {
            self.prompt.push_str(lora);
        }
        
        Ok(GenerationRequest {
            prompt: self.prompt,
            neg_prompt: self.negative_prompt,
            seed: self.seed,
            sampler: self.sampler_name,
            steps: self.steps.map(|s| s as _),
            scale: self.cfg_scale.map(|s| s as _),
            strength: self.denoising_strength,
            width: self.width.map(|w| w as _),
            height: self.height.map(|h| h as _),
            batch: self.batch_size.map(|b| b as _),
            no_default_prompt: self.no_default_prompt.unwrap_or(false),
            no_default_neg_prompt: self.no_default_neg_prompt.unwrap_or(false),
            init_images: match self.init_image {
                Some(init) => Some(vec![init.download().await?]),
                None => None,
            },
        })
    }

    fn create_dream_result<'a, 'b>(
        &'b self,
        response: &'b mut CreateInteractionResponseFollowup<'a>,
        result: &'a DreamResult,
        command: &ApplicationCommandInteraction,
    ) -> &mut CreateInteractionResponseFollowup<'a> {
        match result {
            Ok((imgs, info, comp_time)) => {
                let imgs: Vec<_> = imgs
                    .iter()
                    .enumerate()
                    .map(|(idx, img)|
                        (img, format!("{idx}.png"))
                    )
                    .collect();
                let mut img_iter = imgs.iter();
                let resp = response                       
                    .content(&command.user)
                    .embed(|embed| { 
                        let mut embed = embed 
                            .title("Generation result")
                            .description(format!("Compute used: {:.2} sec", comp_time.as_secs_f32()))
                            .field("Prompt", &info.prompt, false)                                
                            .opt_field("Negative prompt", info.neg_prompt.as_ref(), false)
                            .opt_field("Seed", info.seed, true)
                            .opt_field("Sampler", info.sampler.as_ref(), true)
                            .opt_field("Steps", info.steps, true)
                            .opt_field("Scale", info.scale, true)
                            .opt_field("Strength", info.strength, true)
                            .opt_field("Width", info.width, true)
                            .opt_field("Height", info.height, true)
                            .attachment("result.png");

                        if let Some(ref att) = self.init_image {
                            embed = embed.thumbnail(&att.url)
                        }

                        embed
                    });
                    if let Some((first, _)) = img_iter.next() {
                        resp.add_file((first.as_slice(), "result.png"));
                    }
                    resp.add_files(img_iter.map(|(img, name)| (img.as_slice(), name.as_str())))
            }
            Err(err) => {
                response
                    .content(&command.user)
                    .embed(|embed| embed
                        .title("Error generating response")
                        .description(err)
                    )
            }
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

        // Send queue pos response
        let busy = GENERATING.load(Ordering::Relaxed);
        let queue_pos = DREAM_POOL.max_capacity() - DREAM_POOL.capacity() + busy as usize;
        command
            .create_interaction_response(&ctx.http, |response| response
                .kind(InteractionResponseType::ChannelMessageWithSource)
                .interaction_response_data(|message| message
                    .content(format!("Queue position: #{queue_pos}")))
            )
            .await
            .map_err(|_| InvocationError)?;

        let mut cmd = self.clone();

        // Admin check
        if !crate::is_admin(command.user.id.0) {
            cmd.batch_size.replace(1);
        } 
        
        // Send command to executor with oneshot and watcher channel
        let (tx, mut rx) = oneshot::channel();
        let (watch_tx, mut watch_rx) = watch::channel((0, 0));

        let task = DreamTask {
            back_tx: tx,
            progress_tx: watch_tx,
            kind: DreamType::Dream(cmd)
        };
        
        DREAM_POOL.send(task)
            .await
            .map_err(|_| InvocationError)?;

        loop {
            tokio::select! {
                biased;
                Ok(_) = watch_rx.changed() => {
                    let (step, steps) = *watch_rx.borrow();
                    let percents = (step as f64 / steps as f64) * 100.;
                    
                    let res = command.edit_original_interaction_response(&ctx.http, |response| response
                        .content(format!(
                            "Queue position: #{queue_pos}\nStatus: {step}/{steps} | {percents:.2}%"
                        ))
                    ).await;

                    if let Err(e) = res {
                        error!(?e, "failed to edit message");
                    }
                },
                res = async { rx.try_recv() } => { 
                    match res {
                        // Generation end, return result
                        Ok(res) => {
                            command.create_followup_message(&ctx.http, |response| 
                                self.create_dream_result(
                                    response, 
                                    &res, 
                                    command
                            ))
                            .await
                            .map_err(|e| {error!(?e, "dream failed"); InvocationError })?;
                            return Ok(())
                        }
                        // In progress, wait for more
                        Err(TryRecvError::Empty) => {
                            tokio::time::sleep(Duration::from_millis(20)).await;
                            continue
                        }
                        // Unknown fail
                        _ => return Err(InvocationError)
                    }
                },
                else => {}   
            }
        }
    }
}


trait CreateEmbedExt {
    fn opt_field<T, U>(&mut self, name: T, value: Option<U>, inline: bool) -> &mut Self
    where     
        T: ToString,
        U: ToString;
}

impl CreateEmbedExt for CreateEmbed {
    fn opt_field<T, U>(&mut self, name: T, value: Option<U>, inline: bool) -> &mut Self
    where     
        T: ToString,
        U: ToString 
    {
        if let Some(val) = value {
            self.field(name, val, inline)
        } else {
            self
        }
    }
}
