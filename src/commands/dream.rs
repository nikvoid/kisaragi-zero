use std::{sync::atomic::{AtomicBool, Ordering}, time::Duration, str::FromStr, io::Cursor};

use anyhow::anyhow;
use image::{RgbImage, codecs::png::{PngDecoder, PngReader}, ImageFormat, imageops::FilterType};
use once_cell::sync::Lazy;
use serenity::builder::{CreateEmbed, CreateInteractionResponseFollowup};
use serenity::model::prelude::Attachment;
use tokio::{sync::oneshot::error::TryRecvError, task::spawn_blocking};
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

/// Wrapper for choosing backend
macro_rules! sd_fill {
    ($req:expr) => {
        match $crate::CONFIG.sdapi_backend {
            $crate::config::SdapiBackend::Webui => $crate::sdapi::Webui::fill_request($req)
        }    
    }
}

const PADDING: u32 = 5;
const SCALING_COEF: f64 = 0.7;

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
                                match msg.progress_tx.send((progress, (0, 1))) {
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
                }

                DreamType::Matrix(cmd) => {
                    // Parse matrix
                    let result: DreamResult = try {
                        let matrix = MatrixSize::from_str(&cmd.matrix_size)?;

                        // Spawn watcher task
                        let (curr_img_sendr, curr_img_watch) = watch::channel(0);
                        let images = matrix.area();
                        tokio::spawn(async move {
                            loop {
                                // FIXME: Implement overall steps report
                                // Idea: calculate base from processed img count, add received to result
                                if let Ok(progress) = sd_progress!().await {
                                    let report = (progress, (*curr_img_watch.borrow(), images));
                                    match msg.progress_tx.send(report) {
                                        Ok(_) => tokio::time::sleep(UPDATE_INTERVAL).await,
                                        Err(_) => break,
                                }}
                            }
                        });

                        let mut req = cmd.into_request().await?;
                        sd_fill!(&mut req);

                        let img_h = req.height.unwrap();
                        let img_w = req.width.unwrap();
                        let buf_h = 
                            img_h * matrix.height()
                             + PADDING * 2 * matrix.height();
                        let buf_w = 
                            img_w * matrix.width()
                            + PADDING * 2 * matrix.width();

                        let mut buff = RgbImage::new(buf_w, buf_h);

                        // Process matrix
                        for ((strength, steps), (y, x)) in matrix.iter().zip(matrix.iter_yx()) {
                            let mut req = req.clone();
                            req.steps = Some(steps);
                            req.strength = Some(strength);
                            let png = sd_generate!(req).await?;
                            let img = spawn_blocking(move || {
                                let mut reader = image::io::Reader::new(Cursor::new(&png.0[0]));
                                reader.set_format(ImageFormat::Png);
                                reader.decode()
                                    .map(|img| img.into_rgb8())                           
                            }).await??;

                            let x_offset = PADDING + (img_w + PADDING) * x;
                            let y_offset = PADDING + (img_h + PADDING) * y;
                        
                            image::imageops::overlay(&mut buff, &img, x_offset as i64, y_offset as i64);

                            curr_img_sendr.send_modify(|img| *img += 1);
                        }
                        let buff = image::imageops::resize(
                            &buff,
                            (buf_w as f64 * SCALING_COEF) as u32, 
                            (buf_h as f64 * SCALING_COEF) as u32, 
                            FilterType::Lanczos3);
                        let mut out = Cursor::new(vec![]);
                        buff.write_to(&mut out, ImageFormat::Png)?;

                        (vec![out.into_inner()], req, start_time.elapsed())
                    };

                    msg.back_tx.send(result).ok();                  
                }
            }
            GENERATING.store(false, Ordering::SeqCst);
        }
    }); 
    tx
});

fn get_queue_size() -> usize {
    let busy = GENERATING.load(Ordering::Relaxed);
    DREAM_POOL.max_capacity() - DREAM_POOL.capacity() + busy as usize
}


type DreamResult = anyhow::Result<(ImageVec, GenerationRequest, Duration)>;

/// Current image, all images
type ProgressImages = (Progress, Progress);

struct DreamTask {
    back_tx: oneshot::Sender<DreamResult>,
    progress_tx: watch::Sender<ProgressImages>,
    kind: DreamType
}

enum DreamType {
    Dream(DreamCommand),
    Matrix(DreamMatrixCommand),
}

#[derive(Clone, Copy)]
enum MatrixSize {
    M2x2,
    M3x3,
    M5x5,
    M5x7,
}

impl MatrixSize {
    /// Steps (x) of matrix
    fn steps(self) -> &'static [u32] {
        match self {
            Self::M2x2   => &[8, 12],
            Self::M3x3   => &[8, 12, 16],
            Self::M5x5 
            | Self::M5x7 => &[8, 12, 16, 24, 32],
        }
    }

    /// Sterngths (y) of matrix
    fn strengths(self) -> &'static [f64] {
        match self {
            Self::M2x2 => &[0.35, 0.75],
            Self::M3x3 => &[0.35, 0.50, 0.75],
            Self::M5x5 => &[0.35, 0.45, 0.55, 0.65, 0.75],
            Self::M5x7 => &[0.35, 0.40, 0.45, 0.55, 0.60, 0.70, 0.80],
        }
    }

    fn height(self) -> u32 {
        self.strengths().len() as u32
    }

    fn width(self) -> u32 {
        self.steps().len() as u32
    }

    /// Sum of all images' steps in matrix
    fn overall_steps(self) -> u32 {
        self.steps().iter().map(|s| s * self.height() as u32).sum()
    }

    /// Return iterator over matrix
    fn iter(self) -> impl Iterator<Item = (f64, u32)> {
        self.strengths()
            .iter()
            .flat_map(move |&s| [s]
                .into_iter()
                .cycle()
                .zip(self.steps().iter().copied())
            )
    }

    /// Iterator over coords
    fn iter_yx(self) -> impl Iterator<Item = (u32, u32)> {
        (0..self.height())
            .flat_map(move |y| [y]
                .into_iter()
                .cycle()
                .zip(0..self.width())
            )
    }

    /// Matrix area
    fn area(self) -> u32 {
        self.width() * self.height()
    }

    /// Print table with matrix parameters
    fn table(self) -> String {
        const COL_W: usize = 4;
        let mut out = String::from("steps (x):  ");

        for steps in self.steps() {
            out.push_str(&format!("{steps:<COL_W$}"))
        }

        out.push_str("\nstrengths (y):");
        
        for strength in self.strengths() {
            out.push_str(&format!("\n{strength}"))
        }

        out
    }
}


impl FromStr for MatrixSize {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "2x2" => Self::M2x2,
            "3x3" => Self::M3x3,
            "5x5" => Self::M5x5,
            "5x7" => Self::M5x7,
            _ => anyhow::bail!("Unknown matrix size")  
        })
    }
}

/// Request image matrix generation
#[derive(Command, Clone)]
#[name = "dream_matrix"]
pub struct DreamMatrixCommand {
    /// Prompt. Quality improving keywords will be appended
    prompt: String,
    /// Negative prompt. Will be appended to default keywords
    negative_prompt: Option<String>,
    /// Size of matrix. steps x strength
    #[choice("2x2")]
    #[choice("3x3")]
    #[choice("5x5")]
    #[choice("5x7")]
    matrix_size: String,
    /// Resulting width
    #[min = 1]
    #[max = 2048]
    width: Option<i64>,
    /// Resulting height
    #[min = 1]
    #[max = 2048]
    height: Option<i64>,
    /// Images sampler
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
    /// Guidance scale
    #[min = 0]
    #[max = 100]
    cfg_scale: Option<i64>,
    /// Do not add default keywords to prompt
    no_default_prompt: Option<bool>,
    /// Do not add default keywords to negative prompt
    no_default_neg_prompt: Option<bool>,
    /// LoRa that will be appended to prompt.
    /// Only for webui backend
    #[choice("ryukishi")]
    lora: Option<String>,
    /// Initial image to use in img2img
    init_image: Option<Attachment>,
}

impl DreamMatrixCommand {    
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
            steps: None,
            scale: self.cfg_scale.map(|s| s as _),
            strength: None,
            width: self.width.map(|w| w as _),
            height: self.height.map(|h| h as _),
            batch: None,
            no_default_prompt: self.no_default_prompt.unwrap_or(false),
            no_default_neg_prompt: self.no_default_neg_prompt.unwrap_or(false),
            init_images: match self.init_image {
                Some(init) => Some(vec![init.download().await?]),
                None => None,
            },
        })
    }
}

#[async_trait]
impl ApplicationCommandInteractionHandler for DreamMatrixCommand {
    async fn invoke(
        &self,
        ctx: &Context,
        command: &ApplicationCommandInteraction,
    ) -> Result<(), InvocationError> {
        // Send queue pos response
        let queue_pos = get_queue_size();
        let msg = if !crate::is_admin(command.user.id.0) {
            "Only for admins".to_string()
        } else {
            format!("Queue position: #{queue_pos}")
        };
        
        command
            .create_interaction_response(&ctx.http, |response| response
                .kind(InteractionResponseType::ChannelMessageWithSource)
                .interaction_response_data(|message| message.content(msg))
            )
            .await
            .map_err(|_| InvocationError)?;

        // Admin check
        if !crate::is_admin(command.user.id.0) {
            return Ok(())   
        } 

        let cmd = self.clone();

        // Send command to executor with oneshot and watcher channel
        let (tx, mut rx) = oneshot::channel();
        let (watch_tx, mut watch_rx) = watch::channel(((0, 0), (0, 0)));

        let task = DreamTask {
            back_tx: tx,
            progress_tx: watch_tx,
            kind: DreamType::Matrix(cmd)
        };

        let matrix = MatrixSize::from_str(&self.matrix_size).map_err(|_| InvocationError)?;
        
        DREAM_POOL.send(task)
            .await
            .map_err(|_| InvocationError)?;

        loop {
            tokio::select! {
                biased;
                Ok(_) = watch_rx.changed() => {
                    let ((step, steps), (img, imgs)) = *watch_rx.borrow();
                    let percents = (step as f64 / steps as f64) * 100.;
                    
                    let res = command.edit_original_interaction_response(&ctx.http, |response| response
                        .content(format!(concat!(
                            "Queue position: #{queue_pos}\n",
                            "Status: {step}/{steps} | {percents:.2}%\n",
                            "Images: {img}/{imgs}"),
                            queue_pos = queue_pos,
                            step = step,
                            steps = steps,
                            percents = percents,
                            img = img,
                            imgs = imgs
                        ))
                    ).await;

                    if let Err(e) = res {
                        error!(?e, "failed to edit message");
                    }
                },
                res = async { rx.try_recv() } => { 
                    match res {
                        // Generation end, return result
                        Ok(mut res) => {
                            command.create_followup_message(&ctx.http, |response| {
                                if let Ok((_, ref mut info, _)) = res {
                                    info.steps = None;
                                    info.strength = None;
                                }
                                create_dream_result(
                                    &self.init_image,
                                    response, 
                                    &res, 
                                    command);

                                response.embed(|embed| embed
                                    .title("========= Matrix Parameters =========")
                                    .description(matrix.table())
                                )
                            })
                            .await
                            .map_err(|e| {error!(?e, "matrix failed"); InvocationError })?;
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

    
}


#[async_trait]
impl ApplicationCommandInteractionHandler for DreamCommand {
    async fn invoke(
        &self,
        ctx: &Context,
        command: &ApplicationCommandInteraction,
    ) -> Result<(), InvocationError> {

        // Send queue pos response
        let queue_pos = get_queue_size();
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
        let (watch_tx, mut watch_rx) = watch::channel(((0, 0), (0, 0)));

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
                    // TODO: Process multiple batch case
                    let ((step, steps), _) = *watch_rx.borrow();
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
                                create_dream_result(
                                    &self.init_image,
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

fn create_dream_result<'a, 'b>(
    init_image: &'b Option<Attachment>,
    response: &'b mut CreateInteractionResponseFollowup<'a>,
    result: &'a DreamResult,
    command: &ApplicationCommandInteraction,
) -> &'b mut CreateInteractionResponseFollowup<'a> {
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

                    if let Some(ref att) = init_image {
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