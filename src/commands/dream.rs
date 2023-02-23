use std::{sync::atomic::{AtomicBool, Ordering}, time::Duration, str::FromStr, io::Cursor};

use image::{RgbImage, ImageFormat, imageops::FilterType, Rgb};
use imageproc::drawing;
use once_cell::sync::Lazy;
use rusttype::{Font, Scale};
use serenity::{builder::CreateEmbed, http::Http, model::prelude::Message};
use serenity::model::prelude::Attachment;
use tokio::task::spawn_blocking;
use tokio::sync::{watch, mpsc};
use crate::{sd_fill, sd_generate, sd_progress};
use crate::sdapi::{GenerationRequest, ImageVec, Progress};
use super::prelude::*;

/// Progress watcher update interval
const UPDATE_INTERVAL: Duration = Duration::from_secs(5);

/// Is currently generating
static GENERATING: AtomicBool = AtomicBool::new(false);

/// Font used to draw matrix parameters
static FONT: Lazy<Font<'static>> = Lazy::new(|| 
    Font::try_from_bytes(include_bytes!("../../res/nunito.ttf")).expect("Failed to load font")
); 

/// Dream command executor
static DREAM_POOL: Lazy<mpsc::Sender<DreamTask>> = Lazy::new(|| { 
    let (tx, mut rx) = mpsc::channel::<DreamTask>(32);
    tokio::spawn(async move { 
        // Fetch commands
        while let Some(msg) = rx.recv().await {
            // Indicate state
            GENERATING.store(true, Ordering::SeqCst);
            // Capture time
            let progress_tx = msg.back_tx.clone();
            match msg.kind {
                DreamType::Dream(cmd) => {
                    // Spawn watcher task
                    tokio::spawn(async move {
                        loop {
                            if let Ok(progress) = sd_progress!().await {
                                match progress_tx.send(DreamOutput::Progress(progress, (0, 1))).await {
                                    Ok(_) => tokio::time::sleep(UPDATE_INTERVAL).await,
                                    Err(_) => break,
                            }}
                        }
                    });

                    let start_time = tokio::time::Instant::now();        
       
                    let res = cmd.into_request()
                        .await
                        .map(|req| sd_generate!(req));
                    match res {
                        Ok(r) => {
                            let res = r.await.map(|r| (r.0, r.1, start_time.elapsed()));
                            msg.back_tx.send(DreamOutput::Result(res)).await.ok()
                        }
                        Err(e) => msg.back_tx.send(DreamOutput::Result(Err(e))).await.ok(),
                    };
                }

                DreamType::Matrix(cmd) => {
                    let result = generate_matrix(cmd, progress_tx).await;
                    msg.back_tx.send(DreamOutput::Result(result)).await.ok();                  
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

/// Generate matrix of images. Periodically sends back progress
///
/// Feature: if `init_image` is `None`, in output matrix rows will be different 
/// CFG Scale instead of Strength
async fn generate_matrix(
    cmd: DreamMatrixCommand, 
    progress_tx: mpsc::Sender<DreamOutput>
) -> DreamResult {
    // consts
    const FONT_SCALE: Scale = Scale { x: 64., y: 64. };
    const PADDING: u32 = 5;
    const UL_MARGIN: u32 = 120;
    const DEF_SCALING_COEF: f64 = 1.;
    /// Discord limit
    const MAX_PNG_SIZE: u64 = 8_388_284; 

    // Parse matrix size
    let matrix = MatrixSize::from_str(&cmd.matrix_size)?;

    let mut req = cmd.into_request().await?;
    sd_fill!(&mut req);
    let use_cfg_scale = req.init_images.is_none();

    // Spawn watcher task
    let (curr_img_sendr, curr_img_watch) = watch::channel(0);
    let images = matrix.area();
    tokio::spawn(async move {
        loop {
            // FIXME: Implement overall steps report
            // Idea: calculate base from processed img count, add received to result
            // Buuuuut... Requested steps aren't always correspond to real...
            if let Ok((cur_steps, _)) = sd_progress!().await {
                let processed_imgs = *curr_img_watch.borrow();
                // We can fold matrix iterator to find overall steps count
                let steps_made = cur_steps + matrix.iter(use_cfg_scale)
                    .take(processed_imgs)
                    .map(|(_, steps)| steps)
                    .sum::<u32>(); 
                let steps_progress = (steps_made, matrix.overall_steps());
                let imgs_progress = (processed_imgs as u32, images);
                let report = DreamOutput::Progress(steps_progress, imgs_progress);
                match progress_tx.send(report).await {
                    Ok(_) => tokio::time::sleep(UPDATE_INTERVAL).await,
                    Err(_) => break,
            }}
        }
    });


    let img_h = req.height.unwrap();
    let img_w = req.width.unwrap();
    let buf_h = 
        UL_MARGIN + img_h * matrix.height()
         + PADDING * 2 * matrix.height();
    let buf_w = 
        UL_MARGIN + img_w * matrix.width()
        + PADDING * 2 * matrix.width();

    let mut buff = RgbImage::new(buf_w, buf_h);

    let start_time = tokio::time::Instant::now();        

    // Process matrix, compose images
    for ((strength_scale, steps), (y, x)) in matrix.iter(use_cfg_scale).zip(matrix.iter_yx()) {
        let mut req = req.clone();
        req.steps = Some(steps);
        
        if use_cfg_scale {
            req.scale = Some(strength_scale as u32);
        } else {
            req.strength = Some(strength_scale);
        }
        
        let png = sd_generate!(req).await?;
        let img = spawn_blocking(move || {
            let mut reader = image::io::Reader::new(Cursor::new(&png.0[0]));
            reader.set_format(ImageFormat::Png);
            reader.decode()
                .map(|img| img.into_rgb8())                           
        }).await??;

        let x_offset = UL_MARGIN + PADDING + (img_w + PADDING * 2) * x;
        let y_offset = UL_MARGIN + PADDING + (img_h + PADDING * 2) * y;
    
        image::imageops::overlay(&mut buff, &img, x_offset as i64, y_offset as i64);
        curr_img_sendr.send_modify(|img| *img += 1);
    }

    let mut draw_text = |x, y, text: String| imageproc::drawing::draw_text_mut(
        &mut buff, 
        Rgb([0xFF, 0xFF, 0xFF]),
        x,
        y,
        FONT_SCALE,
        &FONT,
        &text
    );

    // Draw steps    
    for (x, steps) in matrix.steps().iter().enumerate() {
        let col_offset = UL_MARGIN + PADDING + (img_w + PADDING * 2) * x as u32;

        let label = format!("{steps}");
        let (text_w, text_h) = drawing::text_size(FONT_SCALE, &FONT, &label);

        let x_off = (img_w - text_w as u32) / 2 + col_offset;
        let y_off = (UL_MARGIN - text_h as u32) / 2;

        draw_text(x_off as i32, y_off as i32, label);
    } 
    
    // Draw strengths (or cfg scales)
    let strength_scales = if !use_cfg_scale {
        matrix.strengths()
    } else {
        matrix.cfg_scales()
    };
    
    for (y, strength_scale) in strength_scales.iter().enumerate() {
        let row_offset = UL_MARGIN + PADDING + (img_h + PADDING * 2) * y as u32;

        let label = if use_cfg_scale {
            format!("{strength_scale:.0}")
        } else {
            format!("{strength_scale:.2}")
        };
        
        let (text_w, text_h) = drawing::text_size(FONT_SCALE, &FONT, &label);

        let y_off = (img_h - text_h as u32) / 2 + row_offset;
        let x_off = (UL_MARGIN - text_w as u32) / 2;
        draw_text(x_off as i32, y_off as i32, label);
    } 

    // Try to make image smaller while it not fits into discord restriction
    let mut scale_coef = DEF_SCALING_COEF;
    let mut out = Cursor::new(vec![]);
    while out.position() == 0 || out.position() >= MAX_PNG_SIZE {
        out = Cursor::new({ 
            let mut buf = out.into_inner();
            buf.clear();
            buf
        });

        let buff = image::imageops::resize(
            &buff,
            (buf_w as f64 * scale_coef) as u32, 
            (buf_h as f64 * scale_coef) as u32, 
            FilterType::Lanczos3);
        buff.write_to(&mut out, ImageFormat::Png)?;
        scale_coef -= 0.1;
    } 

    Ok((vec![out.into_inner()], req, start_time.elapsed()))
}

type DreamResult = anyhow::Result<(ImageVec, GenerationRequest, Duration)>;

struct DreamTask {
    back_tx: mpsc::Sender<DreamOutput>,
    kind: DreamType
}

enum DreamType {
    Dream(DreamCommand),
    Matrix(DreamMatrixCommand),
}

enum DreamOutput {
    Result(DreamResult),
    /// Steps, images
    Progress(Progress, Progress),
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

    /// Ad-hoc alternative mode of generation - CFG scales (y)
    fn cfg_scales(self) -> &'static [f64] {
        match self {
            Self::M2x2 => &[7., 9.],
            Self::M3x3 => &[6., 7., 9.],
            Self::M5x5 => &[6., 7., 9., 12., 15.],
            Self::M5x7 => &[5., 6., 7.,  9., 12., 15., 18.],
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
        self.steps().iter().map(|s| s * self.height()).sum()
    }

    /// Return iterator over matrix.
    ///
    /// If `cfg_scale = true`, iterator will be over cfg scales 
    fn iter(self, cfg_scale: bool) -> impl Iterator<Item = (f64, u32)> {
        if !cfg_scale { self.strengths() } else { self.cfg_scales() } 
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

/// Request image matrix generation.
/// If init_image specified, generation will be with different steps and strengths
/// Otherwise with steps and CFG Scales
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
        let (tx, mut rx) = mpsc::channel(4);
        let task = DreamTask {
            back_tx: tx,
            kind: DreamType::Matrix(cmd)
        };

        DREAM_POOL.send(task)
            .await
            .map_err(|_| InvocationError)?;

        while let Some(msg) = rx.recv().await {
            match msg {
                DreamOutput::Progress((step, steps), (img, imgs)) => {
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
                }
                DreamOutput::Result(mut res) => {
                    if let Ok((_, ref mut info, _)) = res {
                        info.steps = None;
                        info.strength = None;
                    }
                    create_dream_result(
                        &ctx.http,
                        &self.init_image,
                        &res, 
                        command
                    )
                    .await
                    .map_err(|e| {error!(?e, "matrix failed"); InvocationError })?;
                    return Ok(())
                }                
            }
        }
        Err(InvocationError)
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
        let (tx, mut rx) = mpsc::channel(4);

        let task = DreamTask {
            back_tx: tx,
            kind: DreamType::Dream(cmd)
        };
        
        DREAM_POOL.send(task)
            .await
            .map_err(|_| InvocationError)?;

        while let Some(msg) = rx.recv().await {
            match msg {
                DreamOutput::Progress((step, steps), _) => { 
                    let percents = (step as f64 / steps as f64) * 100.;
                    
                    let res = command.edit_original_interaction_response(&ctx.http, |response| response
                        .content(format!(
                            "Queue position: #{queue_pos}\nStatus: {step}/{steps} | {percents:.2}%"
                        ))
                    ).await;

                    if let Err(e) = res {
                        error!(?e, "failed to edit message");
                    } 
                }
                DreamOutput::Result(res) => {
                    create_dream_result(
                        &ctx.http,
                        &self.init_image,
                        &res, 
                        command
                    )
                    .await
                    .map_err(|e| {error!(?e, "dream failed"); InvocationError })?;
                    return Ok(()) 
                }
            }
        }
        Err(InvocationError)
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

/// Send followup message (or fallback to normal message if interaction token invalidated)
/// with dreaming result 
async fn create_dream_result<'a, 'b>(
    http: impl AsRef<Http>,
    init_image: &'b Option<Attachment>,
    result: &'a DreamResult,
    command: &ApplicationCommandInteraction,
) -> Result<Message, serenity::Error> {
    match result {
        Ok((imgs, info, comp_time)) => {
            let imgs: Vec<_> = imgs
                .iter()
                .enumerate()
                .map(|(idx, img)|
                    (img, format!("{idx}.png"))
                )
                .collect();
            let files_iter = imgs.iter().map(|(img, name)| (img.as_slice(), name.as_str()));

            let create_embed = for<'e> |embed: &'e mut CreateEmbed| -> &'e mut CreateEmbed 
            { embed 
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
                    embed.thumbnail(&att.url);
                }

                embed
            };

            let res = command.create_followup_message(&http, |response| { response                       
                .content(&command.user)
                .embed(create_embed);

                // add file to embed or send separated
                if imgs.len() == 1 {
                    response.add_file((imgs[0].0.as_slice(), "result.png"))
                } else {
                    response.add_files(files_iter.clone())
                }
            }).await;

            // Fallback to message if interaction was invalidated
            if res.is_err() {
                command.channel_id.send_message(&http, |msg| { msg
                    .content(&command.user)
                    .embed(create_embed);
                 
                    // add file to embed or send separated
                    if imgs.len() == 1 {
                        msg.add_file((imgs[0].0.as_slice(), "result.png"))
                    } else {
                        msg.add_files(files_iter)
                    }
                }).await
            } else {
                res
            }
        }
        Err(err) => {
            command.channel_id.send_message(&http, |msg| msg
                .content(&command.user)
                .embed(|embed| embed
                    .title("Error generating response")
                    .description(err)
                )
            ).await
        }
    }
}