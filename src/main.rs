#![feature(fs_try_exists)]
#![feature(try_blocks)]
#![feature(result_option_inspect)]

use once_cell::sync::Lazy;
use serenity::{
    async_trait,
    client::{Context, EventHandler},
    model::{
        id::GuildId,
        application::interaction::Interaction,
        prelude::{Ready, Message},
    },
    prelude::GatewayIntents,
    Client,
};
use slashies::{
    register_commands, Commands,
};
use tracing::{warn, info, error};

mod commands;
use commands::*;

mod config;
use config::Config;

mod sdapi;

struct Handler;

#[async_trait]
impl EventHandler for Handler {
    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        match interaction {
            Interaction::ApplicationCommand(command_interaction) => {
                BotCommands::parse(&ctx, &command_interaction)
                    .expect("Failed to parse command")
                    .invoke(&ctx, &command_interaction)
                    .await
                    .expect("Failed to invoke command");
            }
            _ => (),
        }
    }

    async fn message(&self, ctx: Context, mut msg: Message) {
        info!(
            msg = msg.content,
            guild = ?msg.guild_id,
            author = msg.author.name,
            "message"
        );
        if msg.content.starts_with(&CONFIG.prefix) {
            match try_send_emoji(&mut msg, &ctx).await {
                Some(Ok(_)) => return,
                Some(Err(e)) => error!(?e, "Failed to send emoji"),
                None => ()
            }

            match msg.content.strip_prefix(&CONFIG.prefix).unwrap() {
                "emojis" => if let Err(e) = send_emoji_list(&msg, &ctx).await {
                    error!(?e, "Failed to send emoji list")
                },
                _ => ()
            } 
        }
    }

    #[allow(deprecated)]
    async fn ready(&self, ctx: Context, ready: Ready) {
        info!("{} is connected!", ready.user.name);

        let target = CONFIG.target_guild.map(GuildId);
        
        let commands = register_commands!(&ctx, target, [
            HelloCommand,
            RollCommand,
            DreamCommand
        ])
        .expect("Unable to register commands");
                
        info!("Registered {} commands", commands.len());
    }
}

static CONFIG: Lazy<Config> = Lazy::new(|| 
    Config::load("config.toml").expect("Failed to load config")
);

#[tokio::main]
async fn main() -> anyhow::Result<()>  {
    tracing_subscriber::fmt::init();

    Client::builder(&CONFIG.token, GatewayIntents::all())
        .event_handler(Handler)
        .application_id(CONFIG.app_id)
        .await?
        .start()
        .await?;

    Ok(())
}
