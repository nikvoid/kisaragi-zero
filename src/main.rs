#![feature(fs_try_exists)]
#![feature(try_blocks)]
#![feature(closure_lifetime_binder)]
#![feature(async_fn_in_trait)]

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
    register_commands, Commands
};
use tracing::{info, error};

mod commands;
use commands::*;

mod config;
use config::Config;

mod sdapi;

struct Handler;

#[allow(clippy::single_match)]
#[async_trait]
impl EventHandler for Handler {
    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        match interaction {
            Interaction::ApplicationCommand(interaction) => {
                let member = interaction.member.as_ref().unwrap();

                if CONFIG.has_rights(member.user.id, &interaction.data.name) {
                    BotCommands::parse(&ctx, &interaction)
                        .expect("Failed to parse command")
                        .invoke(&ctx, &interaction)
                        .await
                        .expect("Failed to invoke command");
                } else {
                    interaction.create_interaction_response(&ctx.http, |resp| resp
                        .interaction_response_data(|data| data
                            .content("You don't have rights to use this command")
                        )
                    )
                    .await
                    .expect("Failed to send response");
                }                
            }
            _ => (),
        }
    }

    async fn message(&self, ctx: Context, msg: Message) {
        info!(
            msg = msg.content,
            guild = ?msg.guild_id,
            author = msg.author.name,
            "message"
        );
        if let Some(cmd) = msg.content.strip_prefix(&CONFIG.prefix) {
            if CONFIG.has_rights(msg.author.id, "emojis") {
                match try_send_emoji(&msg, &ctx).await {
                    Some(Ok(_)) => return,
                    Some(Err(e)) => error!(?e, "Failed to send emoji"),
                    None => ()
                }
            }

            if !CONFIG.has_rights(msg.author.id, cmd) {
                msg.reply(&ctx.http, "You don't have rights to use this command")
                    .await
                    .expect("Failed to reply");
                return
            }

            match cmd {
                "emojis" => if let Err(e) = send_emoji_list(&msg, &ctx).await {
                    error!(?e, "Failed to send emoji list")
                },
                _ => ()
            } 
        }
    }

    async fn ready(&self, ctx: Context, ready: Ready) {
        info!("{} is connected!", ready.user.name);

        for guild in &CONFIG.target_guilds {
            match register_commands!(&ctx, Some(GuildId(*guild)), [
                HelloCommand,
                RollCommand,
                DreamCommand,
                DreamMatrixCommand
            ]) {
                Ok(cmds) =>         
                    info!(guild, "Registered {} commands", cmds.len()),
                Err(e) =>
                    error!(guild, "Unable to regsiter commands: {e}")
            };
        }
    }
}

pub static CONFIG: Lazy<Config> = Lazy::new(|| {
    let cfg = std::env::args().nth(1).unwrap_or("config.toml".into());
    Config::load(&cfg).expect("Failed to load config")
});

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
