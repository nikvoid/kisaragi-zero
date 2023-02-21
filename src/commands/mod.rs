

pub mod prelude {
    pub use serenity::{
        async_trait,
        client::{Context, EventHandler},
        model::{
            id::GuildId,
            application::interaction::{
                application_command::ApplicationCommandInteraction, Interaction,
                InteractionResponseType,
            },
            prelude::Ready,
        },
        prelude::GatewayIntents,
        Client,
    };
    pub use slashies::{
        parsable::*, register_commands, ApplicationCommandInteractionHandler, Commands,
        InvocationError,
    };
    pub use slashies_macros::{Command, Commands};
    pub use tracing::{warn, info, error};
}

use prelude::*;

mod emoji;
mod hello;
mod roll;
mod dream;

pub use hello::HelloCommand;
pub use roll::RollCommand;
pub use dream::{DreamCommand, DreamMatrixCommand};
pub use emoji::{try_send_emoji, send_emoji_list};

#[derive(slashies_macros::Commands)]
pub enum BotCommands {
    Hello(HelloCommand),
    Roll(RollCommand),
    Dream(DreamCommand),
    DreamMatrix(DreamMatrixCommand)
}
