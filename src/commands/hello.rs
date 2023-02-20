use super::prelude::*;

/// Greet a user
#[derive(Debug, Command)]
#[name = "hello"]
pub struct HelloCommand {
    /// The user to greet
    user: UserInput,
}

#[async_trait]
impl ApplicationCommandInteractionHandler for HelloCommand {
    async fn invoke(
        &self,
        ctx: &Context,
        command: &ApplicationCommandInteraction,
    ) -> Result<(), InvocationError> {
        let nickname = self
            .user
            .member
            .as_ref()
            .and_then(|pm| pm.nick.as_ref());
        let greeting = if let Some(nick) = nickname {
            format!("Hello {} aka {}", self.user.user.name, nick)
        } else {
            format!("Hello {}", self.user.user.name)
        };
        command
            .create_interaction_response(&ctx.http, |response| {
                response
                    .kind(InteractionResponseType::ChannelMessageWithSource)
                    .interaction_response_data(|message| message.content(greeting))
            })
            .await
            .map_err(|_| InvocationError)
    }
}
