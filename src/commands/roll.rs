use rand::Rng;
use super::prelude::*;

/// Greet a user
#[derive(Debug, Command)]
#[name = "roll"]
pub struct RollCommand {
    /// Dice to roll
    #[choice("d2")]
    #[choice("d10")]
    #[choice("d20")]
    dice: String,
    /// Roll count times
    count: Option<i64>,
}

#[async_trait]
impl ApplicationCommandInteractionHandler for RollCommand {
    async fn invoke(
        &self,
        ctx: &Context,
        command: &ApplicationCommandInteraction,
    ) -> Result<(), InvocationError> {
        let user = &command.user.name;

        let dice = self.dice.as_str();
        let range = match dice {
            "d2" => 1..=2,
            "d10" => 1..=10,
            "d20" => 1..=20,            
            _ => panic!("unknown dice")
        };

        let msg = match self.count {
            Some(cnt @ 1..=100) => {
                let mut buf = format!("{user} rolls {dice} {cnt} times: ");
                for _ in 1..cnt {
                    let res = rand::thread_rng().gen_range(range.clone());
                    buf.push_str(&res.to_string());
                    buf.push_str(", ");
                }
                buf
            },
            Some(_) => {
                String::from("Cannot roll more than 100 times at once")
            }
            None => {
                let res = rand::thread_rng().gen_range(range);
                format!("{user} rolls {dice}: {res}")
            }
        };

        command
            .create_interaction_response(&ctx.http, |response| {
                response
                    .kind(InteractionResponseType::ChannelMessageWithSource)
                    .interaction_response_data(|message| message.embed(|e| e
                        .title(msg)
                    ))
            })
            .await
            .map_err(|_| InvocationError)
    }
}
