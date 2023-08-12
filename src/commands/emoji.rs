use super::prelude::*;
use serenity::model::prelude::Message;
use serenity::prelude::SerenityError;

const EMOJI_DIR: &str = "emoji";

/// Try to parse command input and check if it is emoji command.
///
/// Returns embed with emoji and requester name, if emoji exists
pub async fn try_send_emoji(
    msg: &Message,
    ctx: &Context,
) -> Option<Result<Message, SerenityError>> {
    let emo = &msg.content.as_str().strip_prefix(&crate::CONFIG.prefix).unwrap();

    // Prevent path attack
    for c in emo.chars() {
        if !c.is_ascii_lowercase() && !c.is_ascii_digit() && c != '_' {
            return None;
        }
    }
        
    let emo_path = format!("{EMOJI_DIR}/{emo}.png");
    let author = &msg.author.name;

    match std::fs::try_exists(&emo_path) {
        Ok(true) => (),
        _ => return None
    }

    if let Err(e) = msg.delete(&ctx.http).await {
        error!(?e, "Failed to delete message");
    };

    let emoji = match tokio::fs::File::open(emo_path).await {
        Ok(f) => f,
        Err(e) => return Some(Err(e.into()))
    };

    let res = msg.channel_id.send_message(&ctx.http, |msg| msg
        .embed(|embed| embed
            .title(format!("{author}:"))                                
            .attachment("emo.png")
        )
        .add_file((&emoji, "emo.png"))
    ).await;

    Some(res)
}

/// Send list of available emojis
pub async fn send_emoji_list(msg: &Message, ctx: &Context) -> Result<Message, SerenityError> {
    let mut out = String::from("```");
    let mut iter = tokio::fs::read_dir(EMOJI_DIR).await?;

    while let Some(e) = iter.next_entry().await? {
        if let Some(name) = e.file_name().to_str().and_then(|n| n.strip_suffix(".png")) {
            out.push_str(name);
            out.push('\n');
        }
    }

    // pop last LF
    out.pop();
    out.push_str("```");
    
    msg.channel_id.send_message(&ctx.http, |msg| msg.content(out)).await
}