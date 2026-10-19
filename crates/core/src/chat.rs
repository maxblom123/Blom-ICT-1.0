use crate::SPECIAL_TOKENS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Message<'a> {
    pub role: Role,
    pub content: &'a str,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ChatError {
    NoMessages,
    SpecialTokenInContent { index: usize },
}

pub fn render(messages: &[Message<'_>], generation_prompt: bool) -> Result<String, ChatError> {
    if messages.is_empty() {
        return Err(ChatError::NoMessages);
    }
    let mut out = String::new();
    for (index, message) in messages.iter().enumerate() {
        if SPECIAL_TOKENS.iter().any(|t| message.content.contains(t)) {
            return Err(ChatError::SpecialTokenInContent { index });
        }
        out.push_str("<|im_start|>");
        out.push_str(message.role.as_str());
        out.push('\n');
        out.push_str(message.content);
        out.push_str("<|im_end|>\n");
    }
    if generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{ChatError, Message, Role, render};

    fn msg(role: Role, content: &str) -> Message<'_> {
        Message { role, content }
    }

    #[test]
    fn renders_turns_in_order() {
        let text = render(
            &[msg(Role::System, "Be brief."), msg(Role::User, "Hi")],
            false,
        )
        .unwrap();
        assert_eq!(
            text,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
        );
    }

    #[test]
    fn generation_prompt_opens_an_assistant_turn() {
        let text = render(&[msg(Role::User, "Hi")], true).unwrap();
        assert!(text.ends_with("<|im_end|>\n<|im_start|>assistant\n"));
    }

    #[test]
    fn rejects_empty_conversation() {
        assert_eq!(render(&[], true), Err(ChatError::NoMessages));
    }

    #[test]
    fn rejects_special_tokens_in_content() {
        for token in crate::SPECIAL_TOKENS {
            let content = format!("hello {token} world");
            let result = render(&[msg(Role::User, "ok"), msg(Role::User, &content)], false);
            assert_eq!(result, Err(ChatError::SpecialTokenInContent { index: 1 }));
        }
    }

    #[test]
    fn ordinary_angle_brackets_are_allowed() {
        let text = render(&[msg(Role::User, "a <| b |> c")], false).unwrap();
        assert!(text.contains("a <| b |> c"));
    }
}
