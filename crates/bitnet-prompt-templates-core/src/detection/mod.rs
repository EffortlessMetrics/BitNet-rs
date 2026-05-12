mod chat_template;
mod tokenizer_name;

use super::TemplateType;

pub(crate) fn detect(
    tokenizer_name: Option<&str>,
    chat_template_jinja: Option<&str>,
) -> TemplateType {
    chat_template_jinja
        .and_then(chat_template::detect)
        .or_else(|| tokenizer_name.and_then(tokenizer_name::detect))
        .unwrap_or_else(|| {
            tracing::warn!(template = "Raw", "no template signature found; falling back to Raw");
            TemplateType::Raw
        })
}
