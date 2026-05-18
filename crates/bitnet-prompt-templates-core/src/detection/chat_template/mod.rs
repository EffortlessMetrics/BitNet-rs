//! Chat-template Jinja detection orchestration.
//!
//! The matcher list, logging, and public detection entry point live in separate
//! submodules so adding a new GGUF `tokenizer.chat_template` signature does not
//! require touching tracing or fallback control flow.

mod logging;
mod signatures;

use crate::TemplateType;

pub(super) fn detect(jinja: &str) -> Option<TemplateType> {
    let template = signatures::detect(jinja)?;
    logging::detected(template);
    Some(template)
}
