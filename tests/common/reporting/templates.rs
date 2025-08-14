//! Template utilities for report generation

use std::collections::HashMap;

/// Simple template engine for report generation
pub struct TemplateEngine {
    templates: HashMap<String, String>,
}

impl TemplateEngine {
    /// Create a new template engine
    pub fn new() -> Self {
        Self { templates: HashMap::new() }
    }

    /// Register a template with the given name
    pub fn register_template(&mut self, name: &str, template: &str) {
        self.templates.insert(name.to_string(), template.to_string());
    }

    /// Render a template with the given variables
    pub fn render(
        &self,
        template_name: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String, String> {
        let template = self
            .templates
            .get(template_name)
            .ok_or_else(|| format!("Template '{}' not found", template_name))?;

        let mut result = template.clone();
        for (key, value) in variables {
            let placeholder = format!("{{{{{}}}}}", key);
            result = result.replace(&placeholder, value);
        }

        Ok(result)
    }

    /// Load default templates for HTML reports
    pub fn load_default_html_templates(&mut self) {
        self.register_template("html_base", include_str!("templates/html_base.html"));
        self.register_template("test_suite_row", include_str!("templates/test_suite_row.html"));
        self.register_template("test_case_row", include_str!("templates/test_case_row.html"));
    }
}

impl Default for TemplateEngine {
    fn default() -> Self {
        let mut engine = Self::new();
        engine.load_default_html_templates();
        engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_rendering() {
        let mut engine = TemplateEngine::new();
        engine.register_template("test", "Hello {{name}}! You have {{count}} messages.");

        let mut variables = HashMap::new();
        variables.insert("name".to_string(), "Alice".to_string());
        variables.insert("count".to_string(), "5".to_string());

        let result = engine.render("test", &variables).unwrap();
        assert_eq!(result, "Hello Alice! You have 5 messages.");
    }

    #[test]
    fn test_missing_template() {
        let engine = TemplateEngine::new();
        let variables = HashMap::new();

        let result = engine.render("nonexistent", &variables);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Template 'nonexistent' not found"));
    }
}
