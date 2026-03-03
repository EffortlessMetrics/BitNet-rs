//! Template registry and enumeration API.
//!
//! Enumerate, search, and query available prompt templates.

/// Template metadata.
#[derive(Debug, Clone)]
pub struct TemplateInfo {
    pub name: String,
    pub family: String,
    pub format: TemplateFormat,
    pub supports_system: bool,
    pub supports_multi_turn: bool,
    pub description: String,
}

/// Template format classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemplateFormat {
    ChatML,
    Llama2,
    Llama3,
    Alpaca,
    Vicuna,
    Zephyr,
    Phi,
    Custom,
}

impl TemplateFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ChatML => "chatml",
            Self::Llama2 => "llama2",
            Self::Llama3 => "llama3",
            Self::Alpaca => "alpaca",
            Self::Vicuna => "vicuna",
            Self::Zephyr => "zephyr",
            Self::Phi => "phi",
            Self::Custom => "custom",
        }
    }

    pub fn all() -> &'static [TemplateFormat] {
        &[
            Self::ChatML,
            Self::Llama2,
            Self::Llama3,
            Self::Alpaca,
            Self::Vicuna,
            Self::Zephyr,
            Self::Phi,
            Self::Custom,
        ]
    }
}

/// Registry of all known templates.
#[derive(Debug, Clone)]
pub struct TemplateRegistry {
    templates: Vec<TemplateInfo>,
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TemplateRegistry {
    pub fn new() -> Self {
        Self { templates: Vec::new() }
    }

    /// Build registry with all known templates.
    pub fn builtin() -> Self {
        let mut reg = Self::new();
        // ChatML family
        for name in
            &["chatml", "qwen", "qwen2.5", "yi", "internlm", "deepseek", "dolphin", "tinyllama"]
        {
            reg.add(TemplateInfo {
                name: name.to_string(),
                family: "chatml".into(),
                format: TemplateFormat::ChatML,
                supports_system: true,
                supports_multi_turn: true,
                description: format!("ChatML-based template for {name}"),
            });
        }
        // Llama family
        for name in &["llama2", "llama3", "codellama"] {
            reg.add(TemplateInfo {
                name: name.to_string(),
                family: "llama".into(),
                format: if *name == "llama3" {
                    TemplateFormat::Llama3
                } else {
                    TemplateFormat::Llama2
                },
                supports_system: true,
                supports_multi_turn: true,
                description: format!("Meta LLaMA template for {name}"),
            });
        }
        // Phi family
        for name in &["phi3", "phi4"] {
            reg.add(TemplateInfo {
                name: name.to_string(),
                family: "phi".into(),
                format: TemplateFormat::Phi,
                supports_system: true,
                supports_multi_turn: true,
                description: format!("Microsoft Phi template for {name}"),
            });
        }
        // Other formats
        reg.add(TemplateInfo {
            name: "alpaca".into(),
            family: "alpaca".into(),
            format: TemplateFormat::Alpaca,
            supports_system: false,
            supports_multi_turn: false,
            description: "Stanford Alpaca instruction format".into(),
        });
        reg.add(TemplateInfo {
            name: "vicuna".into(),
            family: "vicuna".into(),
            format: TemplateFormat::Vicuna,
            supports_system: true,
            supports_multi_turn: true,
            description: "LMSYS Vicuna chat format".into(),
        });
        reg.add(TemplateInfo {
            name: "zephyr".into(),
            family: "zephyr".into(),
            format: TemplateFormat::Zephyr,
            supports_system: true,
            supports_multi_turn: true,
            description: "HuggingFace Zephyr format".into(),
        });
        reg
    }

    pub fn add(&mut self, info: TemplateInfo) {
        self.templates.push(info);
    }

    pub fn count(&self) -> usize {
        self.templates.len()
    }

    pub fn all(&self) -> &[TemplateInfo] {
        &self.templates
    }

    pub fn get(&self, name: &str) -> Option<&TemplateInfo> {
        self.templates.iter().find(|t| t.name == name)
    }

    /// List all template names.
    pub fn names(&self) -> Vec<&str> {
        self.templates.iter().map(|t| t.name.as_str()).collect()
    }

    /// Filter by format.
    pub fn by_format(&self, fmt: TemplateFormat) -> Vec<&TemplateInfo> {
        self.templates.iter().filter(|t| t.format == fmt).collect()
    }

    /// Filter by family.
    pub fn by_family(&self, family: &str) -> Vec<&TemplateInfo> {
        self.templates.iter().filter(|t| t.family == family).collect()
    }

    /// Templates that support system messages.
    pub fn with_system_support(&self) -> Vec<&TemplateInfo> {
        self.templates.iter().filter(|t| t.supports_system).collect()
    }

    /// Templates that support multi-turn.
    pub fn with_multi_turn(&self) -> Vec<&TemplateInfo> {
        self.templates.iter().filter(|t| t.supports_multi_turn).collect()
    }

    /// Search by name substring.
    pub fn search(&self, query: &str) -> Vec<&TemplateInfo> {
        let q = query.to_lowercase();
        self.templates
            .iter()
            .filter(|t| {
                t.name.to_lowercase().contains(&q) || t.description.to_lowercase().contains(&q)
            })
            .collect()
    }

    /// Unique families.
    pub fn families(&self) -> Vec<String> {
        let mut fams: Vec<_> = self.templates.iter().map(|t| t.family.clone()).collect();
        fams.sort();
        fams.dedup();
        fams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_count() {
        let reg = TemplateRegistry::builtin();
        assert!(reg.count() >= 15);
    }

    #[test]
    fn test_get_existing() {
        let reg = TemplateRegistry::builtin();
        let t = reg.get("chatml").unwrap();
        assert_eq!(t.format, TemplateFormat::ChatML);
    }

    #[test]
    fn test_get_missing() {
        let reg = TemplateRegistry::builtin();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_by_format() {
        let reg = TemplateRegistry::builtin();
        let chatml = reg.by_format(TemplateFormat::ChatML);
        assert!(chatml.len() >= 5);
    }

    #[test]
    fn test_by_family() {
        let reg = TemplateRegistry::builtin();
        let llama = reg.by_family("llama");
        assert!(llama.len() >= 2);
    }

    #[test]
    fn test_with_system_support() {
        let reg = TemplateRegistry::builtin();
        let sys = reg.with_system_support();
        assert!(sys.len() > 10);
    }

    #[test]
    fn test_with_multi_turn() {
        let reg = TemplateRegistry::builtin();
        let mt = reg.with_multi_turn();
        assert!(mt.len() > 5);
    }

    #[test]
    fn test_search() {
        let reg = TemplateRegistry::builtin();
        let results = reg.search("phi");
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_names() {
        let reg = TemplateRegistry::builtin();
        let names = reg.names();
        assert!(names.contains(&"chatml"));
        assert!(names.contains(&"llama3"));
    }

    #[test]
    fn test_families() {
        let reg = TemplateRegistry::builtin();
        let fams = reg.families();
        assert!(fams.contains(&"chatml".to_string()));
        assert!(fams.contains(&"phi".to_string()));
    }

    #[test]
    fn test_format_all() {
        assert_eq!(TemplateFormat::all().len(), 8);
    }

    #[test]
    fn test_format_str() {
        assert_eq!(TemplateFormat::ChatML.as_str(), "chatml");
        assert_eq!(TemplateFormat::Llama3.as_str(), "llama3");
    }

    #[test]
    fn test_add_custom() {
        let mut reg = TemplateRegistry::new();
        reg.add(TemplateInfo {
            name: "custom".into(),
            family: "test".into(),
            format: TemplateFormat::Custom,
            supports_system: false,
            supports_multi_turn: false,
            description: "test".into(),
        });
        assert_eq!(reg.count(), 1);
    }
}
