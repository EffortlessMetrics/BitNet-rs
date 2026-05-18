/// Client information.
#[derive(Debug, Clone, Default)]
pub struct ClientInfo {
    pub ip: Option<String>,
    pub user_agent: Option<String>,
    pub api_key_id: Option<String>,
}
