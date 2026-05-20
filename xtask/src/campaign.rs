use anyhow::{Context, Result, bail};
use clap::Subcommand;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const CAMPAIGNS_DIR: &str = "docs/tracking/campaigns";
const GENERATED_DIR: &str = "docs/tracking/generated";
const GENERATED_HEADER: &str = "<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->\n";

const WORK_ITEM_STATUSES: &[&str] =
    &["proposed", "ready", "in_progress", "pr_open", "blocked", "merged", "superseded"];

const CAMPAIGN_STATUSES: &[&str] = &["proposed", "active", "blocked", "complete", "archived"];
const EVENT_TYPES: &[&str] = &[
    "in_progress",
    "pr_open",
    "blocked",
    "superseded",
    "merged",
    "closeout",
    "migrated_from_bitnet_rs",
];
const REVIEW_MODES: &[&str] = &["codex_premerge", "human_required", "external_required", "none"];
const MERGE_POLICIES: &[&str] =
    &["automerge_when_green", "codex_merge_when_green", "manual_only", "no_merge"];
const HUMAN_GATES: &[&str] = &["never", "on_blocker_only", "before_merge", "always"];
const REQUIRED_CAMPAIGNS: &[&str] = &[
    "apple-m4",
    "apple-m4-local-answer",
    "apple-m4-operational",
    "apple-m4-slm-answer",
    "apple-m4-slm-hardening",
    "cpu-proof",
    "cpu-qk256-performance",
    "intel-a770",
    "intel-npu",
    "intel-258v-platform",
    "nvidia-5070ti",
    "amd-cpu-baselines",
    "crate-collapse",
    "server-real-inference",
    "ci-coverage",
    "tracker-infra",
];

#[derive(Subcommand)]
pub enum CampaignCmd {
    /// List campaign manifests.
    List,
    /// Print one campaign's status.
    Status { campaign: String },
    /// Print the next runnable item for a campaign.
    Next { campaign: String },
    /// Validate one campaign manifest and event log.
    Check { campaign: String },
    /// Generate campaign and global dashboards.
    Generate {
        /// Check that generated dashboards are current without writing files.
        #[arg(long, default_value_t = false)]
        check: bool,
    },
    /// Run cross-campaign advisory checks.
    Doctor,
}

pub fn run(cmd: CampaignCmd) -> Result<()> {
    let root = std::env::current_dir().context("resolve current directory")?;
    match cmd {
        CampaignCmd::List => cmd_list(&root),
        CampaignCmd::Status { campaign } => cmd_status(&root, &campaign),
        CampaignCmd::Next { campaign } => cmd_next(&root, &campaign),
        CampaignCmd::Check { campaign } => cmd_check(&root, &campaign),
        CampaignCmd::Generate { check } => cmd_generate(&root, check),
        CampaignCmd::Doctor => cmd_doctor(&root),
    }
}

#[derive(Debug, Deserialize)]
struct CampaignManifest {
    id: String,
    title: String,
    status: String,
    #[serde(default)]
    objective: String,
    #[serde(default)]
    end_state: Vec<String>,
    #[serde(default)]
    hard_constraints: Vec<String>,
    #[serde(default, rename = "work_item")]
    work_items: Vec<WorkItem>,
}

#[derive(Clone, Debug, Deserialize)]
struct WorkItem {
    id: String,
    status: String,
    branch: String,
    #[serde(default)]
    stackable: Option<bool>,
    #[serde(default)]
    requires_human_merge: Option<bool>,
    #[serde(default)]
    review_mode: Option<String>,
    #[serde(default)]
    merge_policy: Option<String>,
    #[serde(default)]
    human_gate: Option<String>,
    #[serde(default)]
    blocked_by: Vec<String>,
    #[serde(default)]
    acceptance: Option<TextList>,
    #[serde(default)]
    commands: Vec<String>,
    #[serde(default)]
    allowed_paths: Vec<String>,
    #[serde(default)]
    forbidden_paths: Vec<String>,
    #[serde(default)]
    may_claim: Vec<String>,
    #[serde(default)]
    must_not_claim: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum TextList {
    One(String),
    Many(Vec<String>),
}

impl TextList {
    fn summary(&self) -> String {
        match self {
            TextList::One(value) => value.clone(),
            TextList::Many(values) => values.join("; "),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            TextList::One(value) => value.trim().is_empty(),
            TextList::Many(values) => values.iter().all(|value| value.trim().is_empty()),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct Event {
    timestamp: String,
    campaign: String,
    item: String,
    event: String,
    #[serde(default)]
    pr: Option<u64>,
    #[serde(default)]
    head_sha: Option<String>,
    #[serde(default)]
    merge_sha: Option<String>,
    #[serde(default)]
    actor: Option<String>,
    #[serde(default)]
    notes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct GithubPullRequest {
    number: u64,
    #[serde(default)]
    title: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    merged_at: Option<String>,
    #[serde(default)]
    merge_commit_sha: Option<String>,
    #[serde(default)]
    labels: Vec<GithubLabel>,
}

#[derive(Debug, Deserialize)]
struct GithubLabel {
    name: String,
}

struct GithubContext {
    repository: String,
    token: String,
    current_pr_number: Option<u64>,
}

struct LoadedCampaign {
    dir: PathBuf,
    manifest: CampaignManifest,
    events: Vec<Event>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Severity {
    Error,
    Warning,
}

#[derive(Debug)]
struct Problem {
    severity: Severity,
    message: String,
}

impl Problem {
    fn error(message: impl Into<String>) -> Self {
        Self { severity: Severity::Error, message: message.into() }
    }

    fn warning(message: impl Into<String>) -> Self {
        Self { severity: Severity::Warning, message: message.into() }
    }
}

fn cmd_list(root: &Path) -> Result<()> {
    for campaign in load_all_campaigns(root)? {
        println!(
            "{}\t{}\t{}\t{} items",
            campaign.manifest.id,
            campaign.manifest.status,
            campaign.manifest.title,
            campaign.manifest.work_items.len()
        );
    }
    Ok(())
}

fn cmd_status(root: &Path, campaign_id: &str) -> Result<()> {
    let campaign = load_campaign(root, campaign_id)?;
    let active = current_item(&campaign.manifest);
    println!("campaign: {}", campaign.manifest.id);
    println!("title: {}", campaign.manifest.title);
    println!("status: {}", campaign.manifest.status);
    if !campaign.manifest.objective.is_empty() {
        println!("objective: {}", campaign.manifest.objective);
    }
    match active {
        Some(item) => {
            println!("active_item: {}", item.id);
            println!("item_status: {}", item.status);
            if let Some(pr) = latest_pr(&campaign.events, &item.id) {
                println!("pr: #{pr}");
            }
        }
        None => println!("active_item: none"),
    }
    Ok(())
}

fn cmd_next(root: &Path, campaign_id: &str) -> Result<()> {
    let campaign = load_campaign(root, campaign_id)?;
    let next = next_runnable_item(&campaign.manifest);

    match next {
        Some(item) => {
            println!("campaign: {}", campaign.manifest.id);
            println!("next_item: {}", item.id);
            println!("status: {}", item.status);
            println!("branch: {}", item.branch);
            println!("review_mode: {}", item.review_mode.as_deref().unwrap_or("<missing>"));
            println!("merge_policy: {}", item.merge_policy.as_deref().unwrap_or("<missing>"));
            println!("human_gate: {}", item.human_gate.as_deref().unwrap_or("<missing>"));
            if let Some(acceptance) = &item.acceptance {
                println!("acceptance: {}", acceptance.summary());
            }
            println!("commands:");
            for command in &item.commands {
                println!("- {command}");
            }
        }
        None => println!("campaign: {}\nnext_item: none", campaign.manifest.id),
    }
    Ok(())
}

fn cmd_check(root: &Path, campaign_id: &str) -> Result<()> {
    let campaign = load_campaign(root, campaign_id)?;
    let problems = validate_campaign(&campaign);
    print_problems(&problems);
    fail_on_errors(&problems)?;
    println!("campaign check passed: {campaign_id}");
    Ok(())
}

fn cmd_generate(root: &Path, check: bool) -> Result<()> {
    let campaigns = load_all_campaigns(root)?;
    let writes = expected_dashboard_writes(root, &campaigns);
    let stale = stale_generated_dashboards(&writes);

    if check {
        if stale.is_empty() {
            println!("generated dashboards are current");
            return Ok(());
        }
        bail!(
            "generated dashboards are stale:\n{}",
            stale.iter().map(|path| format!("- {}", path.display())).collect::<Vec<_>>().join("\n")
        );
    }

    for (path, content) in writes {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create generated directory {}", parent.display()))?;
        }
        fs::write(&path, content).with_context(|| format!("write {}", path.display()))?;
    }
    println!("generated campaign dashboards");
    Ok(())
}

fn cmd_doctor(root: &Path) -> Result<()> {
    let campaigns = load_all_campaigns(root)?;
    let mut problems = Vec::new();
    let campaign_ids: BTreeSet<_> =
        campaigns.iter().map(|campaign| campaign.manifest.id.as_str()).collect();
    for required in REQUIRED_CAMPAIGNS {
        if !campaign_ids.contains(required) {
            problems.push(Problem::error(format!("missing required campaign `{required}`")));
        }
    }

    let mut item_ids: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut branches: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut prs: BTreeMap<u64, Vec<String>> = BTreeMap::new();

    for campaign in &campaigns {
        problems.extend(validate_campaign(campaign));
        for item in &campaign.manifest.work_items {
            item_ids.entry(&item.id).or_default().push(&campaign.manifest.id);
            if !item.branch.trim().is_empty() {
                branches.entry(&item.branch).or_default().push(&item.id);
            }
        }
        for event in &campaign.events {
            if let Some(pr) = event.pr {
                prs.entry(pr).or_default().push(format!("{}:{}", campaign.manifest.id, event.item));
            }
        }
    }

    for (item_id, owners) in item_ids {
        if owners.len() > 1 {
            problems.push(Problem::error(format!(
                "item `{item_id}` appears in multiple campaigns: {}",
                owners.join(", ")
            )));
        }
    }
    for (branch, owners) in branches {
        if owners.len() > 1 {
            problems.push(Problem::error(format!(
                "branch `{branch}` is claimed by multiple items: {}",
                owners.join(", ")
            )));
        }
    }
    for (pr, owners) in prs {
        let unique: BTreeSet<_> = owners.iter().collect();
        if unique.len() > 1 {
            problems.push(Problem::error(format!(
                "PR #{pr} is claimed by multiple items: {}",
                owners.join(", ")
            )));
        }
    }

    let stale_dashboards = stale_generated_dashboards(&expected_dashboard_writes(root, &campaigns));
    if !stale_dashboards.is_empty() {
        problems.push(Problem::error(format!(
            "generated dashboards are stale; run `cargo run -p xtask --no-default-features -- campaign generate`:\n{}",
            stale_dashboards
                .iter()
                .map(|path| format!("- {}", path.display()))
                .collect::<Vec<_>>()
                .join("\n")
        )));
    }

    for path in changed_legacy_tracker_files(root)? {
        problems.push(Problem::error(format!(
            "legacy tracker changed in this branch: {}; normal item PRs should use campaign files",
            path.display()
        )));
    }

    problems.extend(reconcile_github_pull_requests(&campaigns));

    print_problems(&problems);
    fail_on_errors(&problems)?;
    println!("campaign doctor passed");
    Ok(())
}

fn load_all_campaigns(root: &Path) -> Result<Vec<LoadedCampaign>> {
    let campaigns_root = root.join(CAMPAIGNS_DIR);
    let mut dirs = Vec::new();
    for entry in fs::read_dir(&campaigns_root)
        .with_context(|| format!("read {}", campaigns_root.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir() && entry.path().join("active.toml").exists() {
            dirs.push(entry.path());
        }
    }
    dirs.sort();
    dirs.into_iter().map(load_campaign_dir).collect()
}

fn load_campaign(root: &Path, campaign_id: &str) -> Result<LoadedCampaign> {
    load_campaign_dir(root.join(CAMPAIGNS_DIR).join(campaign_id))
}

fn load_campaign_dir(dir: PathBuf) -> Result<LoadedCampaign> {
    let manifest_path = dir.join("active.toml");
    let raw = fs::read_to_string(&manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest: CampaignManifest =
        toml::from_str(&raw).with_context(|| format!("parse {}", manifest_path.display()))?;
    let events = load_events(&dir)?;
    Ok(LoadedCampaign { dir, manifest, events })
}

fn load_events(campaign_dir: &Path) -> Result<Vec<Event>> {
    let events_dir = campaign_dir.join("events");
    if !events_dir.exists() {
        return Ok(Vec::new());
    }

    let mut paths = Vec::new();
    for entry in
        fs::read_dir(&events_dir).with_context(|| format!("read {}", events_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) == Some("toml") {
            paths.push(path);
        }
    }
    paths.sort();

    let mut events = Vec::new();
    for path in paths {
        let raw = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let event: Event =
            toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        events.push(event);
    }
    Ok(events)
}

fn validate_campaign(campaign: &LoadedCampaign) -> Vec<Problem> {
    let mut problems = Vec::new();
    let manifest = &campaign.manifest;
    let dir_id = campaign.dir.file_name().and_then(|value| value.to_str()).unwrap_or_default();

    if manifest.id != dir_id {
        problems.push(Problem::error(format!(
            "campaign id `{}` does not match directory `{dir_id}`",
            manifest.id
        )));
    }
    if !CAMPAIGN_STATUSES.contains(&manifest.status.as_str()) {
        problems.push(Problem::error(format!(
            "campaign `{}` has invalid status `{}`",
            manifest.id, manifest.status
        )));
    }
    if manifest.objective.trim().is_empty() {
        problems.push(Problem::error(format!("campaign `{}` has empty objective", manifest.id)));
    }
    if manifest.end_state.is_empty() {
        problems.push(Problem::warning(format!("campaign `{}` has no end_state", manifest.id)));
    }
    if manifest.hard_constraints.is_empty() {
        problems
            .push(Problem::warning(format!("campaign `{}` has no hard_constraints", manifest.id)));
    }

    let mut seen = BTreeSet::new();
    let item_by_id = item_map(manifest);
    for item in &manifest.work_items {
        if !seen.insert(item.id.as_str()) {
            problems.push(Problem::error(format!(
                "campaign `{}` has duplicate item `{}`",
                manifest.id, item.id
            )));
        }
        if !WORK_ITEM_STATUSES.contains(&item.status.as_str()) {
            problems.push(Problem::error(format!(
                "item `{}` has invalid status `{}`",
                item.id, item.status
            )));
        }
        if item.branch.trim().is_empty() {
            problems.push(Problem::error(format!("item `{}` has empty branch", item.id)));
        }
        if item.stackable.is_none() {
            problems.push(Problem::warning(format!("item `{}` does not set stackable", item.id)));
        }
        if item.requires_human_merge.is_some() {
            problems.push(Problem::error(format!(
                "item `{}` uses deprecated requires_human_merge; use review_mode, merge_policy, and human_gate",
                item.id
            )));
        }
        validate_enum_field(
            &mut problems,
            &item.id,
            "review_mode",
            item.review_mode.as_deref(),
            REVIEW_MODES,
        );
        validate_enum_field(
            &mut problems,
            &item.id,
            "merge_policy",
            item.merge_policy.as_deref(),
            MERGE_POLICIES,
        );
        validate_enum_field(
            &mut problems,
            &item.id,
            "human_gate",
            item.human_gate.as_deref(),
            HUMAN_GATES,
        );
        if item.merge_policy.as_deref() == Some("no_merge") && item.status == "merged" {
            problems.push(Problem::error(format!(
                "item `{}` is merged but has merge_policy=no_merge",
                item.id
            )));
        }
        for dep in &item.blocked_by {
            if !item_by_id.contains_key(dep.as_str()) {
                problems.push(Problem::error(format!(
                    "item `{}` has unknown blocked_by dependency `{dep}`",
                    item.id
                )));
            }
        }
        let acceptance_empty = match item.acceptance.as_ref() {
            Some(acceptance) => acceptance.is_empty(),
            None => true,
        };
        if acceptance_empty {
            problems.push(Problem::error(format!("item `{}` has empty acceptance", item.id)));
        }
        if item.commands.is_empty() {
            problems.push(Problem::error(format!("item `{}` has no commands", item.id)));
        }
        if item.allowed_paths.is_empty() {
            problems.push(Problem::warning(format!("item `{}` has no allowed_paths", item.id)));
        }
        if item.forbidden_paths.is_empty() {
            problems.push(Problem::warning(format!("item `{}` has no forbidden_paths", item.id)));
        }
        if item.status != "proposed" && item.may_claim.is_empty() {
            problems.push(Problem::warning(format!("item `{}` has no may_claim", item.id)));
        }
        if item.status != "proposed" && item.must_not_claim.is_empty() {
            problems.push(Problem::warning(format!("item `{}` has no must_not_claim", item.id)));
        }
    }

    let item_ids: BTreeSet<_> = manifest.work_items.iter().map(|item| item.id.as_str()).collect();
    let mut merged_events = BTreeSet::new();
    let mut pr_open_events_with_pr = BTreeSet::new();
    for event in &campaign.events {
        if event.timestamp.trim().is_empty() {
            problems.push(Problem::error(format!(
                "event for `{}` in campaign `{}` has empty timestamp",
                event.item, manifest.id
            )));
        }
        if event.campaign != manifest.id {
            problems.push(Problem::error(format!(
                "event `{}` points at campaign `{}` but is stored under `{}`",
                event.item, event.campaign, manifest.id
            )));
        }
        if !EVENT_TYPES.contains(&event.event.as_str()) {
            problems.push(Problem::error(format!(
                "event for `{}` has invalid event type `{}`",
                event.item, event.event
            )));
        }
        if !item_ids.contains(event.item.as_str()) {
            problems.push(Problem::error(format!(
                "event references unknown item `{}` in campaign `{}`",
                event.item, manifest.id
            )));
        }
        if event.event == "merged" {
            if event.merge_sha.as_deref().unwrap_or("").trim().is_empty() {
                problems.push(Problem::error(format!(
                    "merged event for `{}` is missing merge_sha",
                    event.item
                )));
            }
            merged_events.insert(event.item.as_str());
        }
        if event.event == "migrated_from_bitnet_rs" {
            if event.merge_sha.as_deref().unwrap_or("").trim().is_empty() {
                problems.push(Problem::error(format!(
                    "migrated_from_bitnet_rs event for `{}` is missing source merge_sha",
                    event.item
                )));
            }
            merged_events.insert(event.item.as_str());
        }
        if event.event == "pr_open" && event.pr.is_none() {
            problems
                .push(Problem::error(format!("pr_open event for `{}` is missing pr", event.item)));
        }
        if event.event == "pr_open" && event.pr.is_some() {
            pr_open_events_with_pr.insert(event.item.as_str());
        }
        if event.event == "pr_open" && event.head_sha.as_deref().unwrap_or("").trim().is_empty() {
            problems.push(Problem::warning(format!(
                "pr_open event for `{}` is missing head_sha",
                event.item
            )));
        }
        if event.actor.as_deref().unwrap_or("").trim().is_empty() {
            problems.push(Problem::warning(format!(
                "event `{}` for `{}` is missing actor",
                event.event, event.item
            )));
        }
        if event.notes.is_empty() {
            problems.push(Problem::warning(format!(
                "event `{}` for `{}` has no notes",
                event.event, event.item
            )));
        }
    }

    for item in &manifest.work_items {
        if item.status == "merged" && !merged_events.contains(item.id.as_str()) {
            problems.push(Problem::error(format!(
                "item `{}` is merged but has no merged event with merge_sha",
                item.id
            )));
        }
        if item.status == "pr_open" && !pr_open_events_with_pr.contains(item.id.as_str()) {
            problems.push(Problem::error(format!(
                "item `{}` is pr_open but has no pr_open event with a PR number",
                item.id
            )));
        }
    }

    problems
}

fn item_map(manifest: &CampaignManifest) -> BTreeMap<&str, &WorkItem> {
    manifest.work_items.iter().map(|item| (item.id.as_str(), item)).collect()
}

fn validate_enum_field(
    problems: &mut Vec<Problem>,
    item_id: &str,
    field: &str,
    value: Option<&str>,
    allowed: &[&str],
) {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        Some(value) if allowed.contains(&value) => {}
        Some(value) => problems.push(Problem::error(format!(
            "item `{item_id}` has invalid {field} `{value}`; expected one of {}",
            allowed.join(", ")
        ))),
        None => problems.push(Problem::error(format!("item `{item_id}` does not set {field}"))),
    }
}

fn deps_met<'a>(item: &WorkItem, item_by_id: &BTreeMap<&'a str, &'a WorkItem>) -> bool {
    item.blocked_by
        .iter()
        .all(|dep| item_by_id.get(dep.as_str()).is_some_and(|dep_item| dep_item.status == "merged"))
}

fn next_runnable_item(manifest: &CampaignManifest) -> Option<&WorkItem> {
    let item_by_id = item_map(manifest);
    manifest
        .work_items
        .iter()
        .find(|item| item.status == "ready" && deps_met(item, &item_by_id))
        .or_else(|| {
            manifest
                .work_items
                .iter()
                .find(|item| item.status == "proposed" && deps_met(item, &item_by_id))
        })
}

fn current_item(manifest: &CampaignManifest) -> Option<&WorkItem> {
    for status in ["pr_open", "in_progress"] {
        if let Some(item) = manifest.work_items.iter().find(|item| item.status == status) {
            return Some(item);
        }
    }
    if let Some(item) = next_runnable_item(manifest) {
        return Some(item);
    }
    if let Some(item) = manifest.work_items.iter().find(|item| item.status == "blocked") {
        return Some(item);
    }
    manifest.work_items.iter().rev().find(|item| item.status == "merged")
}

fn latest_pr(events: &[Event], item_id: &str) -> Option<u64> {
    events.iter().filter(|event| event.item == item_id).filter_map(|event| event.pr).next_back()
}

fn render_campaign_status(campaign: &LoadedCampaign) -> String {
    let mut out = String::new();
    out.push_str(GENERATED_HEADER);
    out.push_str(&format!("# {} Campaign Status\n\n", campaign.manifest.title));
    out.push_str(&format!("- Campaign: `{}`\n", campaign.manifest.id));
    out.push_str(&format!("- State: `{}`\n", campaign.manifest.status));
    out.push_str(&format!("- Objective: {}\n\n", campaign.manifest.objective));
    out.push_str("## Work Items\n\n");
    out.push_str("| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |\n");
    out.push_str("|---|---|---:|---|---|---|---|---|\n");
    for item in &campaign.manifest.work_items {
        let pr = latest_pr(&campaign.events, &item.id)
            .map(|pr| format!("#{pr}"))
            .unwrap_or_else(|| "TBD".to_string());
        let acceptance = item
            .acceptance
            .as_ref()
            .map(TextList::summary)
            .unwrap_or_else(|| "".to_string())
            .replace('|', "\\|");
        out.push_str(&format!(
            "| {} | {} | {} | `{}` | `{}` | `{}` | `{}` | {} |\n",
            item.id,
            item.status,
            pr,
            item.branch,
            policy_display(item.review_mode.as_deref()),
            policy_display(item.merge_policy.as_deref()),
            policy_display(item.human_gate.as_deref()),
            acceptance
        ));
    }
    out.push('\n');
    out.push_str("## Hard Constraints\n\n");
    for constraint in &campaign.manifest.hard_constraints {
        out.push_str(&format!("- {constraint}\n"));
    }
    out
}

fn policy_display(value: Option<&str>) -> &str {
    value.unwrap_or("<missing>")
}

fn render_global_dashboard(campaigns: &[LoadedCampaign]) -> String {
    let mut out = String::new();
    out.push_str(GENERATED_HEADER);
    out.push_str("# BitNet Campaign Dashboard\n\n");
    out.push_str("| Campaign | Active item | PR | State | Next | Notes |\n");
    out.push_str("|---|---|---:|---|---|---|\n");
    for campaign in campaigns {
        let active = current_item(&campaign.manifest);
        let active_id = active.map(|item| item.id.as_str()).unwrap_or("none");
        let state = active.map(|item| item.status.as_str()).unwrap_or("none");
        let pr = active
            .and_then(|item| latest_pr(&campaign.events, &item.id))
            .map(|pr| format!("#{pr}"))
            .unwrap_or_else(|| "TBD".to_string());
        let next = next_after_current(&campaign.manifest, active_id)
            .map(|item| item.id.as_str())
            .unwrap_or("none");
        let note = campaign.manifest.hard_constraints.first().map(String::as_str).unwrap_or("");
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            campaign.manifest.id,
            active_id,
            pr,
            state,
            next,
            note.replace('|', "\\|")
        ));
    }
    out
}

fn next_after_current<'a>(manifest: &'a CampaignManifest, active_id: &str) -> Option<&'a WorkItem> {
    let active_index = manifest.work_items.iter().position(|item| item.id == active_id)?;
    manifest
        .work_items
        .iter()
        .skip(active_index + 1)
        .find(|item| !matches!(item.status.as_str(), "merged" | "superseded"))
}

fn render_active_prs(campaigns: &[LoadedCampaign]) -> String {
    let mut out = String::new();
    out.push_str(GENERATED_HEADER);
    out.push_str("# Active Campaign PRs\n\n");
    out.push_str("| Campaign | Item | PR | Branch | Notes |\n");
    out.push_str("|---|---|---:|---|---|\n");
    for campaign in campaigns {
        for item in &campaign.manifest.work_items {
            if item.status == "pr_open" {
                let pr = latest_pr(&campaign.events, &item.id)
                    .map(|pr| format!("#{pr}"))
                    .unwrap_or_else(|| "TBD".to_string());
                out.push_str(&format!(
                    "| {} | {} | {} | `{}` | {} |\n",
                    campaign.manifest.id,
                    item.id,
                    pr,
                    item.branch,
                    item.acceptance.as_ref().map(TextList::summary).unwrap_or_default()
                ));
            }
        }
    }
    out
}

fn render_lane_dashboard(campaigns: &[LoadedCampaign]) -> String {
    let mut out = String::new();
    out.push_str(GENERATED_HEADER);
    out.push_str("# Campaign Lane Dashboard\n\n");
    out.push_str("| Campaign | Title | Current item | Boundary |\n");
    out.push_str("|---|---|---|---|\n");
    for campaign in campaigns {
        let current =
            current_item(&campaign.manifest).map(|item| item.id.as_str()).unwrap_or("none");
        let boundary = campaign
            .manifest
            .hard_constraints
            .first()
            .map(String::as_str)
            .unwrap_or("")
            .replace('|', "\\|");
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            campaign.manifest.id, campaign.manifest.title, current, boundary
        ));
    }
    out
}

fn render_blocked_items(campaigns: &[LoadedCampaign]) -> String {
    let mut out = String::new();
    out.push_str(GENERATED_HEADER);
    out.push_str("# Blocked Campaign Items\n\n");
    out.push_str("| Campaign | Item | Blocked by | State |\n");
    out.push_str("|---|---|---|---|\n");
    for campaign in campaigns {
        for item in &campaign.manifest.work_items {
            if item.status == "blocked" || !item.blocked_by.is_empty() {
                out.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    campaign.manifest.id,
                    item.id,
                    item.blocked_by.join(", "),
                    item.status
                ));
            }
        }
    }
    out
}

fn expected_dashboard_writes(
    root: &Path,
    campaigns: &[LoadedCampaign],
) -> BTreeMap<PathBuf, String> {
    let mut writes = BTreeMap::new();

    for campaign in campaigns {
        let rel = format!("{CAMPAIGNS_DIR}/{}/generated/status.md", campaign.manifest.id);
        writes.insert(root.join(&rel), render_campaign_status(campaign));
    }

    writes.insert(
        root.join(format!("{GENERATED_DIR}/global-dashboard.md")),
        render_global_dashboard(campaigns),
    );
    writes
        .insert(root.join(format!("{GENERATED_DIR}/active-prs.md")), render_active_prs(campaigns));
    writes.insert(
        root.join(format!("{GENERATED_DIR}/lane-dashboard.md")),
        render_lane_dashboard(campaigns),
    );
    writes.insert(
        root.join(format!("{GENERATED_DIR}/blocked-items.md")),
        render_blocked_items(campaigns),
    );

    writes
}

fn stale_generated_dashboards(writes: &BTreeMap<PathBuf, String>) -> Vec<PathBuf> {
    writes
        .iter()
        .filter_map(|(path, content)| {
            let current = fs::read_to_string(path).ok();
            if current.as_deref() == Some(content.as_str()) { None } else { Some(path.clone()) }
        })
        .collect()
}

fn changed_legacy_tracker_files(root: &Path) -> Result<Vec<PathBuf>> {
    let branch = current_branch_name(root);
    if legacy_tracker_change_exception(&branch) {
        return Ok(Vec::new());
    }

    let output = Command::new("git")
        .args(["diff", "--name-only", "origin/main...HEAD"])
        .current_dir(root)
        .output();
    let Ok(output) = output else {
        return Ok(Vec::new());
    };
    if !output.status.success() {
        return Ok(Vec::new());
    }
    let paths = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|path| {
            matches!(
                *path,
                "docs/tracking/bitnet-alignment/status.md"
                    | "docs/tracking/bitnet-alignment/workstream-ledger.yaml"
            )
        })
        .map(PathBuf::from)
        .collect();
    Ok(paths)
}

fn reconcile_github_pull_requests(campaigns: &[LoadedCampaign]) -> Vec<Problem> {
    let Some(context) = github_context() else {
        return Vec::new();
    };

    let mut problems = Vec::new();
    let client = match reqwest::blocking::Client::builder()
        .user_agent("bitnet-rs-xtask-campaign-doctor")
        .build()
    {
        Ok(client) => client,
        Err(err) => {
            problems
                .push(github_reconciliation_problem(format!("build GitHub client failed: {err}")));
            return problems;
        }
    };

    let open_prs = match fetch_open_pull_requests(&client, &context) {
        Ok(prs) => prs,
        Err(err) => {
            problems.push(github_reconciliation_problem(format!(
                "fetch open GitHub PRs failed: {err}"
            )));
            return problems;
        }
    };

    let mut items = BTreeMap::new();
    let mut pr_open_event_prs = BTreeMap::new();
    let mut merged_events = BTreeSet::new();
    for campaign in campaigns {
        for item in &campaign.manifest.work_items {
            items.insert(item.id.as_str(), (&campaign.manifest.id, item));
        }
        for event in &campaign.events {
            if event.event == "pr_open"
                && let Some(pr) = event.pr
            {
                pr_open_event_prs.insert(event.item.as_str(), pr);
            }
            if event.event == "merged" {
                merged_events.insert(event.item.as_str());
            }
        }
    }

    let mut claims: BTreeMap<&str, Vec<&GithubPullRequest>> = BTreeMap::new();
    let mut closeout_claims = BTreeSet::new();
    let has_merge_closeout_pr = open_prs.iter().any(pull_request_is_merge_closeout);
    for pr in &open_prs {
        for item_id in items.keys() {
            if pull_request_claims_item(pr, item_id) {
                if pull_request_is_merge_closeout(pr) {
                    closeout_claims.insert(*item_id);
                    continue;
                }
                claims.entry(item_id).or_default().push(pr);
            }
        }
    }

    for (item_id, claimed_prs) in &claims {
        if claimed_prs.len() > 1 {
            problems.push(Problem::error(format!(
                "multiple open GitHub PRs claim `{item_id}`: {}",
                claimed_prs
                    .iter()
                    .map(|pr| format!("#{}", pr.number))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }

        let Some((_campaign_id, item)) = items.get(item_id).copied() else {
            continue;
        };
        let pr = claimed_prs[0];
        if context.current_pr_number.is_some_and(|current_pr| pr.number != current_pr) {
            continue;
        }
        if item.status != "pr_open" {
            problems.push(Problem::error(format!(
                "open GitHub PR #{} claims `{item_id}`, but active.toml status is `{}`",
                pr.number, item.status
            )));
        }
        match pr_open_event_prs.get(item_id) {
            Some(recorded_pr) if *recorded_pr == pr.number => {}
            Some(recorded_pr) => problems.push(Problem::error(format!(
                "open GitHub PR #{} claims `{item_id}`, but the latest pr_open event records #{}",
                pr.number, recorded_pr
            ))),
            None => problems.push(Problem::error(format!(
                "open GitHub PR #{} claims `{item_id}`, but no pr_open event records a PR number",
                pr.number
            ))),
        }
    }

    let open_pr_numbers: BTreeSet<_> = open_prs.iter().map(|pr| pr.number).collect();
    for (item_id, (_campaign_id, item)) in &items {
        if item.status != "pr_open" {
            continue;
        }
        let Some(recorded_pr) = pr_open_event_prs.get(item_id) else {
            continue;
        };
        if open_pr_numbers.contains(recorded_pr) {
            if !claims.contains_key(item_id) {
                match fetch_pull_request(&client, &context, *recorded_pr) {
                    Ok(pr) if pull_request_claims_item(&pr, item_id) => continue,
                    Ok(_) => problems.push(Problem::error(format!(
                        "item `{item_id}` is pr_open for GitHub PR #{recorded_pr}, but that open PR does not claim the item in its title, body, or labels"
                    ))),
                    Err(err) => problems.push(github_reconciliation_problem(format!(
                        "fetch GitHub PR #{recorded_pr} for `{item_id}` claim check failed: {err}"
                    ))),
                }
            }
            continue;
        }
        if closeout_claims.contains(item_id) {
            continue;
        }

        match fetch_pull_request(&client, &context, *recorded_pr) {
            Ok(pr) if pr.merged_at.is_some() && has_merge_closeout_pr => {}
            Ok(pr) if pr.merged_at.is_some() && !merged_events.contains(item_id) => {
                let merge_ref = pr
                    .merge_commit_sha
                    .as_deref()
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or("<unknown merge sha>");
                problems.push(Problem::error(format!(
                    "item `{item_id}` is pr_open, but GitHub PR #{} merged at {}; add a merged event with merge_sha `{merge_ref}` and mark the item merged",
                    pr.number,
                    pr.merged_at.unwrap_or_default()
                )));
            }
            Ok(pr) => {
                problems.push(Problem::error(format!(
                    "item `{item_id}` is pr_open for GitHub PR #{}, but the PR is not open",
                    pr.number
                )));
            }
            Err(err) => problems.push(github_reconciliation_problem(format!(
                "fetch GitHub PR #{recorded_pr} for `{item_id}` failed: {err}"
            ))),
        }
    }

    problems
}

fn github_context() -> Option<GithubContext> {
    if std::env::var("GITHUB_EVENT_NAME").ok().as_deref() == Some("push") {
        return None;
    }

    let repository = std::env::var("GITHUB_REPOSITORY").ok();
    let token = std::env::var("GITHUB_TOKEN").or_else(|_| std::env::var("GH_TOKEN")).ok();
    match (repository, token) {
        (Some(repository), Some(token))
            if !repository.trim().is_empty() && !token.trim().is_empty() =>
        {
            Some(GithubContext { repository, token, current_pr_number: current_pr_number() })
        }
        _ => None,
    }
}

fn current_pr_number() -> Option<u64> {
    let event_name = std::env::var("GITHUB_EVENT_NAME").ok()?;
    if !matches!(event_name.as_str(), "pull_request" | "pull_request_target") {
        return None;
    }

    let ref_value =
        std::env::var("GITHUB_REF").ok().or_else(|| std::env::var("GITHUB_REF_NAME").ok())?;
    parse_pull_request_ref(&ref_value)
}

fn parse_pull_request_ref(ref_value: &str) -> Option<u64> {
    let trimmed = ref_value.trim();
    let rest = trimmed
        .strip_prefix("refs/pull/")
        .or_else(|| trimmed.strip_prefix("pull/"))
        .unwrap_or(trimmed);
    let pr_number = rest.split('/').next()?.trim();
    if pr_number.is_empty() { None } else { pr_number.parse().ok() }
}

fn github_reconciliation_problem(message: String) -> Problem {
    if std::env::var("GITHUB_ACTIONS").ok().as_deref() == Some("true") {
        Problem::error(format!("GitHub PR reconciliation failed: {message}"))
    } else {
        Problem::warning(format!("GitHub PR reconciliation skipped: {message}"))
    }
}

fn fetch_open_pull_requests(
    client: &reqwest::blocking::Client,
    context: &GithubContext,
) -> Result<Vec<GithubPullRequest>> {
    let url = format!(
        "https://api.github.com/repos/{}/pulls?state=open&per_page=100",
        context.repository
    );
    client
        .get(url)
        .bearer_auth(&context.token)
        .header("Accept", "application/vnd.github+json")
        .send()
        .context("send GitHub open PR request")?
        .error_for_status()
        .context("GitHub open PR request failed")?
        .json()
        .context("parse GitHub open PR response")
}

fn fetch_pull_request(
    client: &reqwest::blocking::Client,
    context: &GithubContext,
    pr: u64,
) -> Result<GithubPullRequest> {
    let url = format!("https://api.github.com/repos/{}/pulls/{pr}", context.repository);
    client
        .get(url)
        .bearer_auth(&context.token)
        .header("Accept", "application/vnd.github+json")
        .send()
        .with_context(|| format!("send GitHub PR #{pr} request"))?
        .error_for_status()
        .with_context(|| format!("GitHub PR #{pr} request failed"))?
        .json()
        .with_context(|| format!("parse GitHub PR #{pr} response"))
}

fn pull_request_claims_item(pr: &GithubPullRequest, item_id: &str) -> bool {
    let item_label = format!("item:{item_id}").to_ascii_lowercase();
    if pr.labels.iter().any(|label| label.name.eq_ignore_ascii_case(&item_label)) {
        return true;
    }
    if text_contains_item_token(&pr.title, item_id) {
        return true;
    }

    pr.body
        .as_deref()
        .is_some_and(|body| body.lines().any(|line| body_line_claims_item(line, item_id)))
}

fn body_line_claims_item(line: &str, item_id: &str) -> bool {
    let trimmed = line
        .trim()
        .trim_start_matches('-')
        .trim_start_matches('*')
        .trim_start_matches('>')
        .trim()
        .trim_matches('`')
        .trim();
    if text_starts_with_item_token(trimmed, item_id) {
        return true;
    }

    let lower = trimmed.to_ascii_lowercase();
    let explicit_claim = lower.starts_with("work item")
        || lower.starts_with("item:")
        || lower.starts_with("item ")
        || lower.starts_with("scope:")
        || lower.starts_with("boundary:");
    explicit_claim && text_contains_item_token(trimmed, item_id)
}

fn text_starts_with_item_token(text: &str, item_id: &str) -> bool {
    text.strip_prefix(item_id).is_some_and(|rest| match rest.chars().next() {
        Some(ch) => !is_item_token_char(ch),
        None => true,
    })
}

fn text_contains_item_token(text: &str, item_id: &str) -> bool {
    text.match_indices(item_id).any(|(start, matched)| {
        let before = text[..start].chars().next_back();
        let after = text[start + matched.len()..].chars().next();
        let before_boundary = match before {
            Some(ch) => !is_item_token_char(ch),
            None => true,
        };
        let after_boundary = match after {
            Some(ch) => !is_item_token_char(ch),
            None => true,
        };
        before_boundary && after_boundary
    })
}

fn is_item_token_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '-'
}

fn pull_request_is_merge_closeout(pr: &GithubPullRequest) -> bool {
    let mut text = pr.title.to_ascii_lowercase();
    if let Some(body) = &pr.body {
        text.push('\n');
        text.push_str(&body.to_ascii_lowercase());
    }
    ["closeout", "merged", "merge sha", "merge state", "sync"]
        .iter()
        .any(|term| text.contains(term))
}

fn current_branch_name(root: &Path) -> String {
    for key in ["GITHUB_HEAD_REF", "GITHUB_REF_NAME"] {
        if let Ok(value) = std::env::var(key) {
            let value = value.trim().to_string();
            if !value.is_empty() {
                return value;
            }
        }
    }

    Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_default()
}

fn legacy_tracker_change_exception(branch: &str) -> bool {
    ["tracker-infra", "legacy-migration", "generated-dashboard"]
        .iter()
        .any(|marker| branch.contains(marker))
}

fn print_problems(problems: &[Problem]) {
    for problem in problems {
        match problem.severity {
            Severity::Error => eprintln!("error: {}", problem.message),
            Severity::Warning => eprintln!("warning: {}", problem.message),
        }
    }
}

fn fail_on_errors(problems: &[Problem]) -> Result<()> {
    let errors = problems.iter().filter(|problem| problem.severity == Severity::Error).count();
    if errors > 0 {
        bail!("{errors} campaign tracker error(s)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CampaignManifest, LoadedCampaign, Severity, TextList, WorkItem, current_item,
        parse_pull_request_ref, text_contains_item_token, validate_campaign,
    };
    use std::path::PathBuf;

    #[test]
    fn parses_github_pull_request_refs() {
        assert_eq!(parse_pull_request_ref("refs/pull/3721/merge"), Some(3721));
        assert_eq!(parse_pull_request_ref("3722/merge"), Some(3722));
        assert_eq!(parse_pull_request_ref("pull/3723/head"), Some(3723));
    }

    #[test]
    fn rejects_non_pull_request_refs() {
        assert_eq!(parse_pull_request_ref("refs/heads/main"), None);
        assert_eq!(parse_pull_request_ref(""), None);
    }

    #[test]
    fn github_pr_claim_matching_uses_item_token_boundaries() {
        assert!(text_contains_item_token("[codex][apple-m4] Add lane scaffold (M4-001)", "M4-001"));
        assert!(text_contains_item_token("Work item: SLM-M4-001", "SLM-M4-001"));
        assert!(!text_contains_item_token(
            "[codex][apple-m4-slm-answer] Seed campaign (SLM-M4-001)",
            "M4-001"
        ));
        assert!(!text_contains_item_token("Work item: SLM-M4-001", "M4-001"));
    }

    #[test]
    fn validates_work_item_merge_policy_fields() {
        let campaign = policy_campaign(WorkItem {
            id: "POLICY-001".to_string(),
            status: "ready".to_string(),
            branch: "codex/policy/POLICY-001".to_string(),
            stackable: Some(false),
            requires_human_merge: None,
            review_mode: Some("codex_premerge".to_string()),
            merge_policy: Some("automerge_when_green".to_string()),
            human_gate: Some("on_blocker_only".to_string()),
            blocked_by: Vec::new(),
            acceptance: Some(TextList::One("Prove policy validation.".to_string())),
            commands: vec!["cargo fmt --all -- --check".to_string()],
            allowed_paths: vec!["docs/**".to_string()],
            forbidden_paths: vec!["crates/**".to_string()],
            may_claim: vec!["Policy fields are accepted.".to_string()],
            must_not_claim: vec!["Deprecated merge field is accepted.".to_string()],
        });

        assert!(
            validate_campaign(&campaign).iter().all(|problem| problem.severity != Severity::Error)
        );
    }

    #[test]
    fn rejects_deprecated_requires_human_merge() {
        let mut item = WorkItem {
            id: "POLICY-002".to_string(),
            status: "ready".to_string(),
            branch: "codex/policy/POLICY-002".to_string(),
            stackable: Some(false),
            requires_human_merge: Some(true),
            review_mode: Some("codex_premerge".to_string()),
            merge_policy: Some("automerge_when_green".to_string()),
            human_gate: Some("on_blocker_only".to_string()),
            blocked_by: Vec::new(),
            acceptance: Some(TextList::One("Reject deprecated field.".to_string())),
            commands: vec!["cargo fmt --all -- --check".to_string()],
            allowed_paths: vec!["docs/**".to_string()],
            forbidden_paths: vec!["crates/**".to_string()],
            may_claim: vec!["Deprecated field is rejected.".to_string()],
            must_not_claim: vec!["Deprecated field is accepted.".to_string()],
        };
        let deprecated = policy_campaign(item.clone());
        let problems = validate_campaign(&deprecated);

        assert!(problems.iter().any(|problem| {
            problem.severity == Severity::Error
                && problem.message.contains("deprecated requires_human_merge")
        }));

        item.requires_human_merge = None;
        item.merge_policy = Some("ship_it".to_string());
        let invalid_policy = policy_campaign(item);
        let problems = validate_campaign(&invalid_policy);
        assert!(problems.iter().any(|problem| {
            problem.severity == Severity::Error
                && problem.message.contains("invalid merge_policy `ship_it`")
        }));
    }

    #[test]
    fn current_item_prefers_actionable_work_over_blocked_followup() {
        let manifest = campaign_manifest(vec![
            work_item("POLICY-001", "merged", &[]),
            work_item("POLICY-002", "ready", &["POLICY-001"]),
            work_item("POLICY-003", "blocked", &["POLICY-002"]),
        ]);

        assert_eq!(current_item(&manifest).map(|item| item.id.as_str()), Some("POLICY-002"));
    }

    #[test]
    fn current_item_prefers_runnable_proposed_work_over_blocked_followup() {
        let manifest = campaign_manifest(vec![
            work_item("POLICY-001", "merged", &[]),
            work_item("POLICY-002", "blocked", &[]),
            work_item("POLICY-003", "proposed", &["POLICY-001"]),
        ]);

        assert_eq!(current_item(&manifest).map(|item| item.id.as_str()), Some("POLICY-003"));
    }

    #[test]
    fn current_item_keeps_open_pr_and_in_progress_items_first() {
        let pr_open = campaign_manifest(vec![
            work_item("POLICY-001", "merged", &[]),
            work_item("POLICY-002", "pr_open", &["POLICY-001"]),
            work_item("POLICY-003", "ready", &["POLICY-001"]),
        ]);
        let in_progress = campaign_manifest(vec![
            work_item("POLICY-001", "merged", &[]),
            work_item("POLICY-002", "in_progress", &["POLICY-001"]),
            work_item("POLICY-003", "ready", &["POLICY-001"]),
        ]);

        assert_eq!(current_item(&pr_open).map(|item| item.id.as_str()), Some("POLICY-002"));
        assert_eq!(current_item(&in_progress).map(|item| item.id.as_str()), Some("POLICY-002"));
    }

    #[test]
    fn current_item_skips_unblocked_statuses_with_unmet_dependencies() {
        let manifest = campaign_manifest(vec![
            work_item("POLICY-001", "ready", &["POLICY-000"]),
            work_item("POLICY-002", "blocked", &["POLICY-001"]),
        ]);

        assert_eq!(current_item(&manifest).map(|item| item.id.as_str()), Some("POLICY-002"));
    }

    fn campaign_manifest(work_items: Vec<WorkItem>) -> CampaignManifest {
        CampaignManifest {
            id: "policy-campaign".to_string(),
            title: "Policy Campaign".to_string(),
            status: "active".to_string(),
            objective: "Validate campaign policy fields.".to_string(),
            end_state: vec!["Policy validation is covered.".to_string()],
            hard_constraints: vec!["Do not accept deprecated policy fields.".to_string()],
            work_items,
        }
    }

    fn work_item(id: &str, status: &str, blocked_by: &[&str]) -> WorkItem {
        WorkItem {
            id: id.to_string(),
            status: status.to_string(),
            branch: format!("codex/policy/{id}"),
            stackable: Some(false),
            requires_human_merge: None,
            review_mode: Some("codex_premerge".to_string()),
            merge_policy: Some("automerge_when_green".to_string()),
            human_gate: Some("on_blocker_only".to_string()),
            blocked_by: blocked_by.iter().map(|dep| dep.to_string()).collect(),
            acceptance: Some(TextList::One(format!("Complete {id}."))),
            commands: vec!["cargo fmt --all -- --check".to_string()],
            allowed_paths: vec!["docs/**".to_string()],
            forbidden_paths: vec!["crates/**".to_string()],
            may_claim: vec![format!("{id} is complete.")],
            must_not_claim: vec![format!("{id} has runtime impact.")],
        }
    }

    fn policy_campaign(item: WorkItem) -> LoadedCampaign {
        LoadedCampaign {
            dir: PathBuf::from("policy-campaign"),
            manifest: campaign_manifest(vec![item]),
            events: Vec::new(),
        }
    }
}
