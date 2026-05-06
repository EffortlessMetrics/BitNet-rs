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
const EVENT_TYPES: &[&str] =
    &["in_progress", "pr_open", "blocked", "superseded", "merged", "closeout"];
const REQUIRED_CAMPAIGNS: &[&str] = &[
    "apple-m4",
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
    let item_by_id = item_map(&campaign.manifest);
    let next = campaign
        .manifest
        .work_items
        .iter()
        .find(|item| item.status == "ready" && deps_met(item, &item_by_id))
        .or_else(|| {
            campaign
                .manifest
                .work_items
                .iter()
                .find(|item| item.status == "proposed" && deps_met(item, &item_by_id))
        });

    match next {
        Some(item) => {
            println!("campaign: {}", campaign.manifest.id);
            println!("next_item: {}", item.id);
            println!("status: {}", item.status);
            println!("branch: {}", item.branch);
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
    let mut writes = BTreeMap::new();

    for campaign in &campaigns {
        let rel = format!("{CAMPAIGNS_DIR}/{}/generated/status.md", campaign.manifest.id);
        writes.insert(root.join(&rel), render_campaign_status(campaign));
    }

    writes.insert(
        root.join(format!("{GENERATED_DIR}/global-dashboard.md")),
        render_global_dashboard(&campaigns),
    );
    writes
        .insert(root.join(format!("{GENERATED_DIR}/active-prs.md")), render_active_prs(&campaigns));
    writes.insert(
        root.join(format!("{GENERATED_DIR}/lane-dashboard.md")),
        render_lane_dashboard(&campaigns),
    );
    writes.insert(
        root.join(format!("{GENERATED_DIR}/blocked-items.md")),
        render_blocked_items(&campaigns),
    );

    let stale: Vec<_> = writes
        .iter()
        .filter_map(|(path, content)| {
            let current = fs::read_to_string(path).ok();
            if current.as_deref() == Some(content.as_str()) { None } else { Some(path.clone()) }
        })
        .collect();

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

    if generated_is_stale(root)? {
        problems.push(Problem::warning(
            "generated dashboards are stale; run `cargo run -p xtask --no-default-features -- campaign generate`",
        ));
    }

    for path in changed_legacy_tracker_files(root)? {
        problems.push(Problem::warning(format!(
            "legacy tracker changed in this branch: {}; normal item PRs should use campaign files",
            path.display()
        )));
    }

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
        if item.requires_human_merge.is_none() {
            problems.push(Problem::warning(format!(
                "item `{}` does not set requires_human_merge",
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
        if event.event == "pr_open" && event.pr.is_none() {
            problems.push(Problem::warning(format!(
                "pr_open event for `{}` is missing pr",
                event.item
            )));
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
    }

    problems
}

fn item_map(manifest: &CampaignManifest) -> BTreeMap<&str, &WorkItem> {
    manifest.work_items.iter().map(|item| (item.id.as_str(), item)).collect()
}

fn deps_met<'a>(item: &WorkItem, item_by_id: &BTreeMap<&'a str, &'a WorkItem>) -> bool {
    item.blocked_by
        .iter()
        .all(|dep| item_by_id.get(dep.as_str()).is_some_and(|dep_item| dep_item.status == "merged"))
}

fn current_item(manifest: &CampaignManifest) -> Option<&WorkItem> {
    for status in ["pr_open", "in_progress", "blocked", "ready", "proposed"] {
        if let Some(item) = manifest.work_items.iter().find(|item| item.status == status) {
            return Some(item);
        }
    }
    manifest.work_items.iter().find(|item| item.status == "merged")
}

fn latest_pr(events: &[Event], item_id: &str) -> Option<u64> {
    events.iter().filter(|event| event.item == item_id).filter_map(|event| event.pr).last()
}

fn render_campaign_status(campaign: &LoadedCampaign) -> String {
    let mut out = String::new();
    out.push_str(GENERATED_HEADER);
    out.push_str(&format!("# {} Campaign Status\n\n", campaign.manifest.title));
    out.push_str(&format!("- Campaign: `{}`\n", campaign.manifest.id));
    out.push_str(&format!("- State: `{}`\n", campaign.manifest.status));
    out.push_str(&format!("- Objective: {}\n\n", campaign.manifest.objective));
    out.push_str("## Work Items\n\n");
    out.push_str("| Item | State | PR | Branch | Acceptance |\n");
    out.push_str("|---|---|---:|---|---|\n");
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
            "| {} | {} | {} | `{}` | {} |\n",
            item.id, item.status, pr, item.branch, acceptance
        ));
    }
    out.push('\n');
    out.push_str("## Hard Constraints\n\n");
    for constraint in &campaign.manifest.hard_constraints {
        out.push_str(&format!("- {constraint}\n"));
    }
    out
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

fn generated_is_stale(root: &Path) -> Result<bool> {
    let campaigns = load_all_campaigns(root)?;
    let expected = root.join(format!("{GENERATED_DIR}/global-dashboard.md"));
    let rendered = render_global_dashboard(&campaigns);
    Ok(fs::read_to_string(expected).ok().as_deref() != Some(rendered.as_str()))
}

fn changed_legacy_tracker_files(root: &Path) -> Result<Vec<PathBuf>> {
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
