//! GPU topology discovery, NUMA affinity mapping, and placement
//! optimization.
//!
//! Provides [`GpuTopology`] for representing device interconnect graphs,
//! [`BandwidthMatrix`] for peer-to-peer transfer estimation,
//! [`PlacementOptimizer`] for topology-aware tensor placement, and
//! [`TopologyVisualizer`] for ASCII-art debugging output.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fmt::Write as _;

// ── Link characterization ─────────────────────────────────────────────

/// Physical interconnect technology between two topology nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum LinkType {
    PCIeGen3,
    PCIeGen4,
    PCIeGen5,
    NVLink3,
    NVLink4,
    XeLink,
    InfinityFabric,
    CXL,
}

impl LinkType {
    /// Theoretical per-direction bandwidth in GB/s for a single link.
    #[must_use]
    pub const fn theoretical_bandwidth_gbps(&self) -> f64 {
        match self {
            Self::PCIeGen3 => 15.75,
            Self::PCIeGen4 => 31.5,
            Self::PCIeGen5 => 63.0,
            Self::NVLink3 => 50.0,
            Self::NVLink4 => 100.0,
            Self::XeLink => 53.0,
            Self::InfinityFabric => 36.0,
            Self::CXL => 64.0,
        }
    }

    /// Approximate one-way latency in nanoseconds.
    #[must_use]
    pub const fn typical_latency_ns(&self) -> u64 {
        match self {
            Self::PCIeGen3 | Self::PCIeGen4 | Self::PCIeGen5 => 700,
            Self::NVLink3 | Self::NVLink4 => 300,
            Self::XeLink => 400,
            Self::InfinityFabric => 500,
            Self::CXL => 250,
        }
    }
}

impl fmt::Display for LinkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::PCIeGen3 => "PCIe-Gen3",
            Self::PCIeGen4 => "PCIe-Gen4",
            Self::PCIeGen5 => "PCIe-Gen5",
            Self::NVLink3 => "NVLink3",
            Self::NVLink4 => "NVLink4",
            Self::XeLink => "XeLink",
            Self::InfinityFabric => "Infinity-Fabric",
            Self::CXL => "CXL",
        };
        f.write_str(name)
    }
}

// ── Device link ───────────────────────────────────────────────────────

/// A directed connection between two topology nodes.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct DeviceLink {
    pub source: String,
    pub target: String,
    pub link_type: LinkType,
    pub bandwidth_gbps: f64,
    pub latency_ns: u64,
    /// Number of physical links bonded together.
    pub lane_count: u32,
}

impl DeviceLink {
    /// Create a link using defaults from the `LinkType`.
    #[must_use]
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        link_type: LinkType,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            bandwidth_gbps: link_type.theoretical_bandwidth_gbps(),
            latency_ns: link_type.typical_latency_ns(),
            link_type,
            lane_count: 1,
        }
    }

    /// Builder: override bandwidth.
    #[must_use]
    pub const fn with_bandwidth(mut self, gbps: f64) -> Self {
        self.bandwidth_gbps = gbps;
        self
    }

    /// Builder: override latency.
    #[must_use]
    pub const fn with_latency(mut self, ns: u64) -> Self {
        self.latency_ns = ns;
        self
    }

    /// Builder: set lane count.
    #[must_use]
    pub const fn with_lanes(mut self, lanes: u32) -> Self {
        self.lane_count = lanes;
        self
    }

    /// Effective bandwidth = per-link × lane count.
    #[must_use]
    pub fn effective_bandwidth_gbps(&self) -> f64 {
        self.bandwidth_gbps * f64::from(self.lane_count)
    }
}

// ── Topology node ─────────────────────────────────────────────────────

/// Variant tag for a node in the topology graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum NodeKind {
    Gpu,
    CpuSocket,
    PcieSwitch,
    NvlinkBridge,
    XeLinkBridge,
}

impl fmt::Display for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tag = match self {
            Self::Gpu => "GPU",
            Self::CpuSocket => "CPU",
            Self::PcieSwitch => "PCIe-SW",
            Self::NvlinkBridge => "NVL-BR",
            Self::XeLinkBridge => "XeL-BR",
        };
        f.write_str(tag)
    }
}

/// A single device or interconnect component in the topology.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct TopologyNode {
    pub id: String,
    pub kind: NodeKind,
    /// Vendor-specific description (e.g. "NVIDIA A100 80GB").
    pub description: String,
    /// Total device memory in bytes (0 for bridges/switches).
    pub memory_bytes: u64,
    /// NUMA node affinity (`None` if unknown).
    pub numa_node: Option<u32>,
}

impl TopologyNode {
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        kind: NodeKind,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            description: description.into(),
            memory_bytes: 0,
            numa_node: None,
        }
    }

    /// Builder: set device memory.
    #[must_use]
    pub const fn with_memory(mut self, bytes: u64) -> Self {
        self.memory_bytes = bytes;
        self
    }

    /// Builder: set NUMA affinity.
    #[must_use]
    pub const fn with_numa(mut self, node: u32) -> Self {
        self.numa_node = Some(node);
        self
    }
}

// ── NUMA affinity ─────────────────────────────────────────────────────

/// Maps device IDs to their NUMA node assignments.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct NumaAffinity {
    device_to_numa: BTreeMap<String, u32>,
}

impl NumaAffinity {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a device→NUMA mapping.
    pub fn insert(&mut self, device_id: impl Into<String>, numa_node: u32) {
        self.device_to_numa.insert(device_id.into(), numa_node);
    }

    /// Look up the NUMA node for a device.
    #[must_use]
    pub fn get(&self, device_id: &str) -> Option<u32> {
        self.device_to_numa.get(device_id).copied()
    }

    /// All devices on a given NUMA node.
    #[must_use]
    pub fn devices_on_node(&self, numa_node: u32) -> Vec<&str> {
        self.device_to_numa
            .iter()
            .filter(|(_, n)| **n == numa_node)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// The set of all NUMA nodes that have at least one device.
    #[must_use]
    pub fn active_nodes(&self) -> Vec<u32> {
        let mut nodes: Vec<u32> =
            self.device_to_numa.values().copied().collect::<HashSet<_>>()
                .into_iter()
                .collect();
        nodes.sort_unstable();
        nodes
    }

    /// Number of device→NUMA entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.device_to_numa.len()
    }

    /// Whether the map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.device_to_numa.is_empty()
    }
}

// ── Bandwidth matrix ──────────────────────────────────────────────────

/// N×N matrix of peer-to-peer transfer rates (GB/s) between devices.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BandwidthMatrix {
    /// Ordered list of device IDs (row/column index).
    device_ids: Vec<String>,
    /// Row-major f64 storage.
    data: Vec<f64>,
}

impl BandwidthMatrix {
    /// Create a zero-initialized matrix for the given devices.
    #[must_use]
    pub fn new(device_ids: Vec<String>) -> Self {
        let n = device_ids.len();
        Self { device_ids, data: vec![0.0; n * n] }
    }

    /// Number of devices.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.device_ids.len()
    }

    /// Device IDs in index order.
    #[must_use]
    pub fn device_ids(&self) -> &[String] {
        &self.device_ids
    }

    fn index_of(&self, id: &str) -> Option<usize> {
        self.device_ids.iter().position(|d| d == id)
    }

    /// Set the bandwidth from `src` to `dst`.
    ///
    /// Returns `false` if either device ID is unknown.
    pub fn set(&mut self, src: &str, dst: &str, gbps: f64) -> bool {
        if let (Some(r), Some(c)) = (self.index_of(src), self.index_of(dst)) {
            let n = self.device_ids.len();
            self.data[r * n + c] = gbps;
            true
        } else {
            false
        }
    }

    /// Get the bandwidth from `src` to `dst`.
    #[must_use]
    pub fn get(&self, src: &str, dst: &str) -> Option<f64> {
        let r = self.index_of(src)?;
        let c = self.index_of(dst)?;
        let n = self.device_ids.len();
        Some(self.data[r * n + c])
    }

    /// Set bandwidth symmetrically for both directions.
    pub fn set_symmetric(
        &mut self,
        a: &str,
        b: &str,
        gbps: f64,
    ) -> bool {
        self.set(a, b, gbps) && self.set(b, a, gbps)
    }

    /// The minimum non-zero bandwidth across the entire matrix.
    #[must_use]
    pub fn min_bandwidth(&self) -> f64 {
        self.data
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .fold(f64::INFINITY, f64::min)
    }

    /// The maximum bandwidth in the matrix.
    #[must_use]
    pub fn max_bandwidth(&self) -> f64 {
        self.data.iter().copied().fold(0.0_f64, f64::max)
    }

    /// Raw row-major data slice.
    #[must_use]
    pub fn raw_data(&self) -> &[f64] {
        &self.data
    }
}

// ── Peer access table ─────────────────────────────────────────────────

/// Tracks which device-pairs support direct memory access.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct PeerAccessTable {
    pairs: HashSet<(String, String)>,
}

impl PeerAccessTable {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `src` can directly access `dst` memory.
    pub fn grant(&mut self, src: impl Into<String>, dst: impl Into<String>) {
        self.pairs.insert((src.into(), dst.into()));
    }

    /// Record bidirectional access between two devices.
    pub fn grant_bidirectional(
        &mut self,
        a: impl Into<String> + Clone,
        b: impl Into<String> + Clone,
    ) {
        let a_s: String = a.into();
        let b_s: String = b.into();
        self.pairs.insert((a_s.clone(), b_s.clone()));
        self.pairs.insert((b_s, a_s));
    }

    /// Can `src` directly access `dst` memory?
    #[must_use]
    pub fn can_access(&self, src: &str, dst: &str) -> bool {
        self.pairs.contains(&(src.to_owned(), dst.to_owned()))
    }

    /// All devices that `src` can reach via P2P.
    #[must_use]
    pub fn peers_of(&self, src: &str) -> Vec<&str> {
        self.pairs
            .iter()
            .filter(|(s, _)| s == src)
            .map(|(_, d)| d.as_str())
            .collect()
    }

    /// Total number of directional access grants.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

// ── GpuTopology ───────────────────────────────────────────────────────

/// Complete device topology graph.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuTopology {
    nodes: Vec<TopologyNode>,
    links: Vec<DeviceLink>,
    numa: NumaAffinity,
    peer_access: PeerAccessTable,
}

impl GpuTopology {
    /// Build a topology from discovered nodes and links.
    #[must_use]
    pub const fn new(
        nodes: Vec<TopologyNode>,
        links: Vec<DeviceLink>,
        numa: NumaAffinity,
        peer_access: PeerAccessTable,
    ) -> Self {
        Self { nodes, links, numa, peer_access }
    }

    #[must_use]
    pub fn nodes(&self) -> &[TopologyNode] {
        &self.nodes
    }

    #[must_use]
    pub fn links(&self) -> &[DeviceLink] {
        &self.links
    }

    #[must_use]
    pub const fn numa(&self) -> &NumaAffinity {
        &self.numa
    }

    #[must_use]
    pub const fn peer_access(&self) -> &PeerAccessTable {
        &self.peer_access
    }

    /// All GPU nodes in the topology.
    #[must_use]
    pub fn gpus(&self) -> Vec<&TopologyNode> {
        self.nodes.iter().filter(|n| n.kind == NodeKind::Gpu).collect()
    }

    /// Number of GPU devices.
    #[must_use]
    pub fn gpu_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.kind == NodeKind::Gpu).count()
    }

    /// Total device memory across all GPUs.
    #[must_use]
    pub fn total_gpu_memory(&self) -> u64 {
        self.nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Gpu)
            .map(|n| n.memory_bytes)
            .sum()
    }

    /// Compute the [`BandwidthMatrix`] for GPU nodes only.
    #[must_use]
    pub fn bandwidth_matrix(&self) -> BandwidthMatrix {
        let gpu_ids: Vec<String> =
            self.gpus().iter().map(|g| g.id.clone()).collect();
        let mut mat = BandwidthMatrix::new(gpu_ids);
        for link in &self.links {
            // Only include links between GPUs.
            let src_gpu = self
                .nodes
                .iter()
                .any(|n| n.id == link.source && n.kind == NodeKind::Gpu);
            let dst_gpu = self
                .nodes
                .iter()
                .any(|n| n.id == link.target && n.kind == NodeKind::Gpu);
            if src_gpu && dst_gpu {
                mat.set(
                    &link.source,
                    &link.target,
                    link.effective_bandwidth_gbps(),
                );
            }
        }
        mat
    }

    /// Look up a node by id.
    #[must_use]
    pub fn node(&self, id: &str) -> Option<&TopologyNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Links originating from a given node.
    #[must_use]
    pub fn links_from(&self, id: &str) -> Vec<&DeviceLink> {
        self.links.iter().filter(|l| l.source == id).collect()
    }
}

// ── TopologyDiscoverer trait ──────────────────────────────────────────

/// Trait for platform-specific topology discovery.
pub trait TopologyDiscoverer {
    /// Discover the system topology.
    ///
    /// # Errors
    /// Returns an error string when discovery fails (e.g. missing sysfs).
    fn discover(&self) -> Result<GpuTopology, String>;
}

/// Discovers topology from Linux `/sys/bus/pci` (stub – real impl needs
/// OS access).
pub struct SysfsDiscoverer {
    sysfs_root: String,
}

impl SysfsDiscoverer {
    #[must_use]
    pub fn new() -> Self {
        Self { sysfs_root: "/sys/bus/pci/devices".to_owned() }
    }

    /// Point at a custom sysfs path (useful for tests with a tmpdir).
    #[must_use]
    pub fn with_root(root: impl Into<String>) -> Self {
        Self { sysfs_root: root.into() }
    }

    /// The sysfs root being scanned.
    #[must_use]
    pub fn sysfs_root(&self) -> &str {
        &self.sysfs_root
    }
}

impl Default for SysfsDiscoverer {
    fn default() -> Self {
        Self::new()
    }
}

impl TopologyDiscoverer for SysfsDiscoverer {
    fn discover(&self) -> Result<GpuTopology, String> {
        // Stub: real implementation would parse sysfs. Return empty
        // topology so callers get a valid (but empty) result on
        // non-Linux or when sysfs is unavailable.
        Ok(GpuTopology::new(
            Vec::new(),
            Vec::new(),
            NumaAffinity::new(),
            PeerAccessTable::new(),
        ))
    }
}

/// Fully in-memory topology for deterministic testing.
pub struct MockDiscoverer {
    topology: GpuTopology,
}

impl MockDiscoverer {
    #[must_use]
    pub const fn new(topology: GpuTopology) -> Self {
        Self { topology }
    }

    /// Create a simple 2-GPU `NVLink` topology for quick tests.
    #[must_use]
    pub fn two_gpu_nvlink() -> Self {
        let nodes = vec![
            TopologyNode::new("gpu0", NodeKind::Gpu, "GPU 0")
                .with_memory(80 * 1024 * 1024 * 1024)
                .with_numa(0),
            TopologyNode::new("gpu1", NodeKind::Gpu, "GPU 1")
                .with_memory(80 * 1024 * 1024 * 1024)
                .with_numa(0),
            TopologyNode::new("cpu0", NodeKind::CpuSocket, "CPU 0")
                .with_numa(0),
        ];
        let links = vec![
            DeviceLink::new("gpu0", "gpu1", LinkType::NVLink4),
            DeviceLink::new("gpu1", "gpu0", LinkType::NVLink4),
            DeviceLink::new("cpu0", "gpu0", LinkType::PCIeGen4),
            DeviceLink::new("cpu0", "gpu1", LinkType::PCIeGen4),
        ];
        let mut numa = NumaAffinity::new();
        numa.insert("gpu0", 0);
        numa.insert("gpu1", 0);
        numa.insert("cpu0", 0);

        let mut peer = PeerAccessTable::new();
        peer.grant_bidirectional("gpu0", "gpu1");

        Self::new(GpuTopology::new(nodes, links, numa, peer))
    }

    /// Create a 4-GPU `PCIe` topology across two NUMA nodes.
    #[must_use]
    pub fn four_gpu_pcie() -> Self {
        let nodes = vec![
            TopologyNode::new("gpu0", NodeKind::Gpu, "GPU 0")
                .with_memory(24 * 1024 * 1024 * 1024)
                .with_numa(0),
            TopologyNode::new("gpu1", NodeKind::Gpu, "GPU 1")
                .with_memory(24 * 1024 * 1024 * 1024)
                .with_numa(0),
            TopologyNode::new("gpu2", NodeKind::Gpu, "GPU 2")
                .with_memory(24 * 1024 * 1024 * 1024)
                .with_numa(1),
            TopologyNode::new("gpu3", NodeKind::Gpu, "GPU 3")
                .with_memory(24 * 1024 * 1024 * 1024)
                .with_numa(1),
            TopologyNode::new("sw0", NodeKind::PcieSwitch, "PCIe SW 0"),
            TopologyNode::new("sw1", NodeKind::PcieSwitch, "PCIe SW 1"),
            TopologyNode::new("cpu0", NodeKind::CpuSocket, "CPU 0")
                .with_numa(0),
            TopologyNode::new("cpu1", NodeKind::CpuSocket, "CPU 1")
                .with_numa(1),
        ];

        let pcie4 = LinkType::PCIeGen4;
        let links = vec![
            DeviceLink::new("sw0", "gpu0", pcie4),
            DeviceLink::new("sw0", "gpu1", pcie4),
            DeviceLink::new("sw1", "gpu2", pcie4),
            DeviceLink::new("sw1", "gpu3", pcie4),
            DeviceLink::new("cpu0", "sw0", pcie4),
            DeviceLink::new("cpu1", "sw1", pcie4),
            // Direct GPU-GPU links through the switch
            DeviceLink::new("gpu0", "gpu1", pcie4),
            DeviceLink::new("gpu1", "gpu0", pcie4),
            DeviceLink::new("gpu2", "gpu3", pcie4),
            DeviceLink::new("gpu3", "gpu2", pcie4),
        ];

        let mut numa = NumaAffinity::new();
        for (id, node) in
            [("gpu0", 0), ("gpu1", 0), ("gpu2", 1), ("gpu3", 1)]
        {
            numa.insert(id, node);
        }

        let mut peer = PeerAccessTable::new();
        peer.grant_bidirectional("gpu0", "gpu1");
        peer.grant_bidirectional("gpu2", "gpu3");

        Self::new(GpuTopology::new(nodes, links, numa, peer))
    }
}

impl TopologyDiscoverer for MockDiscoverer {
    fn discover(&self) -> Result<GpuTopology, String> {
        Ok(self.topology.clone())
    }
}

// ── PlacementOptimizer ────────────────────────────────────────────────

/// Suggest optimal tensor placement based on topology.
pub struct PlacementOptimizer<'a> {
    topology: &'a GpuTopology,
}

/// A single tensor placement recommendation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementDecision {
    pub tensor_name: String,
    pub device_id: String,
    pub reason: String,
}

impl<'a> PlacementOptimizer<'a> {
    #[must_use]
    pub const fn new(topology: &'a GpuTopology) -> Self {
        Self { topology }
    }

    /// Place tensor on the GPU with the most free memory.
    ///
    /// `memory_usage` maps device ID → bytes currently consumed.
    #[must_use]
    pub fn place_by_memory(
        &self,
        tensor_name: &str,
        tensor_bytes: u64,
        memory_usage: &HashMap<String, u64>,
    ) -> Option<PlacementDecision> {
        self.topology
            .gpus()
            .iter()
            .filter_map(|g| {
                let used = memory_usage.get(&g.id).copied().unwrap_or(0);
                let free = g.memory_bytes.saturating_sub(used);
                if free >= tensor_bytes {
                    Some((&g.id, free))
                } else {
                    None
                }
            })
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(id, free)| PlacementDecision {
                tensor_name: tensor_name.to_owned(),
                device_id: id.clone(),
                reason: format!("{free} bytes free"),
            })
    }

    /// Place tensor near an existing tensor (locality-aware).
    ///
    /// Picks the peer with the highest bandwidth to `anchor_device`.
    #[must_use]
    pub fn place_near(
        &self,
        tensor_name: &str,
        anchor_device: &str,
    ) -> Option<PlacementDecision> {
        let bw = self.topology.bandwidth_matrix();
        let gpu_ids = bw.device_ids().to_vec();
        gpu_ids
            .iter()
            .filter(|id| id.as_str() != anchor_device)
            .filter_map(|id| {
                bw.get(anchor_device, id).map(|b| (id.clone(), b))
            })
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, b)| PlacementDecision {
                tensor_name: tensor_name.to_owned(),
                device_id: id,
                reason: format!("{b:.1} GB/s to {anchor_device}"),
            })
    }

    /// Place tensor on the same NUMA node as an anchor device.
    #[must_use]
    pub fn place_numa_local(
        &self,
        tensor_name: &str,
        anchor_device: &str,
        tensor_bytes: u64,
        memory_usage: &HashMap<String, u64>,
    ) -> Option<PlacementDecision> {
        let anchor_numa = self.topology.numa().get(anchor_device)?;
        self.topology
            .gpus()
            .into_iter()
            .filter(|g| g.numa_node == Some(anchor_numa))
            .filter(|g| g.id != anchor_device)
            .find_map(|g| {
                let used = memory_usage.get(&g.id).copied().unwrap_or(0);
                let free = g.memory_bytes.saturating_sub(used);
                if free >= tensor_bytes {
                    Some(PlacementDecision {
                        tensor_name: tensor_name.to_owned(),
                        device_id: g.id.clone(),
                        reason: format!(
                            "NUMA-local (node {anchor_numa}), {free} B free"
                        ),
                    })
                } else {
                    None
                }
            })
    }

    /// List GPU IDs grouped by NUMA node.
    #[must_use]
    pub fn gpus_by_numa(&self) -> BTreeMap<u32, Vec<String>> {
        let mut map: BTreeMap<u32, Vec<String>> = BTreeMap::new();
        for g in self.topology.gpus() {
            if let Some(n) = g.numa_node {
                map.entry(n).or_default().push(g.id.clone());
            }
        }
        map
    }
}

// ── TopologyVisualizer ────────────────────────────────────────────────

/// Renders an ASCII-art representation of the topology.
pub struct TopologyVisualizer<'a> {
    topology: &'a GpuTopology,
}

impl<'a> TopologyVisualizer<'a> {
    #[must_use]
    pub const fn new(topology: &'a GpuTopology) -> Self {
        Self { topology }
    }

    /// Render a compact ASCII tree grouped by NUMA node.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::from("=== GPU Topology ===\n");
        let nodes = self.topology.nodes();

        if nodes.is_empty() {
            out.push_str("  (empty topology)\n");
            return out;
        }

        // Group GPUs by NUMA
        let mut numa_groups: BTreeMap<u32, Vec<&TopologyNode>> =
            BTreeMap::new();
        let mut no_numa: Vec<&TopologyNode> = Vec::new();
        for n in nodes {
            if n.kind == NodeKind::Gpu {
                if let Some(nn) = n.numa_node {
                    numa_groups.entry(nn).or_default().push(n);
                } else {
                    no_numa.push(n);
                }
            }
        }

        for (numa, gpus) in &numa_groups {
            let _ = writeln!(out, "  NUMA node {numa}:");
            for g in gpus {
                let mem_gb = g.memory_bytes / (1024 * 1024 * 1024);
                let _ = writeln!(
                    out,
                    "    [{kind}] {id} — {desc} ({mem_gb} GB)",
                    kind = g.kind,
                    id = g.id,
                    desc = g.description,
                );
            }
        }

        if !no_numa.is_empty() {
            out.push_str("  NUMA unknown:\n");
            for g in &no_numa {
                let _ = writeln!(
                    out,
                    "    [{kind}] {id} — {desc}",
                    kind = g.kind,
                    id = g.id,
                    desc = g.description,
                );
            }
        }

        // Links summary
        let links = self.topology.links();
        if !links.is_empty() {
            out.push_str("  Links:\n");
            for l in links {
                let _ = writeln!(
                    out,
                    "    {src} -> {dst}  [{lt}] {bw:.1} GB/s",
                    src = l.source,
                    dst = l.target,
                    lt = l.link_type,
                    bw = l.effective_bandwidth_gbps(),
                );
            }
        }

        out
    }

    /// Render just the bandwidth matrix as a table.
    #[must_use]
    pub fn render_bandwidth_matrix(&self) -> String {
        let bw = self.topology.bandwidth_matrix();
        let ids = bw.device_ids();
        if ids.is_empty() {
            return String::from("(no GPUs)\n");
        }

        let col_w = 10;
        let mut out = format!("{:>col_w$}", "");
        for id in ids {
            let _ = write!(out, " {id:>col_w$}");
        }
        out.push('\n');

        for src in ids {
            let _ = write!(out, "{src:>col_w$}");
            for dst in ids {
                let val = bw.get(src, dst).unwrap_or(0.0);
                let _ = write!(out, " {val:>col_w$.1}");
            }
            out.push('\n');
        }
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── LinkType ──────────────────────────────────────────────────────

    #[test]
    fn link_type_bandwidth_positive() {
        let types = [
            LinkType::PCIeGen3,
            LinkType::PCIeGen4,
            LinkType::PCIeGen5,
            LinkType::NVLink3,
            LinkType::NVLink4,
            LinkType::XeLink,
            LinkType::InfinityFabric,
            LinkType::CXL,
        ];
        for lt in types {
            assert!(lt.theoretical_bandwidth_gbps() > 0.0, "{lt:?}");
        }
    }

    #[test]
    fn link_type_latency_positive() {
        let types = [
            LinkType::PCIeGen3,
            LinkType::PCIeGen4,
            LinkType::PCIeGen5,
            LinkType::NVLink3,
            LinkType::NVLink4,
            LinkType::XeLink,
            LinkType::InfinityFabric,
            LinkType::CXL,
        ];
        for lt in types {
            assert!(lt.typical_latency_ns() > 0, "{lt:?}");
        }
    }

    #[test]
    fn link_type_display() {
        assert_eq!(LinkType::PCIeGen4.to_string(), "PCIe-Gen4");
        assert_eq!(LinkType::NVLink4.to_string(), "NVLink4");
        assert_eq!(LinkType::XeLink.to_string(), "XeLink");
        assert_eq!(LinkType::InfinityFabric.to_string(), "Infinity-Fabric");
        assert_eq!(LinkType::CXL.to_string(), "CXL");
    }

    #[test]
    fn pcie_gen_bandwidth_ordering() {
        let g3 = LinkType::PCIeGen3.theoretical_bandwidth_gbps();
        let g4 = LinkType::PCIeGen4.theoretical_bandwidth_gbps();
        let g5 = LinkType::PCIeGen5.theoretical_bandwidth_gbps();
        assert!(g3 < g4);
        assert!(g4 < g5);
    }

    #[test]
    fn nvlink_bandwidth_ordering() {
        let nv3 = LinkType::NVLink3.theoretical_bandwidth_gbps();
        let nv4 = LinkType::NVLink4.theoretical_bandwidth_gbps();
        assert!(nv3 < nv4);
    }

    #[test]
    fn nvlink_latency_lower_than_pcie() {
        assert!(
            LinkType::NVLink4.typical_latency_ns()
                < LinkType::PCIeGen4.typical_latency_ns()
        );
    }

    // ── DeviceLink ───────────────────────────────────────────────────

    #[test]
    fn device_link_defaults() {
        let l = DeviceLink::new("a", "b", LinkType::PCIeGen4);
        assert_eq!(l.source, "a");
        assert_eq!(l.target, "b");
        assert_eq!(l.link_type, LinkType::PCIeGen4);
        assert_eq!(l.lane_count, 1);
        assert!(
            (l.bandwidth_gbps
                - LinkType::PCIeGen4.theoretical_bandwidth_gbps())
            .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn device_link_with_bandwidth() {
        let l = DeviceLink::new("a", "b", LinkType::NVLink4)
            .with_bandwidth(90.0);
        assert!((l.bandwidth_gbps - 90.0).abs() < f64::EPSILON);
    }

    #[test]
    fn device_link_with_latency() {
        let l = DeviceLink::new("a", "b", LinkType::NVLink4)
            .with_latency(200);
        assert_eq!(l.latency_ns, 200);
    }

    #[test]
    fn device_link_with_lanes() {
        let l = DeviceLink::new("a", "b", LinkType::NVLink4)
            .with_lanes(4);
        assert_eq!(l.lane_count, 4);
    }

    #[test]
    fn device_link_effective_bandwidth() {
        let l = DeviceLink::new("a", "b", LinkType::NVLink4)
            .with_lanes(4);
        let expected = LinkType::NVLink4.theoretical_bandwidth_gbps() * 4.0;
        assert!((l.effective_bandwidth_gbps() - expected).abs() < 0.001);
    }

    #[test]
    fn device_link_builder_chaining() {
        let l = DeviceLink::new("x", "y", LinkType::XeLink)
            .with_bandwidth(50.0)
            .with_latency(350)
            .with_lanes(2);
        assert!((l.bandwidth_gbps - 50.0).abs() < f64::EPSILON);
        assert_eq!(l.latency_ns, 350);
        assert_eq!(l.lane_count, 2);
        assert!((l.effective_bandwidth_gbps() - 100.0).abs() < 0.001);
    }

    // ── NodeKind ─────────────────────────────────────────────────────

    #[test]
    fn node_kind_display() {
        assert_eq!(NodeKind::Gpu.to_string(), "GPU");
        assert_eq!(NodeKind::CpuSocket.to_string(), "CPU");
        assert_eq!(NodeKind::PcieSwitch.to_string(), "PCIe-SW");
        assert_eq!(NodeKind::NvlinkBridge.to_string(), "NVL-BR");
        assert_eq!(NodeKind::XeLinkBridge.to_string(), "XeL-BR");
    }

    // ── TopologyNode ─────────────────────────────────────────────────

    #[test]
    fn topology_node_defaults() {
        let n = TopologyNode::new("gpu0", NodeKind::Gpu, "Test GPU");
        assert_eq!(n.id, "gpu0");
        assert_eq!(n.kind, NodeKind::Gpu);
        assert_eq!(n.memory_bytes, 0);
        assert_eq!(n.numa_node, None);
    }

    #[test]
    fn topology_node_with_memory() {
        let n = TopologyNode::new("gpu0", NodeKind::Gpu, "A100")
            .with_memory(80 * 1024 * 1024 * 1024);
        assert_eq!(n.memory_bytes, 80 * 1024 * 1024 * 1024);
    }

    #[test]
    fn topology_node_with_numa() {
        let n = TopologyNode::new("gpu0", NodeKind::Gpu, "A100")
            .with_numa(1);
        assert_eq!(n.numa_node, Some(1));
    }

    #[test]
    fn topology_node_builder_chaining() {
        let n = TopologyNode::new("gpu0", NodeKind::Gpu, "A100")
            .with_memory(40 * 1024 * 1024 * 1024)
            .with_numa(0);
        assert_eq!(n.memory_bytes, 40 * 1024 * 1024 * 1024);
        assert_eq!(n.numa_node, Some(0));
    }

    // ── NumaAffinity ─────────────────────────────────────────────────

    #[test]
    fn numa_empty() {
        let numa = NumaAffinity::new();
        assert!(numa.is_empty());
        assert_eq!(numa.len(), 0);
        assert_eq!(numa.get("gpu0"), None);
    }

    #[test]
    fn numa_insert_and_get() {
        let mut numa = NumaAffinity::new();
        numa.insert("gpu0", 0);
        numa.insert("gpu1", 1);
        assert_eq!(numa.get("gpu0"), Some(0));
        assert_eq!(numa.get("gpu1"), Some(1));
        assert_eq!(numa.len(), 2);
    }

    #[test]
    fn numa_devices_on_node() {
        let mut numa = NumaAffinity::new();
        numa.insert("gpu0", 0);
        numa.insert("gpu1", 0);
        numa.insert("gpu2", 1);
        let on_0 = numa.devices_on_node(0);
        assert_eq!(on_0.len(), 2);
        assert!(on_0.contains(&"gpu0"));
        assert!(on_0.contains(&"gpu1"));
        assert_eq!(numa.devices_on_node(1), vec!["gpu2"]);
    }

    #[test]
    fn numa_active_nodes() {
        let mut numa = NumaAffinity::new();
        numa.insert("a", 2);
        numa.insert("b", 0);
        numa.insert("c", 2);
        assert_eq!(numa.active_nodes(), vec![0, 2]);
    }

    #[test]
    fn numa_devices_on_nonexistent_node() {
        let numa = NumaAffinity::new();
        assert!(numa.devices_on_node(99).is_empty());
    }

    // ── BandwidthMatrix ──────────────────────────────────────────────

    #[test]
    fn bandwidth_matrix_empty() {
        let bw = BandwidthMatrix::new(Vec::new());
        assert_eq!(bw.size(), 0);
        assert!(bw.device_ids().is_empty());
    }

    #[test]
    fn bandwidth_matrix_set_get() {
        let ids = vec!["g0".into(), "g1".into()];
        let mut bw = BandwidthMatrix::new(ids);
        assert!(bw.set("g0", "g1", 50.0));
        assert_eq!(bw.get("g0", "g1"), Some(50.0));
        assert_eq!(bw.get("g1", "g0"), Some(0.0));
    }

    #[test]
    fn bandwidth_matrix_unknown_device() {
        let mut bw = BandwidthMatrix::new(vec!["g0".into()]);
        assert!(!bw.set("g0", "g_missing", 10.0));
        assert_eq!(bw.get("g0", "g_missing"), None);
    }

    #[test]
    fn bandwidth_matrix_symmetric() {
        let ids = vec!["g0".into(), "g1".into()];
        let mut bw = BandwidthMatrix::new(ids);
        assert!(bw.set_symmetric("g0", "g1", 100.0));
        assert_eq!(bw.get("g0", "g1"), Some(100.0));
        assert_eq!(bw.get("g1", "g0"), Some(100.0));
    }

    #[test]
    fn bandwidth_matrix_min_max() {
        let ids = vec!["a".into(), "b".into(), "c".into()];
        let mut bw = BandwidthMatrix::new(ids);
        bw.set("a", "b", 10.0);
        bw.set("b", "c", 50.0);
        assert!((bw.min_bandwidth() - 10.0).abs() < f64::EPSILON);
        assert!((bw.max_bandwidth() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bandwidth_matrix_raw_data_length() {
        let ids = vec!["a".into(), "b".into(), "c".into()];
        let bw = BandwidthMatrix::new(ids);
        assert_eq!(bw.raw_data().len(), 9);
    }

    #[test]
    fn bandwidth_matrix_single_device() {
        let mut bw = BandwidthMatrix::new(vec!["g0".into()]);
        bw.set("g0", "g0", 900.0);
        assert_eq!(bw.get("g0", "g0"), Some(900.0));
    }

    // ── PeerAccessTable ──────────────────────────────────────────────

    #[test]
    fn peer_access_empty() {
        let p = PeerAccessTable::new();
        assert!(p.is_empty());
        assert!(!p.can_access("a", "b"));
    }

    #[test]
    fn peer_access_grant() {
        let mut p = PeerAccessTable::new();
        p.grant("a", "b");
        assert!(p.can_access("a", "b"));
        assert!(!p.can_access("b", "a"));
    }

    #[test]
    fn peer_access_bidirectional() {
        let mut p = PeerAccessTable::new();
        p.grant_bidirectional("a", "b");
        assert!(p.can_access("a", "b"));
        assert!(p.can_access("b", "a"));
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn peer_access_peers_of() {
        let mut p = PeerAccessTable::new();
        p.grant("x", "y");
        p.grant("x", "z");
        let peers = p.peers_of("x");
        assert_eq!(peers.len(), 2);
        assert!(peers.contains(&"y"));
        assert!(peers.contains(&"z"));
        assert!(p.peers_of("y").is_empty());
    }

    #[test]
    fn peer_access_no_self_access_unless_granted() {
        let p = PeerAccessTable::new();
        assert!(!p.can_access("a", "a"));
    }

    // ── GpuTopology ──────────────────────────────────────────────────

    #[test]
    fn topology_empty() {
        let t = GpuTopology::new(
            Vec::new(),
            Vec::new(),
            NumaAffinity::new(),
            PeerAccessTable::new(),
        );
        assert_eq!(t.gpu_count(), 0);
        assert_eq!(t.total_gpu_memory(), 0);
        assert!(t.gpus().is_empty());
    }

    #[test]
    fn topology_gpu_count() {
        let mock = MockDiscoverer::two_gpu_nvlink();
        let topo = mock.discover().unwrap();
        assert_eq!(topo.gpu_count(), 2);
    }

    #[test]
    fn topology_total_memory() {
        let mock = MockDiscoverer::two_gpu_nvlink();
        let topo = mock.discover().unwrap();
        assert_eq!(
            topo.total_gpu_memory(),
            2 * 80 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn topology_node_lookup() {
        let mock = MockDiscoverer::two_gpu_nvlink();
        let topo = mock.discover().unwrap();
        let n = topo.node("gpu0").unwrap();
        assert_eq!(n.kind, NodeKind::Gpu);
        assert!(topo.node("nonexistent").is_none());
    }

    #[test]
    fn topology_links_from() {
        let mock = MockDiscoverer::two_gpu_nvlink();
        let topo = mock.discover().unwrap();
        let links = topo.links_from("gpu0");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "gpu1");
    }

    #[test]
    fn topology_bandwidth_matrix() {
        let mock = MockDiscoverer::two_gpu_nvlink();
        let topo = mock.discover().unwrap();
        let bw = topo.bandwidth_matrix();
        assert_eq!(bw.size(), 2);
        let g0g1 = bw.get("gpu0", "gpu1").unwrap();
        assert!(g0g1 > 0.0);
    }

    #[test]
    fn topology_four_gpu_count() {
        let mock = MockDiscoverer::four_gpu_pcie();
        let topo = mock.discover().unwrap();
        assert_eq!(topo.gpu_count(), 4);
    }

    #[test]
    fn topology_four_gpu_numa() {
        let mock = MockDiscoverer::four_gpu_pcie();
        let topo = mock.discover().unwrap();
        assert_eq!(topo.numa().get("gpu0"), Some(0));
        assert_eq!(topo.numa().get("gpu2"), Some(1));
    }

    #[test]
    fn topology_four_gpu_peer_access() {
        let mock = MockDiscoverer::four_gpu_pcie();
        let topo = mock.discover().unwrap();
        assert!(topo.peer_access().can_access("gpu0", "gpu1"));
        assert!(!topo.peer_access().can_access("gpu0", "gpu2"));
    }

    #[test]
    fn topology_four_gpu_bandwidth_matrix_cross_numa() {
        let mock = MockDiscoverer::four_gpu_pcie();
        let topo = mock.discover().unwrap();
        let bw = topo.bandwidth_matrix();
        // gpu0→gpu2 has no direct link
        assert_eq!(bw.get("gpu0", "gpu2"), Some(0.0));
        // gpu0→gpu1 has PCIe Gen4
        let g01 = bw.get("gpu0", "gpu1").unwrap();
        assert!(g01 > 0.0);
    }

    // ── TopologyDiscoverer ───────────────────────────────────────────

    #[test]
    fn mock_discoverer_returns_ok() {
        let mock = MockDiscoverer::two_gpu_nvlink();
        assert!(mock.discover().is_ok());
    }

    #[test]
    fn sysfs_discoverer_returns_empty() {
        let disc = SysfsDiscoverer::new();
        let topo = disc.discover().unwrap();
        assert_eq!(topo.gpu_count(), 0);
    }

    #[test]
    fn sysfs_discoverer_custom_root() {
        let disc = SysfsDiscoverer::with_root("/tmp/fake_sysfs");
        assert_eq!(disc.sysfs_root(), "/tmp/fake_sysfs");
    }

    #[test]
    fn sysfs_discoverer_default() {
        let disc = SysfsDiscoverer::default();
        assert_eq!(disc.sysfs_root(), "/sys/bus/pci/devices");
    }

    // ── PlacementOptimizer ───────────────────────────────────────────

    #[test]
    fn placement_by_memory_picks_emptiest() {
        let mock = MockDiscoverer::two_gpu_nvlink();
        let topo = mock.discover().unwrap();
        let opt = PlacementOptimizer::new(&topo);
        let mut usage = HashMap::new();
        usage.insert("gpu0".into(), 70 * 1024 * 1024 * 1024_u64);
        usage.insert("gpu1".into(), 10 * 1024 * 1024 * 1024_u64);

        let p = opt.place_by_memory("weights", 1024, &usage).unwrap();
        assert_eq!(p.device_id, "gpu1");
    }

    #[test]
    fn placement_by_memory_none_when_full() {
        let mock = MockDiscoverer::two_gpu_nvlink();
        let topo = mock.discover().unwrap();
        let opt = PlacementOptimizer::new(&topo);
        let mut usage = HashMap::new();
        let total = 80 * 1024 * 1024 * 1024_u64;
        usage.insert("gpu0".into(), total);
        usage.insert("gpu1".into(), total);

        assert!(opt.place_by_memory("big", 1, &usage).is_none());
    }

    #[test]
    fn placement_near_picks_best_link() {
        let mock = MockDiscoverer::two_gpu_nvlink();
        let topo = mock.discover().unwrap();
        let opt = PlacementOptimizer::new(&topo);
        let p = opt.place_near("kv_cache", "gpu0").unwrap();
        assert_eq!(p.device_id, "gpu1");
    }

    #[test]
    fn placement_numa_local() {
        let mock = MockDiscoverer::four_gpu_pcie();
        let topo = mock.discover().unwrap();
        let opt = PlacementOptimizer::new(&topo);
        let usage = HashMap::new();
        let p = opt
            .place_numa_local("attn", "gpu0", 1024, &usage)
            .unwrap();
        assert_eq!(p.device_id, "gpu1");
    }

    #[test]
    fn placement_numa_local_cross_node() {
        let mock = MockDiscoverer::four_gpu_pcie();
        let topo = mock.discover().unwrap();
        let opt = PlacementOptimizer::new(&topo);
        let usage = HashMap::new();
        let p = opt
            .place_numa_local("attn", "gpu2", 1024, &usage)
            .unwrap();
        assert_eq!(p.device_id, "gpu3");
    }

    #[test]
    fn gpus_by_numa_grouping() {
        let mock = MockDiscoverer::four_gpu_pcie();
        let topo = mock.discover().unwrap();
        let opt = PlacementOptimizer::new(&topo);
        let groups = opt.gpus_by_numa();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[&0].len(), 2);
        assert_eq!(groups[&1].len(), 2);
    }

    // ── TopologyVisualizer ───────────────────────────────────────────

    #[test]
    fn visualizer_empty_topology() {
        let t = GpuTopology::new(
            Vec::new(),
            Vec::new(),
            NumaAffinity::new(),
            PeerAccessTable::new(),
        );
        let v = TopologyVisualizer::new(&t);
        let out = v.render();
        assert!(out.contains("empty topology"));
    }

    #[test]
    fn visualizer_two_gpu() {
        let mock = MockDiscoverer::two_gpu_nvlink();
        let topo = mock.discover().unwrap();
        let v = TopologyVisualizer::new(&topo);
        let out = v.render();
        assert!(out.contains("GPU Topology"));
        assert!(out.contains("gpu0"));
        assert!(out.contains("gpu1"));
        assert!(out.contains("NUMA node 0"));
    }

    #[test]
    fn visualizer_shows_links() {
        let mock = MockDiscoverer::two_gpu_nvlink();
        let topo = mock.discover().unwrap();
        let v = TopologyVisualizer::new(&topo);
        let out = v.render();
        assert!(out.contains("Links:"));
        assert!(out.contains("NVLink4"));
    }

    #[test]
    fn visualizer_four_gpu_two_numa() {
        let mock = MockDiscoverer::four_gpu_pcie();
        let topo = mock.discover().unwrap();
        let v = TopologyVisualizer::new(&topo);
        let out = v.render();
        assert!(out.contains("NUMA node 0"));
        assert!(out.contains("NUMA node 1"));
    }

    #[test]
    fn visualizer_bandwidth_matrix_output() {
        let mock = MockDiscoverer::two_gpu_nvlink();
        let topo = mock.discover().unwrap();
        let v = TopologyVisualizer::new(&topo);
        let out = v.render_bandwidth_matrix();
        assert!(out.contains("gpu0"));
        assert!(out.contains("gpu1"));
    }

    #[test]
    fn visualizer_bandwidth_matrix_empty() {
        let t = GpuTopology::new(
            Vec::new(),
            Vec::new(),
            NumaAffinity::new(),
            PeerAccessTable::new(),
        );
        let v = TopologyVisualizer::new(&t);
        let out = v.render_bandwidth_matrix();
        assert!(out.contains("no GPUs"));
    }

    // ── Serialization smoke ──────────────────────────────────────────

    #[test]
    fn link_type_serializes() {
        let json = serde_json::to_string(&LinkType::NVLink4).unwrap();
        assert!(json.contains("NVLink4"));
    }

    #[test]
    fn topology_serializes() {
        let mock = MockDiscoverer::two_gpu_nvlink();
        let topo = mock.discover().unwrap();
        let json = serde_json::to_string(&topo).unwrap();
        assert!(json.contains("gpu0"));
        assert!(json.contains("NVLink4"));
    }

    #[test]
    fn bandwidth_matrix_serializes() {
        let ids = vec!["a".into(), "b".into()];
        let bw = BandwidthMatrix::new(ids);
        let json = serde_json::to_string(&bw).unwrap();
        assert!(json.contains("device_ids"));
    }
}
