//! Connection pooling for multi-client scenarios

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};

use super::CachingConfig;

/// Connection information
#[derive(Debug, Clone)]
pub struct Connection {
    /// Connection identifier
    pub id: String,
    /// Client IP address
    pub client_ip: String,
    /// Connection established time
    pub established_at: Instant,
    /// Last activity time
    pub last_activity: Instant,
    /// Number of requests processed
    pub request_count: u64,
    /// Connection state
    pub state: ConnectionState,
}

/// Connection state
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    Active,
    Idle,
    Closing,
    Closed,
}

/// Connection pool manager
pub struct ConnectionPool {
    config: CachingConfig,
    connections: Arc<RwLock<HashMap<String, Connection>>>,
    connection_semaphore: Arc<Semaphore>,
    statistics: Arc<RwLock<ConnectionStatistics>>,
}

/// Connection pool statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConnectionStatistics {
    pub total_connections: usize,
    pub active_connections: usize,
    pub idle_connections: usize,
    pub peak_connections: usize,
    pub total_requests: u64,
    pub average_requests_per_connection: f64,
    pub connection_utilization: f64,
    pub average_connection_duration_seconds: f64,
    pub connection_timeouts: u64,
}

impl Default for ConnectionStatistics {
    fn default() -> Self {
        Self {
            total_connections: 0,
            active_connections: 0,
            idle_connections: 0,
            peak_connections: 0,
            total_requests: 0,
            average_requests_per_connection: 0.0,
            connection_utilization: 0.0,
            average_connection_duration_seconds: 0.0,
            connection_timeouts: 0,
        }
    }
}

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new(config: &CachingConfig) -> Result<Self> {
        let connection_semaphore = Arc::new(Semaphore::new(config.connection_pool_size));

        Ok(Self {
            config: config.clone(),
            connections: Arc::new(RwLock::new(HashMap::new())),
            connection_semaphore,
            statistics: Arc::new(RwLock::new(ConnectionStatistics::default())),
        })
    }

    /// Acquire a connection from the pool
    pub async fn acquire_connection(&self, client_ip: String) -> Result<Option<String>> {
        // Try to acquire a permit from the semaphore
        let permit = self.connection_semaphore.try_acquire();

        match permit {
            Ok(_permit) => {
                // Create a new connection
                let connection_id = uuid::Uuid::new_v4().to_string();
                let connection = Connection {
                    id: connection_id.clone(),
                    client_ip,
                    established_at: Instant::now(),
                    last_activity: Instant::now(),
                    request_count: 0,
                    state: ConnectionState::Active,
                };

                // Add to connections map
                {
                    let mut connections = self.connections.write().await;
                    connections.insert(connection_id.clone(), connection);
                }

                // Update statistics
                {
                    let mut stats = self.statistics.write().await;
                    stats.total_connections += 1;
                    stats.active_connections += 1;
                    stats.peak_connections = stats.peak_connections.max(stats.active_connections);
                    stats.connection_utilization =
                        stats.active_connections as f64 / self.config.connection_pool_size as f64;
                }

                // Don't drop the permit - it will be released when connection is closed
                std::mem::forget(_permit);

                Ok(Some(connection_id))
            }
            Err(_) => {
                // Pool is full
                Ok(None)
            }
        }
    }

    /// Release a connection back to the pool
    pub async fn release_connection(&self, connection_id: &str) -> Result<()> {
        let connection = {
            let mut connections = self.connections.write().await;
            connections.remove(connection_id)
        };

        if let Some(mut connection) = connection {
            connection.state = ConnectionState::Closed;
            let duration = connection.established_at.elapsed();

            // Update statistics
            {
                let mut stats = self.statistics.write().await;
                stats.active_connections = stats.active_connections.saturating_sub(1);
                stats.connection_utilization =
                    stats.active_connections as f64 / self.config.connection_pool_size as f64;

                // Update average connection duration
                let total_duration = stats.average_connection_duration_seconds
                    * (stats.total_connections - 1) as f64;
                stats.average_connection_duration_seconds =
                    (total_duration + duration.as_secs_f64()) / stats.total_connections as f64;

                // Update average requests per connection
                if stats.total_connections > 0 {
                    stats.average_requests_per_connection =
                        stats.total_requests as f64 / stats.total_connections as f64;
                }
            }

            // Release the semaphore permit
            self.connection_semaphore.add_permits(1);
        }

        Ok(())
    }

    /// Update connection activity
    pub async fn update_connection_activity(&self, connection_id: &str) -> Result<()> {
        let mut connections = self.connections.write().await;

        if let Some(connection) = connections.get_mut(connection_id) {
            connection.last_activity = Instant::now();
            connection.request_count += 1;

            // Update statistics
            {
                let mut stats = self.statistics.write().await;
                stats.total_requests += 1;
                stats.average_requests_per_connection =
                    stats.total_requests as f64 / stats.total_connections as f64;
            }
        }

        Ok(())
    }

    /// Get connection information
    pub async fn get_connection(&self, connection_id: &str) -> Option<Connection> {
        let connections = self.connections.read().await;
        connections.get(connection_id).cloned()
    }

    /// List all active connections
    pub async fn list_connections(&self) -> Vec<Connection> {
        let connections = self.connections.read().await;
        connections.values().filter(|conn| conn.state == ConnectionState::Active).cloned().collect()
    }

    /// Start connection cleanup task
    pub async fn start_cleanup_task(&self) {
        let connections = self.connections.clone();
        let statistics = self.statistics.clone();
        let semaphore = self.connection_semaphore.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // Check every minute

            loop {
                interval.tick().await;

                let now = Instant::now();
                let timeout_duration = Duration::from_secs(300); // 5 minute timeout
                let mut timed_out_connections = Vec::new();

                // Find timed out connections
                {
                    let connections_read = connections.read().await;
                    for (id, connection) in connections_read.iter() {
                        if connection.state == ConnectionState::Active
                            && now.duration_since(connection.last_activity) > timeout_duration
                        {
                            timed_out_connections.push(id.clone());
                        }
                    }
                }

                // Remove timed out connections
                for connection_id in timed_out_connections {
                    let connection = {
                        let mut connections_write = connections.write().await;
                        connections_write.remove(&connection_id)
                    };

                    if connection.is_some() {
                        // Update statistics
                        {
                            let mut stats = statistics.write().await;
                            stats.active_connections = stats.active_connections.saturating_sub(1);
                            stats.connection_timeouts += 1;
                            stats.connection_utilization = stats.active_connections as f64 / 100.0; // Assuming pool size of 100
                        }

                        // Release semaphore permit
                        semaphore.add_permits(1);
                    }
                }

                // Update idle connections count
                {
                    let connections_read = connections.read().await;
                    let idle_count = connections_read
                        .values()
                        .filter(|conn| {
                            conn.state == ConnectionState::Active
                                && now.duration_since(conn.last_activity) > Duration::from_secs(30)
                        })
                        .count();

                    let mut stats = statistics.write().await;
                    stats.idle_connections = idle_count;
                }
            }
        });
    }

    /// Get connection pool statistics
    pub async fn get_statistics(&self) -> ConnectionStatistics {
        let mut stats = self.statistics.read().await.clone();

        // Update current connection counts
        let connections = self.connections.read().await;
        stats.total_connections = connections.len();
        stats.active_connections =
            connections.values().filter(|conn| conn.state == ConnectionState::Active).count();
        stats.idle_connections = connections
            .values()
            .filter(|conn| {
                conn.state == ConnectionState::Active
                    && Instant::now().duration_since(conn.last_activity) > Duration::from_secs(30)
            })
            .count();

        stats.connection_utilization =
            stats.active_connections as f64 / self.config.connection_pool_size as f64;

        stats
    }

    /// Get detailed connection information
    pub async fn get_connection_details(&self) -> HashMap<String, ConnectionDetails> {
        let connections = self.connections.read().await;
        let mut details = HashMap::new();

        for (id, connection) in connections.iter() {
            let detail = ConnectionDetails {
                id: connection.id.clone(),
                client_ip: connection.client_ip.clone(),
                state: connection.state.clone(),
                established_at: connection.established_at,
                last_activity: connection.last_activity,
                request_count: connection.request_count,
                duration_seconds: connection.established_at.elapsed().as_secs(),
                idle_time_seconds: connection.last_activity.elapsed().as_secs(),
            };
            details.insert(id.clone(), detail);
        }

        details
    }

    /// Shutdown the connection pool
    pub async fn shutdown(&self) -> Result<()> {
        println!("Shutting down connection pool");

        // Close all active connections
        let connection_ids: Vec<String> = {
            let connections = self.connections.read().await;
            connections.keys().cloned().collect()
        };

        for connection_id in connection_ids {
            self.release_connection(&connection_id).await?;
        }

        Ok(())
    }
}

/// Detailed connection information
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConnectionDetails {
    pub id: String,
    pub client_ip: String,
    pub state: ConnectionState,
    pub established_at: Instant,
    pub last_activity: Instant,
    pub request_count: u64,
    pub duration_seconds: u64,
    pub idle_time_seconds: u64,
}

impl serde::Serialize for ConnectionState {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ConnectionState::Active => serializer.serialize_str("active"),
            ConnectionState::Idle => serializer.serialize_str("idle"),
            ConnectionState::Closing => serializer.serialize_str("closing"),
            ConnectionState::Closed => serializer.serialize_str("closed"),
        }
    }
}
