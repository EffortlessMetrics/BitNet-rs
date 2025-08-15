// Helper module for serializing SystemTime to/from JSON
use serde::{Deserialize, Deserializer, Serializer};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub fn serialize<S>(t: &SystemTime, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let dur = t.duration_since(UNIX_EPOCH).map_err(serde::ser::Error::custom)?;
    s.serialize_u64(dur.as_millis() as u64)
}

pub fn deserialize<'de, D>(d: D) -> Result<SystemTime, D::Error>
where
    D: Deserializer<'de>,
{
    let ms = u64::deserialize(d)?;
    Ok(UNIX_EPOCH + Duration::from_millis(ms))
}