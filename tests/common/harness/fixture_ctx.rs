/// Type alias for fixture context that provides stable API across feature flags
/// 
/// When fixtures feature is enabled, provides access to FixtureManager
/// When disabled, uses unit type for no-op behavior

#[cfg(feature = "fixtures")]
pub type FixtureCtx<'a> = &'a super::super::fixtures::FixtureManager;

#[cfg(not(feature = "fixtures"))]
pub type FixtureCtx<'a> = ();