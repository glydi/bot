//! Command routing moved to `common::router` (one `CommandQueue`, many
//! actuators, fanned out by target). Re-exported here so existing callers
//! keep compiling.

pub use common::router::{CommandRouter, RouterHandle};
