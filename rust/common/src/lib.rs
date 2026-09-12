//! Shared contracts for every GLYDI crate: what a sense emits, what an
//! actuator consumes, how time is read, and the two channels that connect
//! them.
//!
//! Everything in the loop
//!
//! ```text
//! SENSE -> Observation -> MIND -> Command -> ACT
//! ```
//!
//! goes through the types here. `mind` never learns what a camera is: a
//! sense is anything that produces [`Observation`]s, an actuator is anything
//! that consumes [`Command`]s.

pub mod channel;
pub mod clock;
pub mod types;

pub use channel::{CommandQueue, ObservationRing, RingReceiver, RingSender};
pub use clock::{Clock, FakeClock, RealClock};
pub use types::{Command, EntityHint, EntityId, Observation, Payload, Priority};
