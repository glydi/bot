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
pub mod preview;
pub mod recorded;
pub mod router;
pub mod timeline;
pub mod types;

pub use channel::{CommandQueue, ObservationRing, RingReceiver, RingSender};
pub use clock::{Clock, FakeClock, RealClock};
pub use preview::{MODALITY_CAMERA_PREVIEW, PREVIEW_MAX_WIDTH, Preview, PreviewFace};
pub use recorded::{Recorded, RecordedHint, RecordedPayload};
pub use router::{CommandRouter, RouterHandle};
pub use timeline::{Stage, TurnSummary, TurnTimeline, summary_line};
pub use types::{Command, EntityHint, EntityId, Observation, Payload, Priority};
