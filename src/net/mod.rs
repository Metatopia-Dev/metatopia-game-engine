//! Multiplayer Networking Subsystem
//!
//! Provides a complete, low-latency UDP multiplayer networking stack:
//! - Binary packet serialization (`ClientMessage`, `ServerMessage`)
//! - Non-blocking UDP transport with sequencing and reliability channels
//! - Authoritative game server (`NetServer`) managing client sessions & snapshot broadcasts
//! - Multiplayer client (`NetClient`) with input queuing and RTT tracking
//! - Entity replication, snapshot interpolation, and client-side prediction

pub mod protocol;
pub mod transport;
pub mod replication;
pub mod server;
pub mod client;

pub use protocol::{ClientMessage, ServerMessage, EntityState, ChannelType, PROTOCOL_VERSION, PROTOCOL_MAGIC};
pub use transport::{UdpTransport, Packet, PacketHeader};
pub use replication::{Snapshot, SnapshotBuffer, ClientPrediction, SavedInput};
pub use server::{NetServer, ClientSession};
pub use client::{NetClient, ConnectionState};
