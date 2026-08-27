//! Non-blocking UDP Transport and Reliability Layer
//!
//! Provides packet framing, sequence tracking, ACK bitfields, and non-blocking I/O.

use std::net::{UdpSocket, SocketAddr};
use std::io;
use super::protocol::{PROTOCOL_MAGIC, ChannelType};

const HEADER_SIZE: usize = 11; // 2B magic + 1B channel + 2B seq + 2B ack + 4B ack_bits

/// Framed network packet header
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketHeader {
    pub magic: u16,
    pub channel: ChannelType,
    pub seq: u16,
    pub ack: u16,
    pub ack_bits: u32,
}

impl PacketHeader {
    pub fn encode(&self, buf: &mut [u8]) {
        buf[0..2].copy_from_slice(&self.magic.to_le_bytes());
        buf[2] = self.channel as u8;
        buf[3..5].copy_from_slice(&self.seq.to_le_bytes());
        buf[5..7].copy_from_slice(&self.ack.to_le_bytes());
        buf[7..11].copy_from_slice(&self.ack_bits.to_le_bytes());
    }

    pub fn decode(buf: &[u8]) -> Result<Self, String> {
        if buf.len() < HEADER_SIZE {
            return Err("Packet smaller than header size".into());
        }
        let magic = u16::from_le_bytes(buf[0..2].try_into().unwrap());
        if magic != PROTOCOL_MAGIC {
            return Err("Invalid protocol magic".into());
        }
        let channel = match buf[2] {
            0 => ChannelType::Unreliable,
            1 => ChannelType::Reliable,
            _ => return Err("Invalid channel type".into()),
        };
        let seq = u16::from_le_bytes(buf[3..5].try_into().unwrap());
        let ack = u16::from_le_bytes(buf[5..7].try_into().unwrap());
        let ack_bits = u32::from_le_bytes(buf[7..11].try_into().unwrap());

        Ok(Self { magic, channel, seq, ack, ack_bits })
    }
}

/// Received framed packet
#[derive(Debug, Clone)]
pub struct Packet {
    pub header: PacketHeader,
    pub payload: Vec<u8>,
}

/// Non-blocking UDP Socket Transport
pub struct UdpTransport {
    pub socket: UdpSocket,
    pub local_seq: u16,
    pub remote_seq: u16,
    pub ack_bits: u32,
}

impl UdpTransport {
    /// Bind to a local address (e.g. "0.0.0.0:7777" or "127.0.0.1:0" for client)
    pub fn bind(addr: &str) -> io::Result<Self> {
        let socket = UdpSocket::bind(addr)?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket,
            local_seq: 0,
            remote_seq: 0,
            ack_bits: 0,
        })
    }

    /// Send a framed payload to target address
    pub fn send(&mut self, target: SocketAddr, channel: ChannelType, payload: &[u8]) -> io::Result<usize> {
        self.local_seq = self.local_seq.wrapping_add(1);
        let header = PacketHeader {
            magic: PROTOCOL_MAGIC,
            channel,
            seq: self.local_seq,
            ack: self.remote_seq,
            ack_bits: self.ack_bits,
        };

        let mut buffer = vec![0u8; HEADER_SIZE + payload.len()];
        header.encode(&mut buffer[0..HEADER_SIZE]);
        buffer[HEADER_SIZE..].copy_from_slice(payload);

        self.socket.send_to(&buffer, target)
    }

    /// Receive the next available packet without blocking
    pub fn recv(&mut self) -> io::Result<Option<(Packet, SocketAddr)>> {
        let mut buffer = [0u8; 1500]; // Standard MTU size
        match self.socket.recv_from(&mut buffer) {
            Ok((len, addr)) => {
                if len < HEADER_SIZE {
                    return Ok(None);
                }
                if let Ok(header) = PacketHeader::decode(&buffer[0..HEADER_SIZE]) {
                    // Update remote sequence & ACK bitfield
                    if header.seq > self.remote_seq || (self.remote_seq > 60000 && header.seq < 5000) {
                        let diff = header.seq.wrapping_sub(self.remote_seq) as u32;
                        if diff <= 32 {
                            self.ack_bits = (self.ack_bits << diff) | (1 << (diff - 1));
                        } else {
                            self.ack_bits = 0;
                        }
                        self.remote_seq = header.seq;
                    }

                    let payload = buffer[HEADER_SIZE..len].to_vec();
                    Ok(Some((Packet { header, payload }, addr)))
                } else {
                    Ok(None)
                }
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_header_encoding() {
        let header = PacketHeader {
            magic: PROTOCOL_MAGIC,
            channel: ChannelType::Reliable,
            seq: 1042,
            ack: 1040,
            ack_bits: 0b1101,
        };

        let mut buf = [0u8; HEADER_SIZE];
        header.encode(&mut buf);

        let decoded = PacketHeader::decode(&buf).unwrap();
        assert_eq!(header, decoded);
    }

    #[test]
    fn test_loopback_transport_send_recv() {
        let mut server = UdpTransport::bind("127.0.0.1:0").unwrap();
        let mut client = UdpTransport::bind("127.0.0.1:0").unwrap();

        let s_addr = server.socket.local_addr().unwrap();
        let c_addr = client.socket.local_addr().unwrap();

        let data = b"METATOPIA_NET_TEST";
        client.send(s_addr, ChannelType::Reliable, data).unwrap();

        // Server receives
        std::thread::sleep(std::time::Duration::from_millis(5));
        if let Some((packet, sender)) = server.recv().unwrap() {
            assert_eq!(sender, c_addr);
            assert_eq!(packet.payload, data);
            assert_eq!(packet.header.channel, ChannelType::Reliable);
        } else {
            panic!("Expected packet on server");
        }
    }
}
