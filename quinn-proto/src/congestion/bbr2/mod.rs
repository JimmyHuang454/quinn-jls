//! BBRv2 congestion control
//!
//! The algorithm (`core`, `bandwidth_sampler`, `windowed_filter`, `bandwidth`) is taken from
//! Cloudflare quiche 0.24.8 (`src/recovery/gcongestion/bbr2`), itself a port of Chromium's
//! `Bbr2Sender`. Those files keep their original Chromium and Cloudflare BSD license headers.
//! This module adapts quiche's `CongestionControl` interface to quinn's [`Controller`].

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::{BASE_DATAGRAM_SIZE, Controller, ControllerFactory, ControllerMetrics};
use crate::connection::RttEstimator;
use crate::packet::SpaceId;
use crate::{Duration, Instant};

// Vendored from quiche: keep the code close to upstream rather than trimming unused items
#[allow(unreachable_pub, dead_code, clippy::all, clippy::use_self)]
mod bandwidth;
#[allow(unreachable_pub, dead_code, clippy::all, clippy::use_self)]
mod bandwidth_sampler;
#[allow(unreachable_pub, dead_code, clippy::all, clippy::use_self)]
mod core;
#[allow(unreachable_pub, dead_code, clippy::all, clippy::use_self)]
mod windowed_filter;

use self::bandwidth::Bandwidth;
use self::core::BBRv2;

/// Maximum congestion window, in packets (quiche's `MAX_WINDOW_PACKETS`)
const MAX_WINDOW_PACKETS: usize = 20_000;

/// Experimental! BBRv2 congestion control, ported from Cloudflare quiche / Chromium
///
/// Only application data (1-RTT / 0-RTT) packets feed the model: BBRv2's bandwidth sampler
/// needs one monotonically increasing packet number sequence, and the handshake spaces carry
/// too little data to matter.
#[derive(Debug, Clone)]
pub struct Bbr2 {
    config: Arc<Bbr2Config>,
    core: BBRv2,
    /// Ack-eliciting application data packets in flight, by packet number
    outstanding: BTreeMap<u64, u16>,
    /// Bytes in `outstanding`
    in_flight: usize,
    /// Largest application data packet number acknowledged so far
    largest_acked: Option<u64>,
    /// Packets acknowledged since the last [`Controller::on_end_acks`]
    acked: Vec<Acked>,
    /// Bytes in `acked`
    acked_bytes: usize,
    /// Packets declared lost but not yet reported to the model
    lost: Vec<Lost>,
    /// Bytes in `lost`
    lost_bytes: usize,
}

impl Bbr2 {
    /// Construct a state using the given `config` and current time `now`
    pub fn new(config: Arc<Bbr2Config>, current_mtu: u16) -> Self {
        let mss = usize::from(current_mtu);
        let initial_packets = (config.initial_window as usize).div_ceil(mss).max(1);
        let core = BBRv2::new(
            initial_packets,
            MAX_WINDOW_PACKETS,
            mss,
            config.initial_rtt,
            config.params.as_ref(),
        );
        Self {
            config,
            core,
            outstanding: BTreeMap::new(),
            in_flight: 0,
            largest_acked: None,
            acked: Vec::new(),
            acked_bytes: 0,
            lost: Vec::new(),
            lost_bytes: 0,
        }
    }

    fn least_unacked(&self) -> u64 {
        match self.outstanding.first_key_value() {
            Some((&pn, _)) => pn,
            None => self.largest_acked.map_or(0, |pn| pn + 1),
        }
    }

    /// Report the pending acknowledgements and losses to the model as one congestion event
    fn flush(&mut self, now: Instant) {
        if self.acked.is_empty() && self.lost.is_empty() {
            return;
        }
        let prior_in_flight = self.in_flight + self.acked_bytes + self.lost_bytes;
        self.core.on_congestion_event(
            true,
            prior_in_flight,
            self.in_flight,
            now,
            &self.acked,
            &self.lost,
            self.least_unacked(),
            &RttStats,
            &mut RecoveryStats,
        );
        self.acked.clear();
        self.lost.clear();
        self.acked_bytes = 0;
        self.lost_bytes = 0;
    }
}

impl Controller for Bbr2 {
    fn on_packet_sent(&mut self, now: Instant, bytes: u16, packet_number: u64, space: SpaceId) {
        if space != SpaceId::Data {
            return;
        }
        if self
            .outstanding
            .last_key_value()
            .is_some_and(|(&last, _)| last >= packet_number)
        {
            return;
        }
        self.core
            .on_packet_sent(now, self.in_flight, packet_number, bytes.into(), true);
        self.outstanding.insert(packet_number, bytes);
        self.in_flight += usize::from(bytes);
    }

    fn on_ack(
        &mut self,
        _now: Instant,
        sent: Instant,
        _bytes: u64,
        pn: u64,
        space: SpaceId,
        _app_limited: bool,
        _rtt: &RttEstimator,
    ) {
        if space != SpaceId::Data {
            return;
        }
        let Some(bytes) = self.outstanding.remove(&pn) else {
            return;
        };
        self.in_flight -= usize::from(bytes);
        self.acked_bytes += usize::from(bytes);
        self.largest_acked = Some(self.largest_acked.map_or(pn, |l| l.max(pn)));
        self.acked.push(Acked {
            pkt_num: pn,
            time_sent: sent,
        });
    }

    fn on_end_acks(
        &mut self,
        now: Instant,
        _in_flight: u64,
        _app_limited: bool,
        _largest_packet_num_acked: Option<u64>,
        space: SpaceId,
    ) {
        if space == SpaceId::Data {
            self.flush(now);
        }
    }

    fn on_packet_lost(
        &mut self,
        _lost_bytes: u16,
        packet_number: u64,
        space: SpaceId,
        _now: Instant,
    ) {
        if space != SpaceId::Data {
            return;
        }
        let Some(bytes) = self.outstanding.remove(&packet_number) else {
            return;
        };
        self.in_flight -= usize::from(bytes);
        self.lost_bytes += usize::from(bytes);
        self.lost.push(Lost {
            packet_number,
            bytes_lost: bytes.into(),
        });
    }

    fn on_congestion_event(
        &mut self,
        now: Instant,
        _sent: Instant,
        _is_persistent_congestion: bool,
        _is_ecn: bool,
        _lost_bytes: u64,
        _largest_lost: u64,
        space: SpaceId,
    ) {
        // Losses are reported packet by packet through `on_packet_lost` first. BBRv2 ignores ECN
        // and has no reaction to persistent congestion (quiche's `on_retransmission_timeout` is
        // a no-op).
        if space == SpaceId::Data {
            self.flush(now);
        }
    }

    fn on_packet_discarded(&mut self, _bytes: u16, packet_number: u64, space: SpaceId) {
        if space != SpaceId::Data {
            return;
        }
        if let Some(bytes) = self.outstanding.remove(&packet_number) {
            self.in_flight -= usize::from(bytes);
            self.core.on_packet_neutered(packet_number);
        }
    }

    fn on_app_limited(&mut self, _in_flight: u64) {
        self.core.on_app_limited(self.in_flight);
    }

    fn on_mtu_update(&mut self, new_mtu: u16) {
        self.core.update_mss(new_mtu.into());
    }

    fn window(&self) -> u64 {
        self.core.get_congestion_window() as u64
    }

    fn metrics(&self) -> ControllerMetrics {
        let bandwidth = self.core.bandwidth_estimate(&RttStats);
        ControllerMetrics {
            congestion_window: self.window(),
            ssthresh: None,
            pacing_rate: Some(
                self.core
                    .pacing_rate(self.in_flight, &RttStats)
                    .to_bytes_per_second()
                    .max(1),
            ),
            bandwidth_estimate: (bandwidth != Bandwidth::zero())
                .then(|| bandwidth.to_bits_per_second()),
            send_quantum: None,
        }
    }

    fn clone_box(&self) -> Box<dyn Controller> {
        Box::new(self.clone())
    }

    fn initial_window(&self) -> u64 {
        self.config.initial_window
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

/// Configuration for the [`Bbr2`] congestion controller
#[derive(Debug, Clone)]
pub struct Bbr2Config {
    initial_window: u64,
    initial_rtt: Duration,
    params: Option<BbrParams>,
}

impl Bbr2Config {
    /// Default limit on the amount of outstanding data in bytes.
    ///
    /// Recommended value: `min(10 * max_datagram_size, max(2 * max_datagram_size, 14720))`
    pub fn initial_window(&mut self, value: u64) -> &mut Self {
        self.initial_window = value;
        self
    }

    /// RTT assumed before the first sample, which sets the initial pacing rate
    ///
    /// Should match `TransportConfig::initial_rtt`.
    pub fn initial_rtt(&mut self, value: Duration) -> &mut Self {
        self.initial_rtt = value;
        self
    }

    /// Override BBRv2 tuning parameters (quiche's `BbrParams`); `None` fields keep the defaults
    pub fn params(&mut self, value: BbrParams) -> &mut Self {
        self.params = Some(value);
        self
    }
}

impl Default for Bbr2Config {
    fn default() -> Self {
        Self {
            initial_window: 14720.clamp(2 * BASE_DATAGRAM_SIZE, 10 * BASE_DATAGRAM_SIZE),
            // quinn's default `TransportConfig::initial_rtt`
            initial_rtt: Duration::from_millis(333),
            params: None,
        }
    }
}

impl ControllerFactory for Bbr2Config {
    fn build(self: Arc<Self>, _now: Instant, current_mtu: u16) -> Box<dyn Controller> {
        Box::new(Bbr2::new(self, current_mtu))
    }
}

/// A packet newly acknowledged
#[derive(Debug, Clone)]
pub(crate) struct Acked {
    pub(crate) pkt_num: u64,
    #[allow(dead_code)]
    pub(crate) time_sent: Instant,
}

/// A packet newly declared lost
#[derive(Debug, Clone)]
pub(crate) struct Lost {
    pub(crate) packet_number: u64,
    pub(crate) bytes_lost: usize,
}

/// BBRv2 tuning parameters, as in quiche; `None` keeps the default
#[derive(Debug, Default, Copy, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct BbrParams {
    pub startup_cwnd_gain: Option<f32>,
    pub startup_pacing_gain: Option<f32>,
    pub full_bw_threshold: Option<f32>,
    pub startup_full_bw_rounds: Option<usize>,
    pub startup_full_loss_count: Option<usize>,
    pub drain_cwnd_gain: Option<f32>,
    pub drain_pacing_gain: Option<f32>,
    pub enable_reno_coexistence: Option<bool>,
    pub enable_overestimate_avoidance: Option<bool>,
    pub choose_a0_point_fix: Option<bool>,
    pub probe_bw_probe_up_pacing_gain: Option<f32>,
    pub probe_bw_probe_down_pacing_gain: Option<f32>,
    pub probe_bw_cwnd_gain: Option<f32>,
    pub probe_bw_up_cwnd_gain: Option<f32>,
    pub probe_rtt_pacing_gain: Option<f32>,
    pub probe_rtt_cwnd_gain: Option<f32>,
    pub max_probe_up_queue_rounds: Option<usize>,
    pub loss_threshold: Option<f32>,
    pub use_bytes_delivered_for_inflight_hi: Option<bool>,
    pub decrease_startup_pacing_at_end_of_round: Option<bool>,
    pub bw_lo_reduction_strategy: Option<BbrBwLoReductionStrategy>,
    pub ignore_app_limited_for_no_bandwidth_growth: Option<bool>,
    pub initial_pacing_rate_bytes_per_second: Option<u64>,
    pub scale_pacing_rate_by_mss: Option<bool>,
    pub disable_probe_down_early_exit: Option<bool>,
    pub time_sent_set_to_now: Option<bool>,
}

/// How BBRv2 reduces `bw_lo` on loss, as in quiche
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum BbrBwLoReductionStrategy {
    Default = 0,
    MinRttReduction = 1,
    InflightReduction = 2,
    CwndReduction = 3,
}

/// quiche's congestion controller interface, which `core` implements
#[allow(dead_code)]
pub(crate) trait CongestionControl: std::fmt::Debug {
    #[cfg(feature = "qlog")]
    fn state_str(&self) -> &'static str;
    fn get_congestion_window(&self) -> usize;
    fn get_congestion_window_in_packets(&self) -> usize;
    fn can_send(&self, bytes_in_flight: usize) -> bool;
    fn on_packet_sent(
        &mut self,
        sent_time: Instant,
        bytes_in_flight: usize,
        packet_number: u64,
        bytes: usize,
        is_retransmissible: bool,
    );
    fn on_packet_neutered(&mut self, _packet_number: u64) {}
    #[allow(clippy::too_many_arguments)]
    fn on_congestion_event(
        &mut self,
        rtt_updated: bool,
        prior_in_flight: usize,
        bytes_in_flight: usize,
        event_time: Instant,
        acked_packets: &[Acked],
        lost_packets: &[Lost],
        least_unacked: u64,
        rtt_stats: &RttStats,
        recovery_stats: &mut RecoveryStats,
    );
    fn on_retransmission_timeout(&mut self, packets_retransmitted: bool);
    fn on_connection_migration(&mut self);
    fn limit_cwnd(&mut self, _max_cwnd: usize) {}
    fn is_in_recovery(&self) -> bool;
    fn is_cwnd_limited(&self, bytes_in_flight: usize) -> bool;
    fn pacing_rate(&self, bytes_in_flight: usize, rtt_stats: &RttStats) -> Bandwidth;
    fn bandwidth_estimate(&self, rtt_stats: &RttStats) -> Bandwidth;
    fn max_bandwidth(&self) -> Bandwidth;
    fn update_mss(&mut self, new_mss: usize);
    fn on_app_limited(&mut self, _bytes_in_flight: usize) {}
}

/// Stand-in for quiche's RTT statistics, which BBRv2 accepts but never reads
pub(crate) struct RttStats;

/// Stand-in for quiche's recovery statistics; BBRv2 only reports its startup exit there
pub(crate) struct RecoveryStats;

impl RecoveryStats {
    pub(crate) fn set_startup_exit(&mut self, _exit: StartupExit) {}
}

/// Why BBRv2 left startup (quiche reports this in its statistics)
#[allow(dead_code)]
pub(crate) struct StartupExit {
    cwnd: usize,
    bandwidth: Option<Bandwidth>,
    reason: StartupExitReason,
}

impl StartupExit {
    pub(crate) fn new(
        cwnd: usize,
        bandwidth: Option<Bandwidth>,
        reason: StartupExitReason,
    ) -> Self {
        Self {
            cwnd,
            bandwidth,
            reason,
        }
    }
}

/// See [`StartupExit`]
#[derive(Debug, Clone, Copy)]
pub(crate) enum StartupExitReason {
    Loss,
    BandwidthPlateau,
    PersistentQueue,
}
