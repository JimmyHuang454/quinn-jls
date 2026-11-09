//! Time-windowed minimum and maximum of RTT samples

use std::collections::VecDeque;

use crate::{Duration, Instant};

/// Number of slots the window is divided into
///
/// A new sample that arrives within `window / SLOTS` of the previous kept sample is merged into
/// it, which bounds the memory to about `SLOTS` entries whatever the packet rate. The cost is
/// that a merged extreme can outlive the window by at most one slot.
const SLOTS: u32 = 64;

/// Minimum (or maximum) of the samples taken during the last `window`
///
/// A monotonic deque: front is the current extreme, values behind it are the extremes of
/// successively shorter suffixes of the window. The window length may change between updates
/// (Copa's RTT_standing window is half the smoothed RTT).
#[derive(Debug, Clone)]
pub(super) struct WindowedFilter {
    samples: VecDeque<Entry>,
    kind: Kind,
}

#[derive(Debug, Clone, Copy)]
struct Entry {
    /// When the slot of this entry started; later samples merge into it until one slot later
    slot_start: Instant,
    /// When the last sample merged into this entry was taken; the entry expires from here
    time: Instant,
    value: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Min,
    Max,
}

impl WindowedFilter {
    pub(super) fn new(kind: Kind) -> Self {
        Self {
            samples: VecDeque::new(),
            kind,
        }
    }

    /// `a` is at least as extreme as `b`
    fn dominates(&self, a: Duration, b: Duration) -> bool {
        match self.kind {
            Kind::Min => a <= b,
            Kind::Max => a >= b,
        }
    }

    /// Add `value` taken at `now`, and drop the samples older than `window`
    pub(super) fn update(&mut self, now: Instant, value: Duration, window: Duration) {
        // Samples behind the new one that it dominates can never be the extreme again
        while let Some(back) = self.samples.back() {
            if self.dominates(value, back.value) {
                self.samples.pop_back();
            } else {
                break;
            }
        }
        let slot = window / SLOTS;
        match self.samples.back_mut() {
            // The kept sample is more extreme and recent: let it stand for this one too
            Some(back) if now.saturating_duration_since(back.slot_start) < slot => {
                back.time = back.time.max(now)
            }
            _ => self.samples.push_back(Entry {
                slot_start: now,
                time: now,
                value,
            }),
        }
        self.expire(now, window);
    }

    fn expire(&mut self, now: Instant, window: Duration) {
        while self.samples.len() > 1 {
            if now.saturating_duration_since(self.samples[0].time) > window {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// The extreme over the window, or `None` before the first sample
    pub(super) fn get(&self) -> Option<Duration> {
        self.samples.front().map(|e| e.value)
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.samples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn min_tracks_window() {
        let t0 = Instant::now();
        let w = ms(100);
        let mut f = WindowedFilter::new(Kind::Min);
        assert_eq!(f.get(), None);
        f.update(t0, ms(50), w);
        f.update(t0 + ms(10), ms(70), w);
        f.update(t0 + ms(20), ms(60), w);
        assert_eq!(f.get(), Some(ms(50)));
        // 50 ms expires; 60 ms (taken at t0+20) is the minimum of the remaining window
        f.update(t0 + ms(115), ms(80), w);
        assert_eq!(f.get(), Some(ms(60)));
        // Everything but the newest sample expires
        f.update(t0 + ms(300), ms(90), w);
        assert_eq!(f.get(), Some(ms(90)));
    }

    #[test]
    fn max_tracks_window() {
        let t0 = Instant::now();
        let w = ms(100);
        let mut f = WindowedFilter::new(Kind::Max);
        f.update(t0, ms(50), w);
        f.update(t0 + ms(10), ms(70), w);
        f.update(t0 + ms(20), ms(60), w);
        assert_eq!(f.get(), Some(ms(70)));
        f.update(t0 + ms(111), ms(55), w);
        assert_eq!(f.get(), Some(ms(60)));
    }

    #[test]
    fn shrinking_window_expires_old_samples() {
        let t0 = Instant::now();
        let mut f = WindowedFilter::new(Kind::Min);
        f.update(t0, ms(10), ms(1000));
        f.update(t0 + ms(50), ms(20), ms(1000));
        assert_eq!(f.get(), Some(ms(10)));
        f.update(t0 + ms(60), ms(30), ms(20));
        assert_eq!(f.get(), Some(ms(20)));
    }

    #[test]
    fn memory_is_bounded() {
        // Strictly increasing samples at a high rate would keep every sample in a plain
        // monotonic deque
        let t0 = Instant::now();
        let mut f = WindowedFilter::new(Kind::Min);
        for i in 0..100_000u64 {
            f.update(
                t0 + Duration::from_micros(i * 100),
                Duration::from_micros(i),
                Duration::from_secs(10),
            );
        }
        assert!(f.len() <= SLOTS as usize + 2, "len {}", f.len());
        assert_eq!(f.get(), Some(Duration::ZERO));
    }

    #[test]
    fn time_going_backwards_does_not_panic() {
        let t0 = Instant::now() + ms(100);
        let mut f = WindowedFilter::new(Kind::Min);
        f.update(t0, ms(10), ms(50));
        f.update(t0 - ms(50), ms(20), ms(50));
        assert_eq!(f.get(), Some(ms(10)));
    }
}
