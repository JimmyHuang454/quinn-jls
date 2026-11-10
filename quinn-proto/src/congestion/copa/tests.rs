//! Unit tests that drive [`Copa`] through the [`Controller`] interface

use super::*;
use crate::packet::SpaceId;

const MTU: u64 = 1200;

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

fn rtt_estimator(srtt: Duration) -> RttEstimator {
    let mut rtt = RttEstimator::new(srtt);
    rtt.update(Duration::ZERO, srtt);
    rtt
}

fn copa(config: CopaConfig) -> (Copa, Instant) {
    let now = Instant::now();
    (Copa::new(Arc::new(config), now, MTU as u16), now)
}

/// Acknowledge one MTU-sized packet whose RTT was `rtt_sample`
fn ack(c: &mut Copa, now: Instant, rtt_sample: Duration, rtt: &RttEstimator) {
    c.on_ack(now, now - rtt_sample, MTU, 0, SpaceId::Data, false, rtt);
}

/// RTT_min = 50 ms from a first sample, then RTT_standing = 60 ms (dq = 10 ms), with a smoothed
/// RTT of 60 ms. Returns the time of the last sample.
fn ten_ms_queue(c: &mut Copa, t0: Instant, rtt: &RttEstimator) -> Instant {
    ack(c, t0, ms(50), rtt);
    let t = t0 + ms(100);
    ack(c, t, ms(60), rtt);
    t
}

#[test]
fn increases_below_target_rate() {
    // delta = 0.5, dq = 10 ms: target 200 packets/s; 10 packets / 60 ms = 167 packets/s
    let (mut c, t0) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(60));
    let t = ten_ms_queue(&mut c, t0, &rtt);
    c.slow_start = false;
    c.window = (10 * MTU) as f64;
    ack(&mut c, t + ms(1), ms(60), &rtt);
    assert_eq!(c.queueing_delay(), ms(10));
    assert!(c.window > (10 * MTU) as f64);
}

#[test]
fn decreases_above_target_rate() {
    // 20 packets / 60 ms = 333 packets/s, above the target of 200
    let (mut c, t0) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(60));
    let t = ten_ms_queue(&mut c, t0, &rtt);
    c.slow_start = false;
    c.window = (20 * MTU) as f64;
    ack(&mut c, t + ms(1), ms(60), &rtt);
    assert!(c.window < (20 * MTU) as f64);
}

#[test]
fn step_is_velocity_over_delta_cwnd() {
    let (mut c, t0) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(60));
    let t = ten_ms_queue(&mut c, t0, &rtt);
    c.slow_start = false;
    c.window = (20 * MTU) as f64;
    let before = c.window;
    ack(&mut c, t + ms(1), ms(60), &rtt);
    // v / (delta cwnd) = 1 / (0.5 * 20) packets
    let expected = MTU as f64 / (0.5 * 20.0);
    assert_eq!(c.velocity(), 1);
    assert!((before - c.window - expected).abs() < 1e-6);
}

#[test]
fn slow_start_doubles_until_first_decrease() {
    let (mut c, t0) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(50));
    let w0 = c.window();
    // No queueing: every ACK increases the window by the acknowledged bytes
    for i in 0..10 {
        ack(&mut c, t0 + ms(i), ms(50), &rtt);
    }
    assert!(c.in_slow_start());
    assert_eq!(c.window(), w0 + 10 * MTU);

    // A 30 ms queue: target 67 packets/s, far below 22 packets / 80 ms
    ack(&mut c, t0 + ms(200), ms(80), &rtt);
    assert!(!c.in_slow_start());
    let w = c.window;
    // Back to no queueing: the window grows by v / (delta cwnd), not by a whole packet
    ack(&mut c, t0 + ms(20_000), ms(50), &rtt);
    ack(&mut c, t0 + ms(20_001), ms(50), &rtt);
    assert!(!c.in_slow_start());
    assert!(c.window - w < MTU as f64 / 2.0);
}

/// Drive `duration` of one ACK per millisecond with a constant RTT sample
fn run(c: &mut Copa, t: &mut Instant, duration: Duration, sample: Duration, rtt: &RttEstimator) {
    let end = *t + duration;
    while *t < end {
        ack(c, *t, sample, rtt);
        *t += ms(1);
    }
}

#[test]
fn velocity_doubles_after_three_rtts_and_resets_on_reversal() {
    for rule in [CopaVelocity::Paper, CopaVelocity::GenericCc] {
        let mut config = CopaConfig::default();
        config.velocity(rule);
        let (mut c, mut t) = copa(config);
        let rtt = rtt_estimator(ms(50));
        c.slow_start = false;
        c.window = (200 * MTU) as f64;
        // No queueing: the window keeps increasing
        run(&mut c, &mut t, ms(400), ms(50), &rtt);
        assert!(c.velocity() >= 4, "{rule:?}: velocity {}", c.velocity());

        // A large queue: the window decreases, and the velocity restarts from 1
        let mut lowest = c.velocity();
        for _ in 0..700 {
            run(&mut c, &mut t, ms(1), ms(300), &rtt);
            lowest = lowest.min(c.velocity());
        }
        assert_eq!(lowest, 1, "{rule:?}");
    }
}

#[test]
fn velocity_is_capped_by_delta_cwnd() {
    let (mut c, mut t) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(50));
    c.slow_start = false;
    c.window = (4 * MTU) as f64;
    for _ in 0..200 {
        run(&mut c, &mut t, ms(10), ms(50), &rtt);
        assert!(c.velocity() as f64 <= (0.5 * c.window_packets()).max(1.0));
    }
}

#[test]
fn default_mode_ignores_loss() {
    let (mut c, t0) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(60));
    let t = ten_ms_queue(&mut c, t0, &rtt);
    let (w, delta) = (c.window(), c.delta());
    c.on_congestion_event(t, t, false, false, 10 * MTU, 0, SpaceId::Data);
    c.on_congestion_event(t + ms(500), t, false, true, 0, 0, SpaceId::Data);
    assert_eq!(c.window(), w);
    assert_eq!(c.delta(), delta);
    assert!(!c.is_competitive());
}

/// A queue that never drains: RTT_min 50 ms, then every sample at 70 ms
fn standing_queue(c: &mut Copa, t: &mut Instant, duration: Duration, rtt: &RttEstimator) {
    ack(c, *t, ms(50), rtt);
    *t += ms(1);
    run(c, t, duration, ms(70), rtt);
}

#[test]
fn enters_competitive_mode_when_queue_never_drains() {
    let (mut c, mut t) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(50));
    standing_queue(&mut c, &mut t, ms(200), &rtt);
    assert!(!c.is_competitive(), "not before 5 RTTs");
    run(&mut c, &mut t, ms(100), ms(70), &rtt);
    assert!(c.is_competitive());
    // 1/delta grows by about one per window of ACKs
    let inv = 1.0 / c.delta();
    assert!(inv > 2.0);
    run(&mut c, &mut t, ms(200), ms(70), &rtt);
    assert!(1.0 / c.delta() > inv);
}

#[test]
fn competitive_mode_halves_inverse_delta_once_per_rtt() {
    let (mut c, mut t) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(50));
    standing_queue(&mut c, &mut t, ms(300), &rtt);
    assert!(c.is_competitive());
    c.delta = 0.1;
    c.on_congestion_event(t, t, false, false, MTU, 0, SpaceId::Data);
    assert!((c.delta() - 0.2).abs() < 1e-12);
    // Same RTT: ignored
    c.on_congestion_event(t + ms(10), t, false, false, MTU, 0, SpaceId::Data);
    assert!((c.delta() - 0.2).abs() < 1e-12);
    // Next RTT, and ECN counts as loss; never above the configured delta
    c.on_congestion_event(t + ms(60), t, false, true, 0, 0, SpaceId::Data);
    assert!((c.delta() - 0.4).abs() < 1e-12);
    c.on_congestion_event(t + ms(120), t, false, false, MTU, 0, SpaceId::Data);
    assert!((c.delta() - 0.5).abs() < 1e-12);
}

#[test]
fn spurious_loss_restores_delta_in_competitive_mode() {
    let (mut c, mut t) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(50));
    standing_queue(&mut c, &mut t, ms(300), &rtt);
    c.delta = 0.1;
    c.on_congestion_event(t, t, false, false, MTU, 0, SpaceId::Data);
    c.on_spurious_congestion_event();
    assert!((c.delta() - 0.1).abs() < 1e-12);
}

#[test]
fn returns_to_default_mode_when_queue_drains() {
    let (mut c, mut t) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(50));
    standing_queue(&mut c, &mut t, ms(300), &rtt);
    assert!(c.is_competitive());
    // RTT_standing (25 ms window) falls back to RTT_min
    run(&mut c, &mut t, ms(40), ms(50), &rtt);
    assert!(!c.is_competitive());
    assert_eq!(c.delta(), 0.5);
}

#[test]
fn mode_switching_can_be_disabled() {
    for config in [CopaConfig::copa_nc(), CopaConfig::copa_auto_nc()] {
        let (mut c, mut t) = copa(config);
        let rtt = rtt_estimator(ms(50));
        standing_queue(&mut c, &mut t, ms(1000), &rtt);
        assert!(!c.is_competitive());
        let delta = c.delta();
        c.on_congestion_event(t, t, false, false, MTU, 0, SpaceId::Data);
        assert_eq!(c.delta(), delta);
    }
}

#[test]
fn persistent_congestion_restarts_slow_start() {
    let (mut c, t0) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(60));
    let t = ten_ms_queue(&mut c, t0, &rtt);
    c.slow_start = false;
    c.velocity = 8;
    c.window = (100 * MTU) as f64;
    c.on_congestion_event(t, t, true, false, 100 * MTU, 0, SpaceId::Data);
    assert_eq!(c.window(), MIN_WINDOW_PACKETS * MTU);
    assert!(c.in_slow_start());
    assert_eq!(c.velocity(), 1);
}

#[test]
fn app_limited_does_not_grow_the_window() {
    let (mut c, t0) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(50));
    let w = c.window();
    for i in 0..10 {
        c.on_ack(t0 + ms(i), t0 + ms(i) - ms(50), MTU, 0, SpaceId::Data, true, &rtt);
    }
    assert_eq!(c.window(), w);
    // The RTT filters are still updated
    assert_eq!(c.min_rtt(), Some(ms(50)));
    assert_eq!(c.standing_rtt(), Some(ms(50)));
}

#[test]
fn window_bounds_and_mtu_update() {
    let (mut c, t0) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(50));
    c.slow_start = false;
    c.window = c.minimum_window();
    // A huge queue cannot push the window below the minimum
    ack(&mut c, t0, ms(50), &rtt);
    for i in 0..100 {
        ack(&mut c, t0 + ms(100 + i), ms(5000), &rtt);
    }
    assert_eq!(c.window(), 4 * MTU);
    c.on_mtu_update(1500);
    assert_eq!(c.window(), 4 * 1500);

    // The window never exceeds its upper bound, even after days without queueing
    c.window = MAX_WINDOW;
    let later = t0 + Duration::from_secs(10 * 24 * 3600);
    ack(&mut c, later, ms(50), &rtt);
    assert_eq!(c.window(), MAX_WINDOW as u64);
}

#[test]
fn time_going_backwards_does_not_panic() {
    let (mut c, t0) = copa(CopaConfig::default());
    let rtt = rtt_estimator(ms(50));
    let t = t0 + ms(100);
    ack(&mut c, t, ms(50), &rtt);
    // An ACK "sent" after it was received yields a zero sample
    c.on_ack(t - ms(10), t, MTU, 0, SpaceId::Data, false, &rtt);
    assert!(c.window() >= 4 * MTU);
}

#[test]
fn pacing_rate_and_metrics() {
    let (mut c, t0) = copa(CopaConfig::default());
    assert_eq!(c.pacing_rate(), None);
    assert_eq!(c.metrics().pacing_rate, None);
    let rtt = rtt_estimator(ms(60));
    let t = ten_ms_queue(&mut c, t0, &rtt);
    ack(&mut c, t + ms(1), ms(60), &rtt);
    let expected = (2.0 * c.window / 0.060) as u64;
    assert_eq!(c.pacing_rate(), Some(expected));
    let metrics = c.metrics();
    assert_eq!(metrics.congestion_window, c.window());
    assert_eq!(metrics.pacing_rate, Some(expected));
}

#[test]
fn clone_box_and_into_any() {
    let (mut c, t0) = copa(CopaConfig::copa_auto());
    let rtt = rtt_estimator(ms(60));
    ten_ms_queue(&mut c, t0, &rtt);
    let boxed = c.clone_box();
    assert_eq!(boxed.window(), c.window());
    assert_eq!(boxed.initial_window(), CopaConfig::default().initial_window);
    let back = boxed.into_any().downcast::<Copa>().expect("a Copa");
    assert_eq!(back.queueing_delay(), c.queueing_delay());
    assert!(back.adaptive_estimate().is_some());
}

#[test]
fn factory_presets() {
    let presets = [
        (CopaConfig::default(), true, false),
        (CopaConfig::copa_nc(), false, false),
        (CopaConfig::copa_auto(), true, true),
        (CopaConfig::copa_auto_nc(), false, true),
    ];
    for (config, mode_switching, adaptive) in presets {
        assert_eq!(config.delta, 0.5);
        assert_eq!(config.mode_switching, mode_switching);
        assert_eq!(config.adaptive.is_some(), adaptive);
        let c = Arc::new(config).build(Instant::now(), MTU as u16);
        let c = c.into_any().downcast::<Copa>().unwrap();
        assert_eq!(c.adaptive_estimate().is_some(), adaptive);
    }
}

#[test]
#[should_panic]
fn delta_must_be_positive() {
    CopaConfig::default().delta(0.0);
}

#[test]
fn copa_auto_uses_delta_max_while_probing() {
    let mut auto = CopaAutoConfig::default();
    auto.delta_range(0.001, 0.4)
        .probe_timing(Duration::ZERO, Duration::ZERO);
    let mut config = CopaConfig::default();
    config.adaptive(Some(auto));
    let (mut c, t0) = copa(config);
    let rtt = rtt_estimator(ms(60));
    let t = ten_ms_queue(&mut c, t0, &rtt);
    // Slow start: the configured delta
    assert_eq!(c.delta(), 0.5);
    c.slow_start = false;
    // The first half-periods are a low-load probe
    ack(&mut c, t + ms(1), ms(60), &rtt);
    assert_eq!(c.delta(), 0.4);
}

#[test]
fn pacing_is_capped_for_one_rtt_after_a_restore() {
    let (mut c, t0) = copa(CopaConfig::copa_auto());
    let rtt = rtt_estimator(ms(60));
    let t = ten_ms_queue(&mut c, t0, &rtt);
    let uncapped = c.pacing_rate().unwrap();
    c.pace_cap = Some((t + ms(60), uncapped as f64 / 4.0));
    assert_eq!(c.pacing_rate(), Some(uncapped / 4));
    // The cap ends one smoothed RTT later
    ack(&mut c, t + ms(61), ms(60), &rtt);
    assert!(c.pace_cap.is_none());
    assert!(c.pacing_rate().unwrap() > uncapped / 2);
}
