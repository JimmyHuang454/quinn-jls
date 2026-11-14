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
        (CopaConfig::default(), 0.5, true, false),
        (CopaConfig::copa_nc(), 0.5, false, false),
        (CopaConfig::copa_auto(), 0.05, true, true),
        (CopaConfig::copa_auto_nc(), 0.05, false, true),
    ];
    for (config, delta, mode_switching, adaptive) in presets {
        assert_eq!(config.delta, delta);
        assert_eq!(config.mode_switching, mode_switching);
        assert_eq!(config.adaptive.is_some(), adaptive);
        let c = Arc::new(config).build(Instant::now(), MTU as u16);
        let c = c.into_any().downcast::<Copa>().unwrap();
        assert_eq!(c.delta(), delta);
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
    c.drain
        .set_recovery_cap_for_test(t + ms(60), uncapped as f64 / 4.0);
    assert_eq!(c.pacing_rate(), Some(uncapped / 4));
    // The cap ends one smoothed RTT later
    ack(&mut c, t + ms(61), ms(60), &rtt);
    assert!(!c.drain.is_recovering());
    assert!(c.pacing_rate().unwrap() > uncapped / 2);
}

#[test]
fn gentle_aligned_probe_holds_bdp_instead_of_collapsing() {
    let make = |gentle: bool, t0: Instant| {
        let mut auto = CopaAutoConfig::default();
        auto.delta_range(0.001, 0.4)
            .probe_epoch(Some(t0))
            .probe_timing(Duration::from_secs(20), Duration::from_secs(2))
            .probe_rtts(0, Duration::ZERO)
            .gentle_probe(gentle);
        let mut config = CopaConfig::copa_nc();
        config.delta(0.1).adaptive(Some(auto));
        let (mut c, t0) = copa(config);
        let rtt = rtt_estimator(ms(60));
        let mut t = ten_ms_queue(&mut c, t0, &rtt);
        // Settle past the initial post-slow-start probe into steady state before slot 1 (20 s).
        while !c.auto.as_ref().unwrap().first_probe_done() {
            t += ms(25);
            ack(&mut c, t, ms(60), &rtt);
        }
        // Establish 50 ms RTT_min, 60 ms RTT_standing, and 60 * MTU cwnd right before slot 1.
        ack(&mut c, t0 + Duration::from_secs(19) + ms(900), ms(50), &rtt);
        for i in 1..=5 {
            ack(
                &mut c,
                t0 + Duration::from_secs(19) + ms(900 + i * 19),
                ms(60),
                &rtt,
            );
        }
        c.window = (60 * MTU) as f64;
        (c, t0, rtt)
    };

    let t0 = Instant::now();
    let (mut gentle, _, rtt) = make(true, t0);
    let (mut legacy, _, _) = make(false, t0);
    let pre_probe_window = gentle.window();
    let pre_probe_rate = gentle.pacing_rate().unwrap();
    let probe_start = t0 + Duration::from_secs(20);

    // Feed 30 ms jitter spikes during the probe in slot 1 (first 200 ms: active calibration).
    for step in 0..40 {
        let now = probe_start + ms(20 + step * 5);
        ack(&mut gentle, now, ms(80), &rtt);
        ack(&mut legacy, now, ms(80), &rtt);
    }

    assert!(
        gentle.drain.is_draining(),
        "gentle probe should be active via DrainController"
    );
    assert!(
        gentle.pacing_rate().unwrap() < pre_probe_rate,
        "gentle probe paces smoothly below pre-probe rate"
    );
    assert!(
        legacy.delta() > gentle.delta(),
        "legacy probe jumps to delta_max ({}) while gentle keeps current delta ({})",
        legacy.delta(),
        gentle.delta()
    );
    // Soft window holds near 0.9 * BDP (~45 * MTU) instead of abruptly dropping to 0.5 * BDP.
    assert!(
        gentle.window() >= 38 * MTU && gentle.window() <= pre_probe_window,
        "gentle probe window {} should stay near soft BDP without stalling",
        gentle.window()
    );

    // Continue through the probe window until plateau early-exit + post-idle sampling finishes.
    let mut restored_window = None;
    for step in 40..200 {
        let now = probe_start + ms(20 + step * 5);
        ack(&mut gentle, now, ms(80), &rtt);
        ack(&mut legacy, now, ms(80), &rtt);
        if restored_window.is_none() && !gentle.drain.is_draining() {
            restored_window = Some(gentle.window());
        }
    }

    assert!(
        legacy.window() < gentle.window(),
        "legacy delta_max collapses under jitter ({}) below gentle ({})",
        legacy.window(),
        gentle.window()
    );
    let restored = restored_window.expect("gentle probe should early-exit after plateau convergence");
    assert!(
        restored >= pre_probe_window - MTU,
        "restored window {restored} should match pre-probe {pre_probe_window}"
    );
}

#[test]
fn copa_auto_decouples_initial_delta_from_competitive_mode() {
    let (mut c, mut t) = copa(CopaConfig::copa_auto());
    let rtt = rtt_estimator(ms(50));
    assert_eq!(c.delta(), 0.05);

    // Before slow start ends and AutoDelta produces its first estimate, a steady startup queue
    // must not falsely switch to competitive mode.
    standing_queue(&mut c, &mut t, ms(600), &rtt);
    assert!(!c.is_competitive());

    // Once slow start ends and an adaptive estimate exists, a non-draining queue triggers
    // competitive mode starting at delta_max (0.5) with 1/delta growing, and loss halving caps at
    // 0.5 rather than 0.05.
    c.slow_start = false;
    c.auto.as_mut().unwrap().force_estimate_for_test(0.1);
    run(&mut c, &mut t, ms(400), ms(70), &rtt);
    assert!(c.is_competitive());
    assert!(c.delta() < 0.5 && c.delta() > 0.1);
    c.delta = 0.2;
    c.on_congestion_event(t, t, false, false, MTU, 0, SpaceId::Data);
    assert!((c.delta() - 0.4).abs() < 1e-12);
    c.on_congestion_event(t + ms(60), t, false, false, MTU, 0, SpaceId::Data);
    assert!((c.delta() - 0.5).abs() < 1e-12);
}

fn settled_fixed_jitter_copa(t0: Instant, srtt: Duration) -> (Copa, RttEstimator) {
    let mut auto = CopaAutoConfig::default();
    auto.probe_epoch(Some(t0))
        .probe_timing(Duration::from_secs(20), Duration::from_secs(2))
        .jitter_floor_ms(1.0);
    let mut config = CopaConfig::copa_nc();
    config.delta(0.1).adaptive(Some(auto));
    let mut c = Copa::new(Arc::new(config), t0, MTU as u16);
    let rtt = rtt_estimator(srtt);
    c.slow_start = false;
    c.drain.set_trusted_min(ms(40));
    // Anchor slot 0 as evaluated so slot 1 (20 s) is the first RefreshRttMin trigger
    c.drain.mark_slot_evaluated(0);
    c.window = (60 * MTU) as f64;
    (c, rtt)
}

#[test]
fn drain_smooth_entry_holds_window_while_lowering_pacing() {
    let t0 = Instant::now();
    let (mut c, rtt) = settled_fixed_jitter_copa(t0, ms(50));
    // Prime 50 ms RTT_standing at 60 * MTU window before slot 1 drift
    ack(&mut c, t0 + Duration::from_secs(9) + ms(900), ms(40), &rtt);
    let mut t = t0 + Duration::from_secs(19) + ms(900);
    // Feed > 10 s drifted samples so rtt_min > trusted_min (40 ms) + 1 ms before slot 1 opens
    for _ in 0..4 {
        t += ms(15);
        ack(&mut c, t, ms(48), &rtt);
    }
    assert!(!c.drain.is_draining());
    let pre_window = c.window();
    let pre_rate = c.pacing_rate().unwrap();

    // First ACK inside slot 1 (20 s) triggers RefreshRttMin drain
    let drain_start = t0 + Duration::from_secs(20) + ms(10);
    ack(&mut c, drain_start, ms(48), &rtt);
    assert_eq!(c.drain.active_kind(), Some(DrainKind::RefreshRttMin));
    // Smooth entry: at dt = 0, cwnd remains at pre-drain window while pacing rate drops gently
    assert_eq!(c.window(), pre_window);
    assert!(c.pacing_rate().unwrap() < pre_rate);
    assert!(c.pacing_rate().unwrap() > pre_rate * 3 / 10);

    // Midway (~0.5 * srtt) and end (>= 1 * srtt) of the ramp glide smoothly toward ~0.9 * BDP (~45 * MTU)
    ack(&mut c, drain_start + ms(25), ms(48), &rtt);
    let mid_window = c.window();
    assert!(mid_window < pre_window && mid_window > 40 * MTU);
    ack(&mut c, drain_start + ms(50), ms(48), &rtt);
    assert!(c.window() <= mid_window && c.window() >= 40 * MTU);
}

#[test]
fn drain_exits_early_on_baseline_or_plateau() {
    // 1. Baseline return after >= 1 * srtt when queue empties back to ~40 ms
    let t0 = Instant::now();
    let (mut c, rtt) = settled_fixed_jitter_copa(t0, ms(40));
    ack(&mut c, t0 + Duration::from_secs(20) + ms(10), ms(46), &rtt);
    assert_eq!(c.drain.active_kind(), Some(DrainKind::RefreshRttMin));
    // Halfway through 1 * srtt (20 ms), standing drops to 40 ms but >= 1 * srtt hasn't elapsed yet
    ack(&mut c, t0 + Duration::from_secs(20) + ms(30), ms(40), &rtt);
    assert!(c.drain.is_draining());
    // At >= 1 * srtt (40 ms elapsed) with standing near trusted (40 ms), drain exits immediately
    ack(&mut c, t0 + Duration::from_secs(20) + ms(55), ms(40), &rtt);
    assert!(!c.drain.is_draining());
    assert_eq!(c.drain.trusted_min(), Some(ms(40)));

    // 2. Plateau convergence when physical propagation delay rises (40 ms -> 65 ms)
    let t0 = Instant::now();
    let (mut c, rtt) = settled_fixed_jitter_copa(t0, ms(40));
    let mut t = t0 + Duration::from_secs(20) + ms(10);
    // Keep RTT at 65 ms across rounds (>= 2 * srtt and 2 flat rounds) well before the 2 s deadline
    for _ in 0..10 {
        ack(&mut c, t, ms(65), &rtt);
        t += ms(45);
        if !c.drain.is_draining() {
            break;
        }
    }
    assert!(!c.drain.is_draining());
    assert!(t.saturating_duration_since(t0 + Duration::from_secs(20)) < ms(450));
    assert_eq!(c.drain.trusted_min(), Some(ms(65)));
}

#[test]
fn non_competitive_drain_ignores_random_loss_but_respects_persistent_congestion() {
    let t0 = Instant::now();
    let (mut c, rtt) = settled_fixed_jitter_copa(t0, ms(40));
    let start = t0 + Duration::from_secs(20) + ms(10);
    ack(&mut c, start, ms(46), &rtt);
    let pre_window = 60 * MTU;
    assert!(c.drain.is_draining());

    // Random non-persistent loss in copa-auto-nc must NOT cancel recovery
    c.on_congestion_event(start + ms(20), start, false, false, MTU, 1, SpaceId::Data);
    ack(&mut c, start + ms(55), ms(46), &rtt);
    ack(&mut c, start + ms(100), ms(40), &rtt);
    assert!(!c.drain.is_draining());
    assert!(c.window() >= pre_window);

    // Persistent congestion MUST mark congested and skip restoring up to pre_window
    let t0 = Instant::now();
    let (mut c, rtt) = settled_fixed_jitter_copa(t0, ms(40));
    let start = t0 + Duration::from_secs(20) + ms(10);
    ack(&mut c, start, ms(50), &rtt);
    c.window = (20 * MTU) as f64;
    c.on_congestion_event(start + ms(20), start, true, false, MTU, 2, SpaceId::Data);
    ack(&mut c, start + ms(60), ms(50), &rtt);
    ack(&mut c, start + ms(110), ms(50), &rtt);
    ack(&mut c, start + ms(160), ms(50), &rtt);
    ack(&mut c, start + ms(210), ms(50), &rtt);
    assert!(!c.drain.is_draining());
    assert!(c.window() < pre_window);
}

#[test]
fn refresh_drain_backs_off_when_unhelpful_and_resets_on_improvement() {
    let t0 = Instant::now();
    let (mut c, rtt) = settled_fixed_jitter_copa(t0, ms(40));

    // Slot 1 (20 s): RTT stays flat at 45 ms (no improvement over rtt_min_at_start = 45 ms) -> backoff 2 slots (next eligible: slot 3)
    ack(&mut c, t0 + Duration::from_secs(20) + ms(10), ms(45), &rtt);
    assert_eq!(c.drains, (1, 0));
    for k in 1..=6 {
        ack(
            &mut c,
            t0 + Duration::from_secs(20) + ms(10 + k * 45),
            ms(45),
            &rtt,
        );
    }
    assert!(!c.drain.is_draining());
    // Force trusted_min back to 40 ms to keep drift suspicion true cleanly
    c.drain.set_trusted_min(ms(40));

    // Slot 2 (40 s): skipped by slot-multiple backoff
    ack(&mut c, t0 + Duration::from_secs(40) + ms(10), ms(45), &rtt);
    assert!(!c.drain.is_draining());
    assert_eq!(c.drains, (1, 1));

    // Slot 3 (60 s): eligible; again no improvement -> backoff becomes 4 slots (next eligible: slot 7)
    ack(&mut c, t0 + Duration::from_secs(60) + ms(10), ms(45), &rtt);
    assert!(c.drain.is_draining());
    assert_eq!(c.drains, (2, 1));
    for k in 1..=6 {
        ack(
            &mut c,
            t0 + Duration::from_secs(60) + ms(10 + k * 45),
            ms(45),
            &rtt,
        );
    }
    assert!(!c.drain.is_draining());
    c.drain.set_trusted_min(ms(40));

    // Slots 4, 5, 6 skipped; slot 7 drains and lowers RTT_min (45 ms -> 40 ms) -> resets backoff to 1
    for slot in [4u64, 5, 6] {
        ack(
            &mut c,
            t0 + Duration::from_secs(slot * 20) + ms(10),
            ms(45),
            &rtt,
        );
        assert!(!c.drain.is_draining());
    }
    assert_eq!(c.drains, (2, 4));

    ack(&mut c, t0 + Duration::from_secs(140) + ms(10), ms(45), &rtt);
    assert_eq!(c.drains, (3, 4));
    // Lower RTT sample during drain -> improvement resets backoff_slots to 1
    ack(&mut c, t0 + Duration::from_secs(140) + ms(60), ms(40), &rtt);
    assert!(!c.drain.is_draining());
    // Next slot 8 (160 s) is immediately eligible when drift recurs
    ack(&mut c, t0 + Duration::from_secs(160) + ms(10), ms(45), &rtt);
    assert_eq!(c.drains, (4, 4));
}

#[test]
fn on_sent_expires_drain_and_recovery_timeouts() {
    let t0 = Instant::now();
    let (mut c, rtt) = settled_fixed_jitter_copa(t0, ms(40));
    let start = t0 + Duration::from_secs(20) + ms(10);
    ack(&mut c, start, ms(45), &rtt);
    assert!(c.drain.is_draining());

    // Even if no ACKs arrive for > max_duration, on_sent expires Draining -> Recovering -> Normal
    c.on_sent(start + Duration::from_secs(3), MTU, 10);
    assert!(!c.drain.is_draining());
    assert!(c.drain.is_recovering());

    c.on_sent(start + Duration::from_secs(4), MTU, 11);
    assert!(c.drain.is_normal());
}
