"""CueScheduler (pure, fake clock) and CueRunner (sending, fire log, timer thread)."""
from __future__ import annotations

import dataclasses
import json
import random
import threading
import time
from collections import Counter

import pytest

from lyricprompter_oscsender import scheduler as scheduler_module
from lyricprompter_oscsender import settings as settings_module
from lyricprompter_oscsender.cues import Cue
from lyricprompter_oscsender.osc import SendResult
from lyricprompter_oscsender.scheduler import (
    MOVING,
    SKIP_POLICIES,
    CueRunner,
    CueScheduler,
    FireRecord,
    SchedulerParams,
    TrackPoint,
)

FRAME = 0.04


def cue(cid: int, t: float, address: str = "/go", target: str | None = None, args=()) -> Cue:
    return Cue(id=cid, time=t, address=address, args=tuple(args), target=target, message=address)


def lat_map(m: dict) -> callable:
    return lambda c: m.get(c.target, 0.0)


def zero(c: Cue) -> float:
    return 0.0


def tp(ref: float, t: float, rate: float = 1.0, status: str = "tracking", hold_from=None,
       settling: bool = False) -> TrackPoint:
    return TrackPoint(ref, rate, status, t, hold_from, settling)


def drain(s: CueScheduler, now: float, until: float) -> list:
    """Fire every cue at its exact next_due_time up to `until` (like the offline simulator)."""
    out = []
    for _ in range(100_000):
        nd = s.next_due_time()
        if nd is None or nd > until:
            return out
        fires = s.due(max(nd, now))
        assert fires, "next_due_time() promised a cue that due() did not fire"
        out += fires
    raise AssertionError("drain did not finish")


def advance(s: CueScheduler, *, t0: float, ref0: float, seconds: float, rate: float = 1.0,
            start: bool = False, **kw) -> list:
    """Tracker updates every frame at a constant rate from (t0, ref0); fires between updates."""
    fires = []
    for k in range(round(seconds / FRAME) + 1):
        t = t0 + k * FRAME
        pt = tp(ref0 + rate * (t - t0), t, rate, **kw)
        fires += s.start(pt) if (start and k == 0) else s.update(pt)
        fires += drain(s, t, t + FRAME)
    return fires


def ids(fires) -> list[int]:
    return [f.cue.id for f in fires]


# --------------------------------------------------------------------------------- steady tracking


@pytest.mark.parametrize("rate", [1.0, 0.9])
def test_steady_tracking_fires_each_cue_once_at_its_subframe_time(rate):
    times = [0.0137, 0.5, 1.0271, 1.5555, 2.2222, 3.0005, 3.9999]
    s = CueScheduler([cue(i + 1, t) for i, t in enumerate(times)], zero)
    fires = advance(s, t0=1000.0, ref0=0.0, seconds=5.0, rate=rate, start=True)
    assert ids(fires) == list(range(1, len(times) + 1))
    for f in fires:
        assert abs(f.at - (1000.0 + f.cue.time / rate)) < 1e-3
        assert f.reason == "track" and f.send
        assert f.ref_late < 1e-6
        assert f.ref_time == pytest.approx(f.cue.time, abs=1e-6)
    assert s.fired_ids() == set(range(1, len(times) + 1))
    assert s.next_due_time() is None


@pytest.mark.parametrize("rate", [1.0, 0.9])
def test_latency_lookahead_is_rate_times_latency(rate):
    cues = [cue(1, 2.0, "/a"), cue(2, 2.0, "/v", "video"), cue(3, 3.0, "/l", "lights")]
    s = CueScheduler(cues, lat_map({"video": 0.1, "lights": 0.25}))
    fires = advance(s, t0=1000.0, ref0=0.0, seconds=4.0, rate=rate, start=True)
    assert ids(fires) == [2, 1, 3]
    by_id = {f.cue.id: f for f in fires}
    base = 1000.0 + 2.0 / rate
    assert abs(by_id[1].at - base) < 1e-3
    # sent L live seconds early at any rate: the reference look-ahead is rate * L
    assert abs((base - by_id[2].at) - 0.1) < 1e-3
    assert by_id[2].ref_time == pytest.approx(2.0 - rate * 0.1, abs=1e-6)
    assert abs(by_id[3].at - (1000.0 + 3.0 / rate - 0.25)) < 1e-3
    assert all(f.reason == "track" and f.ref_late < 1e-6 for f in fires)


def test_extrapolation_is_capped_when_updates_stop():
    s = CueScheduler([cue(1, 10.05), cue(2, 10.15), cue(3, 10.18, target="v")], lat_map({"v": 0.1}))
    assert s.start(tp(10.0, 0.0)) == []
    assert s.position(0.5) == pytest.approx(10.1)
    fires = drain(s, 0.0, 10.0)
    # 10.05 at its time; 10.18 (look-ahead 0.1) when the position reaches 10.08; 10.15 lies beyond the cap
    assert [(f.cue.id, round(f.at, 6)) for f in fires] == [(1, 0.05), (3, 0.08)]
    assert s.next_due_time() is None
    assert s.due(100.0) == []
    assert 2 not in s.fired_ids()
    assert s.update(tp(10.2, 0.6)) == []          # the next update moves past it: fired, late
    late = s.due(0.6)
    assert ids(late) == [2] and late[0].reason == "catchup"
    assert late[0].ref_late == pytest.approx(0.05)


@pytest.mark.parametrize("status,rate", [("hold", 1.0), ("waiting", 1.0), ("stopped", 1.0),
                                         ("tracking", 0.0), ("coasting", 0.0), ("tracking", float("nan"))])
def test_not_moving_fires_nothing_new_and_looks_nowhere_ahead(status, rate):
    s = CueScheduler([cue(1, 10.0), cue(2, 10.05, target="v"), cue(3, 10.5)], lat_map({"v": 0.1}))
    s.start(tp(10.0, 0.0, rate, status))
    assert ids(s.due(0.0)) == [1]                 # at the position
    assert s.next_due_time() is None
    assert s.due(100.0) == []
    assert s.position(100.0) == 10.0
    assert [(c.id, eta) for c, eta in s.upcoming(1.0)] == [(2, None), (3, None)]


def test_switching_to_hold_stops_extrapolation_and_lookahead():
    s = CueScheduler([cue(1, 9.12, target="v"), cue(2, 9.3)], lat_map({"v": 0.1}))
    s.start(tp(9.0, 0.0))
    assert drain(s, 0.0, 0.0) == []
    assert s.next_due_time() == pytest.approx(0.02)
    s.update(tp(9.0, 0.01, 0.0, "hold"))
    assert s.next_due_time() is None
    assert s.due(5.0) == []
    assert s.position(5.0) == 9.0
    assert MOVING == ("tracking", "coasting")


# --------------------------------------------------------------------------------- position changes


def test_small_forward_correction_fires_crossed_cues_once_as_catchup_in_time_id_order():
    cues = [cue(1, 1.0), cue(5, 1.2, "/b"), cue(3, 1.2, "/a"), cue(4, 1.5), cue(6, 2.5)]
    s = CueScheduler(cues, zero)
    assert advance(s, t0=0.0, ref0=0.0, seconds=0.8, start=True) == []
    assert s.update(tp(1.64, 0.84)) == []         # +0.8 s against the extrapolation: <= catchup_s
    fires = s.due(0.84)
    assert ids(fires) == [1, 3, 5, 4]
    assert all(f.reason == "catchup" and f.send and f.at == 0.84 for f in fires)
    assert fires[0].ref_late == pytest.approx(0.64)
    later = advance(s, t0=0.88, ref0=1.68, seconds=2.0)
    assert ids(later) == [6] and later[0].reason == "track"


JUMP_CUES = [cue(1, 5.0, "/a"), cue(2, 6.0, "/a"), cue(3, 7.0, "/a", "x"), cue(4, 8.0, "/b"),
             cue(5, 9.0, "/a", "x"), cue(6, 9.5, "/a"), cue(7, 30.0, "/a")]


@pytest.mark.parametrize("policy,chased", [("skip", []), ("latest", [4, 5, 6]), ("all", [1, 2, 3, 4, 5, 6])])
def test_jump_forward_applies_the_skip_policy(policy, chased):
    s = CueScheduler(JUMP_CUES, zero, SchedulerParams(skip_policy=policy))
    s.start(tp(0.0, 0.0))
    fires = s.update(tp(20.0, 0.04))
    skipped = [i for i in range(1, 7) if i not in chased]
    assert ids(fires) == chased + skipped               # sent first, each group in time order
    assert all(f.send and f.reason == "chase" for f in fires[:len(chased)])
    assert all(not f.send and f.reason == "skipped" for f in fires[len(chased):])
    assert all(f.at == 0.04 and f.ref_time == 20.0 for f in fires)
    assert all(f.ref_late == pytest.approx(20.0 - f.cue.time) for f in fires)
    assert s.due(0.04) == []
    assert s.fired_ids() == set(range(1, 7))
    assert ids(advance(s, t0=0.08, ref0=20.04, seconds=10.5)) == [7]


def test_jump_threshold_is_catchup_s():
    s = CueScheduler([cue(1, 1.0), cue(2, 2.0)], zero, SchedulerParams(catchup_s=0.5, skip_policy="skip"))
    s.start(tp(0.0, 0.0))
    fires = s.update(tp(1.6, 0.0))                     # +1.6 > 0.5: a jump
    assert [(f.cue.id, f.reason) for f in fires] == [(1, "skipped")]
    assert s.due(0.0) == []


def test_latest_groups_default_target_cues_with_that_target():
    s = CueScheduler(JUMP_CUES, zero, SchedulerParams(skip_policy="latest"))
    s.set_default_target("x")
    s.start(tp(0.0, 0.0))
    assert [f.cue.id for f in s.update(tp(20.0, 0.04)) if f.send] == [4, 6]


def test_jump_back_rearms_cues_after_the_new_position():
    s = CueScheduler([cue(1, 5.0), cue(2, 10.0), cue(3, 10.5)], zero)
    assert ids(advance(s, t0=0.0, ref0=0.0, seconds=12.0, start=True)) == [1, 2, 3]
    assert s.update(tp(8.0, 12.08)) == []
    assert s.fired_ids() == {1}
    fires = advance(s, t0=12.12, ref0=8.04, seconds=3.0)
    assert ids(fires) == [2, 3]
    assert all(f.reason == "track" for f in fires)
    assert fires[0].at == pytest.approx(12.12 + (10.0 - 8.04))


def test_rearm_threshold_is_inclusive():
    s = CueScheduler([cue(1, 10.5)], zero)
    s.start(tp(11.0, 5.0, 0.0, "hold"))                # policy "latest" chases it at start
    assert 1 in s.fired_ids()
    s.update(tp(10.01, 5.0, 0.0, "hold"))              # back 0.99 s: no new pass
    assert 1 in s.fired_ids()
    s.update(tp(11.0, 5.0, 0.0, "hold"))               # forward 0.99: nothing to fire
    s.update(tp(10.0, 5.0, 0.0, "hold"))               # back exactly rearm_back_s: re-armed
    assert s.fired_ids() == set()


GLITCH_CUES = [cue(1, 16.0, "/a"), cue(2, 17.0, "/b"), cue(3, 18.0, "/a"), cue(4, 19.5, "/c", "v"),
               cue(5, 20.05, "/d", "v"), cue(6, 22.0, "/a")]


def glitch_pass(policy: str = "latest", **kw) -> CueScheduler:
    """Cues 1-5 fired by 20.0 (5 early by look-ahead), then the tracker jumps back 5 s (a wrong re-localisation)."""
    s = CueScheduler(GLITCH_CUES, lat_map({"v": 0.1}), SchedulerParams(skip_policy=policy, **kw))
    assert ids(advance(s, t0=0.0, ref0=10.0, seconds=10.0, start=True)) == [1, 2, 3, 4, 5]
    assert s.update(tp(15.0, 10.04)) == []
    assert s.fired_ids() == set()
    return s


@pytest.mark.parametrize("policy", SKIP_POLICIES)
def test_a_tracker_glitch_back_and_forth_resends_nothing(policy):
    s = glitch_pass(policy)
    assert drain(s, 10.04, 10.08) == []
    assert s.update(tp(20.08, 10.08)) == []           # back where the band is: cues 1-5 were already sent
    assert s.due(10.08) == []
    assert s.fired_ids() == {1, 2, 3, 4, 5}
    rest = advance(s, t0=10.12, ref0=20.12, seconds=3.0)
    assert ids(rest) == [6] and rest[0].reason == "track"


@pytest.mark.parametrize("policy", SKIP_POLICIES)
def test_a_glitch_back_and_forth_within_a_wide_catchup_window_resends_nothing(policy):
    s = glitch_pass(policy, catchup_s=10.0)            # the return (+5 s) is a catch-up move, not a jump
    assert s.update(tp(20.08, 10.08)) == []
    assert s.due(10.08) == []
    assert ids(advance(s, t0=10.12, ref0=20.12, seconds=3.0)) == [6]


def test_a_glitch_return_past_the_reach_of_the_pass_chases_only_the_new_cues():
    s = glitch_pass("all")
    fires = s.update(tp(22.5, 10.08))                  # lands beyond cue 6, which was never sent
    assert [(f.cue.id, f.reason) for f in fires] == [(6, "chase")]
    assert s.fired_ids() == {1, 2, 3, 4, 5, 6}


def test_a_real_repeat_after_a_jump_back_fires_the_cues_again():
    s = glitch_pass("all")
    again = advance(s, t0=10.08, ref0=15.04, seconds=4.5)    # the band really went back: plays on from 15
    assert ids(again) == [1, 2, 3, 4] and {f.reason for f in again} == {"track"}
    # then it skips ahead over cue 5 (sent in the first pass) and past cue 6 (never sent)
    fires = s.update(tp(23.0, 14.62))
    assert [(f.cue.id, f.reason) for f in fires] == [(6, "chase")]
    assert s.fired_ids() == {1, 2, 3, 4, 5, 6}


@pytest.mark.parametrize("policy,chased", [("skip", []), ("latest", [2, 3, 4, 5]), ("all", [1, 2, 3, 4, 5])])
def test_an_operator_jump_after_a_glitch_back_applies_the_policy(policy, chased):
    s = glitch_pass(policy)
    fires = s.jump(tp(20.08, 10.08))                   # the operator wants a new pass from here
    assert [f.cue.id for f in fires if f.send] == chased
    assert sorted(ids(fires)) == [1, 2, 3, 4, 5]
    s = glitch_pass(policy)
    s.start(tp(15.0, 10.08))                           # a new start forgets the glitch too
    assert [f.cue.id for f in s.update(tp(20.08, 10.12)) if f.send] == chased


def test_set_cues_keeps_restorable_marks_by_key():
    s = glitch_pass("all")
    # lines inserted above (ids shift) and a new cue inside the glitched-over range, never sent
    shifted = [dataclasses.replace(c, id=c.id + 10) for c in GLITCH_CUES] + [cue(30, 19.0, "/new")]
    s.set_cues(shifted, 10.06)
    fires = s.update(tp(20.08, 10.08))
    assert [(f.cue.id, f.reason) for f in fires] == [(30, "chase")]


@pytest.mark.parametrize("policy", SKIP_POLICIES)
def test_a_glitch_return_resends_no_cue_the_latency_look_ahead_had_sent(policy):
    # /clip/play goes to a target with 0.5 s output latency: sent at 19.8 for 20.3, before the glitch
    cues = [cue(1, 16.0), cue(2, 20.3, "/clip/play", "video"), cue(3, 30.0)]
    s = CueScheduler(cues, lat_map({"video": 0.5}), SchedulerParams(skip_policy=policy))
    assert ids(advance(s, t0=0.0, ref0=10.0, seconds=10.0, start=True)) == [1, 2]
    assert s.update(tp(12.0, 10.04)) == []             # a wrong re-localisation back
    assert s.update(tp(20.08, 10.08)) == []            # and back: 20.3 lies inside the look-ahead again
    assert s.due(10.08) == []
    assert s.fired_ids() == {1, 2}


@pytest.mark.parametrize("landing,again", [(19.7, []), (18.5, [2])])
def test_a_glitch_return_landing_behind_where_the_pass_was(landing, again):
    s = CueScheduler([cue(1, 16.0), cue(2, 19.9), cue(3, 30.0)], zero)
    assert ids(advance(s, t0=0.0, ref0=10.0, seconds=10.0, start=True)) == [1, 2]    # the pass reaches 20.04
    assert s.update(tp(12.0, 10.04)) == []
    assert s.update(tp(landing, 12.0)) == []
    # 0.3 s behind: a settle, cue 2 was sent before the glitch; rearm_back_s or more behind: a new pass
    assert ids(advance(s, t0=12.04, ref0=landing + 0.04, seconds=2.0)) == again


@pytest.mark.parametrize("policy,sent", [("skip", []), ("latest", [2]), ("all", [2])])
def test_a_glitch_return_does_not_restore_a_cue_whose_state_the_glitch_overwrote(policy, sent):
    # the wrong segment after the glitch sends /scene 3 again: the device no longer holds what cue 2 set, so the
    # return does not mark cue 2 handled silently, it goes through the policy like any cue jumped over
    cues = [cue(1, 16.0, "/scene", args=[("i", 3)]), cue(2, 20.0, "/scene", args=[("i", 4)]), cue(3, 40.0, "/x")]
    s = CueScheduler(cues, zero, SchedulerParams(skip_policy=policy))
    assert ids(advance(s, t0=0.0, ref0=10.0, seconds=11.0, start=True)) == [1, 2]
    assert s.update(tp(15.0, 11.04)) == []             # wrong re-localisation back 6 s
    assert ids(advance(s, t0=11.08, ref0=15.04, seconds=3.0)) == [1]
    fires = s.update(tp(24.1, 14.12))                  # back where the band is
    assert ids(fires) == [2] and [f.cue.id for f in fires if f.send] == sent
    assert s.fired_ids() == {1, 2}


def test_a_glitch_return_landing_behind_a_cue_whose_state_the_glitch_overwrote_fires_it():
    cues = [cue(1, 16.0, "/scene", args=[("i", 3)]), cue(2, 19.9, "/scene", args=[("i", 4)]), cue(3, 19.95, "/x")]
    s = CueScheduler(cues, zero)
    assert ids(advance(s, t0=0.0, ref0=10.0, seconds=10.0, start=True)) == [1, 2, 3]
    assert s.update(tp(15.0, 10.04)) == []
    assert ids(advance(s, t0=10.08, ref0=15.04, seconds=1.0)) == [1]
    assert s.update(tp(19.7, 11.12)) == []             # lands 0.3 s behind: /x is restored, /scene 4 is not
    fires = advance(s, t0=11.16, ref0=19.74, seconds=1.0)
    assert ids(fires) == [2] and fires[0].reason == "track"


def test_small_backward_settle_does_not_refire():
    cues = [cue(1, 10.0), cue(2, 10.1), cue(3, 10.35, target="v")]
    s = CueScheduler(cues, lat_map({"v": 0.2}))
    assert ids(advance(s, t0=0.0, ref0=0.0, seconds=10.2, start=True)) == [1, 2, 3]   # 3 by look-ahead
    assert s.update(tp(9.94, 10.24)) == []             # settles back 0.3 s
    assert advance(s, t0=10.28, ref0=9.98, seconds=2.0) == []
    assert s.fired_ids() == {1, 2, 3}


COAST_CUES = [cue(1, 8.0), cue(2, 9.0), cue(3, 9.5), cue(4, 10.0), cue(5, 10.5)]


def coasted_pass() -> CueScheduler:
    """Vocal mode: the voice is followed to 8.52, then the singer pauses and the tracker coasts on to 10.16;
    cues 2-4 fire on the coasted position."""
    s = CueScheduler(COAST_CUES, zero)
    assert ids(advance(s, t0=0.0, ref0=0.0, seconds=8.48, start=True)) == [1]
    assert ids(advance(s, t0=8.52, ref0=8.52, seconds=1.6, status="coasting")) == [2, 3, 4]
    return s


def test_a_relock_settling_back_over_coasted_positions_refires_nothing():
    s = coasted_pass()
    # the voice comes back where it stopped: the relock pulls the position back 1.5 s, but only over positions
    # coasting reached (never further than rearm_back_s behind the voice): a settle, not a new pass
    assert s.update(tp(8.64, 10.2)) == []
    assert s.fired_ids() == {1, 2, 3, 4}
    later = advance(s, t0=10.24, ref0=8.68, seconds=2.5)
    assert ids(later) == [5] and later[0].reason == "track"


def test_a_move_back_behind_the_followed_position_after_coasting_is_a_new_pass():
    s = coasted_pass()
    assert s.update(tp(7.5, 10.2)) == []               # 1.02 s behind where the voice was followed
    assert s.fired_ids() == set()
    assert ids(advance(s, t0=10.24, ref0=7.54, seconds=3.5)) == [1, 2, 3, 4, 5]


def test_a_provisional_relock_running_on_before_it_settles_back_refires_nothing():
    s = coasted_pass()
    # the voice is back: the provisional relock runs on along the coasting path (cue 5 fires), then settles back
    # 2 s; its points say `settling`, so the move back is still measured from where the voice was followed
    assert ids(advance(s, t0=10.16, ref0=10.16, seconds=0.4, settling=True)) == [5]
    assert s.update(tp(8.64, 10.64, settling=True)) == []
    assert advance(s, t0=10.68, ref0=8.68, seconds=2.5) == []
    assert s.fired_ids() == {1, 2, 3, 4, 5}
    # followed again: a move back of rearm_back_s or more is a new pass as usual
    assert s.update(tp(9.8, 13.22)) == []
    assert s.fired_ids() == {1, 2, 3}
    # the vocal guard (waiting for the voice) does not follow the music either
    s = CueScheduler(COAST_CUES, zero)
    assert ids(advance(s, t0=0.0, ref0=0.0, seconds=8.48, start=True)) == [1]
    assert ids(advance(s, t0=8.52, ref0=8.52, seconds=1.6, hold_from=12.9)) == [2, 3, 4]
    assert s.update(tp(8.64, 10.2)) == [] and s.fired_ids() == {1, 2, 3, 4}


def test_operator_jump_to_a_cue_time_fires_that_cue():
    s = CueScheduler([cue(1, 2.0, "/a"), cue(2, 10.0, "/b"), cue(3, 12.0, "/c")], zero)
    assert ids(advance(s, t0=0.0, ref0=0.0, seconds=3.0, start=True)) == [1]
    assert s.jump(tp(10.0, 3.1)) == []                 # cue 2 sits at the new position: not jumped over
    fires = s.due(3.1)
    assert ids(fires) == [2] and fires[0].reason == "track" and fires[0].ref_late == 0.0
    assert s.jump(tp(2.0, 3.2)) == []                  # back to cue 1: re-armed, fires again
    assert ids(s.due(3.2)) == [1]
    assert s.fired_ids() == {1}
    # forward to cue 3 while holding: cue 2 (re-armed by the jump back) is jumped over and chased
    assert [(f.cue.id, f.reason) for f in s.jump(tp(12.0, 3.3, 0.0, "hold"))] == [(2, "chase")]
    assert ids(s.due(3.3)) == [3]                      # at the new position, no look-ahead needed


def test_operator_jump_forward_applies_the_policy():
    s = CueScheduler(JUMP_CUES, zero, SchedulerParams(skip_policy="latest"))
    s.start(tp(0.0, 0.0))
    fires = s.jump(tp(9.5, 0.04))
    assert [(f.cue.id, f.reason) for f in fires] == [(2, "chase"), (4, "chase"), (5, "chase"),
                                                     (1, "skipped"), (3, "skipped")]
    assert ids(s.due(0.04)) == [6]                     # at the new position


def test_start_and_jump_points_ahead_of_their_boundary_fire_the_cues_in_between_at_once():
    # the tracker publishes the start / jump target plus its input-latency look-ahead (0.4 s here, more than
    # catchup_s): the cues from the requested time to that position fire at once, none is jumped over, and the
    # next update is no jump
    cues = [cue(1, 4.0, "/a"), cue(2, 5.0, "/b"), cue(3, 5.15, "/c"), cue(4, 5.3, "/d"), cue(5, 5.6, "/e"),
            cue(6, 8.0, "/f"), cue(7, 10.0, "/g"), cue(8, 10.2, "/h"), cue(9, 10.6, "/i")]
    s = CueScheduler(cues, zero, SchedulerParams(catchup_s=0.3, skip_policy="skip"))
    assert [(f.cue.id, f.reason, f.send) for f in s.start(tp(5.4, 0.0), boundary=5.0)] == [(1, "skipped", False)]
    fires = s.due(0.0)
    assert [(f.cue.id, f.reason) for f in fires] == [(2, "track"), (3, "track"), (4, "track")]
    assert fires[0].ref_late == pytest.approx(0.4)
    assert ids(advance(s, t0=0.04, ref0=5.44, seconds=1.0)) == [5]
    fires = s.jump(tp(10.4, 1.08), boundary=10.0)     # Go to cue 7
    assert [(f.cue.id, f.reason, f.send) for f in fires] == [(6, "skipped", False)]
    assert [(f.cue.id, f.reason) for f in s.due(1.08)] == [(7, "track"), (8, "track")]
    assert ids(advance(s, t0=1.12, ref0=10.44, seconds=1.0)) == [9]
    # a boundary ahead of the point (a negative input latency): the point moves no further back than that
    s.jump(tp(9.7, 2.2), boundary=10.0)
    assert s.position(2.2) == 10.0 and ids(s.due(2.2)) == [7]
    assert s.update(tp(9.74, 2.24)) == [] and s.due(2.24) == []
    with pytest.raises(ValueError):
        s.jump(tp(9.7, 2.3), boundary=float("nan"))


def test_operator_jump_forward_below_catchup_still_applies_the_policy():
    s = CueScheduler([cue(1, 0.3), cue(2, 0.6)], zero, SchedulerParams(skip_policy="skip"))
    s.start(tp(0.0, 0.0))
    assert [(f.cue.id, f.reason, f.send) for f in s.jump(tp(0.5, 0.0))] == [(1, "skipped", False)]


def test_start_at_zero_fires_a_cue_at_zero_immediately():
    s = CueScheduler([cue(1, 0.0), cue(2, 1.0)], zero)
    assert s.start(tp(0.0, 5.0, 0.0, "waiting")) == []
    assert s.next_due_time() == 5.0
    fires = s.due(5.0)
    assert ids(fires) == [1] and fires[0].reason == "track" and fires[0].at == 5.0


def test_start_mid_song_applies_the_policy_with_reason_start():
    s = CueScheduler(JUMP_CUES, zero, SchedulerParams(skip_policy="latest"))
    fires = s.start(tp(9.5, 0.0))
    assert [(f.cue.id, f.reason, f.send) for f in fires] == [
        (2, "start", True), (4, "start", True), (5, "start", True), (1, "skipped", False), (3, "skipped", False)]
    assert ids(s.due(0.0)) == [6]
    assert s.start(tp(0.0, 1.0)) == []                 # a new start re-arms everything
    assert s.fired_ids() == set()


def test_update_before_start_starts():
    s = CueScheduler([cue(1, 1.0), cue(2, 3.0)], zero, SchedulerParams(skip_policy="all"))
    assert [(f.cue.id, f.reason) for f in s.update(tp(2.0, 0.0))] == [(1, "start")]


def test_hold_from_guard_holds_cues_until_lifted():
    cues = [cue(1, 4.0), cue(2, 5.0), cue(3, 6.0), cue(4, 4.95, target="v")]
    s = CueScheduler(cues, lat_map({"v": 0.1}))
    fires = advance(s, t0=0.0, ref0=0.0, seconds=7.0, start=True, hold_from=5.0)
    assert ids(fires) == [1, 4]                        # 4.95 < hold_from; 5.0 and 6.0 are held
    assert s.next_due_time() is None
    assert [(c.id, eta) for c, eta in s.upcoming(7.0)] == [(2, None), (3, None)]
    s.update(tp(7.04, 7.04))                           # guard lifted
    late = s.due(7.04)
    assert ids(late) == [2, 3] and all(f.reason == "catchup" for f in late)


@pytest.mark.parametrize("policy,sent", [("skip", []), ("latest", [2]), ("all", [1, 2])])
def test_a_jump_over_cues_the_guard_held_behind_the_position_applies_the_policy_to_them(policy, sent):
    # vocal mode waiting at the phrase onset at 5.0: the cue just before it waits for the voice
    cues = [cue(1, 4.98, "/scene", args=[("s", "A")]), cue(2, 8.0, "/scene", args=[("s", "B")])]
    s = CueScheduler(cues, zero, SchedulerParams(skip_policy=policy))
    assert advance(s, t0=0.0, ref0=4.0, seconds=1.0, start=True, hold_from=4.94) == []
    fires = s.update(tp(9.0, 1.04))                    # the voice comes in 4 s later in the song
    assert [f.cue.id for f in fires if f.send] == sent
    assert sorted(ids(fires)) == [1, 2]
    assert s.due(1.04) == []                           # nothing fires late after the jump's records


# --------------------------------------------------------------------------------- edits, order, views


def test_set_cues_carries_fired_state_and_never_fires_new_cues_behind_the_position():
    s = CueScheduler([cue(1, 1.0, "/a"), cue(2, 3.0, "/b"), cue(3, 7.0, "/c")], zero)
    assert ids(advance(s, t0=0.0, ref0=0.0, seconds=5.0, start=True)) == [1, 2]
    # a line was inserted above (ids shift), one cue added behind and one ahead of the position,
    # and the fired cue /b moved ahead (a new cue: it fires at its new time)
    new = [cue(2, 1.0, "/a"), cue(3, 2.0, "/behind"), cue(5, 6.0, "/ahead"), cue(6, 6.5, "/b"),
           cue(8, 7.0, "/c")]
    s.set_cues(new, 5.0)
    assert s.fired_ids() == {2, 3}
    assert s.due(5.0) == []
    assert ids(advance(s, t0=5.04, ref0=5.04, seconds=3.0)) == [5, 6, 8]


def test_set_cues_carries_duplicates_one_by_one():
    s = CueScheduler([cue(1, 1.0), cue(2, 1.0), cue(3, 5.0)], zero)
    advance(s, t0=0.0, ref0=0.0, seconds=2.0, start=True)
    s.set_cues([cue(1, 1.0), cue(2, 1.0), cue(3, 1.0), cue(4, 5.0)], 2.0)
    assert s.fired_ids() == {1, 2, 3}                  # the third copy is new and behind the position


def test_set_cues_before_start_keeps_everything_armed():
    s = CueScheduler([], zero)
    s.set_cues([cue(1, 0.0), cue(2, 1.0)], 100.0)
    assert s.fired_ids() == set()
    s.start(tp(0.0, 0.0))
    assert ids(s.due(0.0)) == [1]


def test_equal_time_cues_fire_in_id_order():
    s = CueScheduler([cue(7, 2.0, "/c"), cue(3, 2.0, "/a"), cue(5, 2.0, "/b")], zero)
    fires = advance(s, t0=0.0, ref0=0.0, seconds=3.0, start=True)
    assert ids(fires) == [3, 5, 7]
    assert len({f.at for f in fires}) == 1


def test_equal_time_cues_on_latencies_equal_to_the_microsecond_fire_together_in_id_order():
    # 100 ms reached by different arithmetic (0.3 - 0.2 == 0.09999999999999998): one lane, not two
    cues = [cue(1, 0.5, "/x", "b"), cue(2, 0.5, "/y", "a"), cue(3, 0.5, "/z", "n")]
    s = CueScheduler(cues, lat_map({"b": 0.1, "a": 0.3 - 0.2, "n": 0.07 + 0.03}))
    fires = advance(s, t0=0.0, ref0=0.0, seconds=1.0, start=True)
    assert ids(fires) == [1, 2, 3] and len({f.at for f in fires}) == 1
    s.set_latency_of(lat_map({"b": 0.2, "a": 0.4 - 0.2, "n": 0.2}))
    s.jump(tp(0.0, 2.0))
    assert ids(advance(s, t0=2.04, ref0=0.04, seconds=1.0)) == [1, 2, 3]


def test_a_catchup_batch_across_latency_lanes_fires_in_time_id_order():
    cues = [cue(1, 2.0, "/v", "v"), cue(2, 1.9, "/a"), cue(3, 2.0, "/b"), cue(4, 2.0, "/l", "l")]
    s = CueScheduler(cues, lat_map({"v": 0.2, "l": 0.05}))
    s.start(tp(1.5, 0.0))
    s.update(tp(2.3, 0.0))                             # +0.8: catch-up, fired by due
    fires = s.due(0.0)
    assert ids(fires) == [2, 1, 3, 4] and {f.reason for f in fires} == {"catchup"}


def test_upcoming_etas():
    cues = [cue(1, 1.0), cue(2, 2.0, target="v"), cue(3, 3.0), cue(4, 4.0)]
    s = CueScheduler(cues, lat_map({"v": 0.5}))
    assert [(c.id, eta) for c, eta in s.upcoming(0.0)] == [(1, None), (2, None), (3, None)]
    s.start(tp(0.0, 100.0, rate=0.5))
    up = s.upcoming(100.0)
    assert [c.id for c, _ in up] == [1, 2, 3]
    assert [eta for _, eta in up] == pytest.approx([2.0, 3.5, 6.0])
    assert [eta for _, eta in s.upcoming(100.04)] == pytest.approx([1.96, 3.46, 5.96])
    # no update for a while: the position stops at the extrapolation cap, and so do the ETAs
    assert [eta for _, eta in s.upcoming(150.0)] == pytest.approx([1.9, 3.4, 5.9])
    assert len(s.upcoming(100.0, n=10)) == 4 and s.upcoming(100.0, n=0) == []
    s.update(tp(1.0, 102.0, rate=0.5))
    assert s.upcoming(102.0, n=1)[0][1] == 0.0         # due, not fired yet
    s.due(102.0)
    assert [c.id for c, _ in s.upcoming(102.0)] == [2, 3, 4]


def test_params_validation_and_set_params():
    with pytest.raises(ValueError):
        SchedulerParams(skip_policy="nope").validated()
    with pytest.raises(ValueError):
        CueScheduler([], zero, SchedulerParams(catchup_s=-1.0))
    with pytest.raises(ValueError):
        CueScheduler([], zero, SchedulerParams(max_extrap_s=float("nan")))
    assert SKIP_POLICIES == ("skip", "latest", "all")
    assert scheduler_module.SKIP_POLICIES is settings_module.SKIP_POLICIES
    s = CueScheduler([cue(1, 5.0), cue(2, 6.0)], zero)
    s.set_params(SchedulerParams(skip_policy="all"))
    s.start(tp(0.0, 0.0))
    assert [f.reason for f in s.update(tp(10.0, 0.0))] == ["chase", "chase"]


def test_bad_latency_values_and_points_are_harmless():
    def bad(c):
        if c.id == 1:
            raise KeyError("no such target")
        return {2: -0.5, 3: float("nan")}.get(c.id, 0.2)

    s = CueScheduler([cue(1, 1.0), cue(2, 1.0), cue(3, 1.0), cue(4, 1.2)], bad)
    fires = advance(s, t0=0.0, ref0=0.0, seconds=2.0, start=True)
    # lanes 0 and 0.2 reach their due times within an ulp of each other: compare as a set
    assert sorted((f.cue.id, round(f.at, 6)) for f in fires) == [(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)]
    assert s.update(tp(float("nan"), 3.0)) == []       # ignored: the last point stays current
    assert s.position(2.04) == pytest.approx(2.04)
    with pytest.raises(ValueError):
        s.start(tp(float("inf"), 0.0))
    s.set_latency_of(zero)
    s.jump(tp(0.0, 3.0))
    assert ids(advance(s, t0=3.04, ref0=0.04, seconds=1.2)) == [1, 2, 3, 4]


def test_due_and_next_due_time_scale_with_cues_near_the_position():
    n = 50_000
    cues = [cue(i + 1, i * 0.01, target="v" if i % 3 == 0 else None) for i in range(n)]
    s = CueScheduler(cues, lat_map({"v": 0.05}))
    t = time.perf_counter()
    fires = advance(s, t0=0.0, ref0=0.0, seconds=20.0, start=True)
    elapsed = time.perf_counter() - t
    got = ids(fires)
    assert len(got) == len(set(got)) >= 2000
    assert set(range(1, 2001)) <= set(got)
    assert elapsed < 1.5, f"{elapsed:.2f} s for {len(got)} fires among {n} cues"


# --------------------------------------------------------------------------------- runner


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeSend:
    def __init__(self, fail=(), boom=()):
        self.calls: list[int] = []
        self.fail, self.boom = set(fail), set(boom)

    def __call__(self, c: Cue) -> SendResult:
        self.calls.append(c.id)
        if c.id in self.boom:
            raise OSError("boom")
        if c.id in self.fail:
            return SendResult(False, c.target or "main", "unreachable")
        return SendResult(True, c.target or "main")


def make_runner(cues, clock, send, *, latency_of=zero, params=None, **kw) -> CueRunner:
    return CueRunner(CueScheduler(cues, latency_of, params), send, clock=clock, timer=False, **kw)


def step(r: CueRunner, clock: FakeClock, ref: float, t: float, **kw) -> None:
    clock.t = t
    r.feed(tp(ref, t, **kw))


def test_disarmed_runner_records_dry_runs_and_never_sends():
    clock, send = FakeClock(), FakeSend()
    r = make_runner([cue(1, 0.0), cue(2, 0.5, target="x"), cue(3, 1.0)], clock, send, armed=False,
                    live_time=lambda: 12.5)
    r.start(tp(0.0, 0.0))
    step(r, clock, 0.5, 0.5)
    recs = r.records()
    assert [(x.cue_id, x.dry_run, x.sent, x.target, x.reason) for x in recs] == [
        (1, True, False, "main", "track"), (2, True, False, "x", "track")]
    assert send.calls == []
    assert recs[0].live_time == 12.5 and [x.seq for x in recs] == [1, 2]
    assert r.counts() == {"sent": 0, "skipped": 0, "errors": 0, "dry": 2}
    r.armed = True
    assert r.armed
    step(r, clock, 1.0, 1.0)
    assert send.calls == [3] and r.records(after_seq=2)[0].sent


def test_fire_manual_sends_even_when_disarmed_and_keeps_the_pass():
    clock, send = FakeClock(), FakeSend()
    c = cue(1, 5.0, "/m")
    r = make_runner([c], clock, send, armed=False)
    r.start(tp(0.0, 0.0))
    rec = r.fire_manual(c)
    assert send.calls == [1]
    assert (rec.reason, rec.sent, rec.dry_run, rec.error, rec.seq) == ("manual", True, False, None, 1)
    r.armed = True
    for k in range(1, 140):
        step(r, clock, k * FRAME, k * FRAME)
    assert send.calls == [1, 1]                        # still fires at its time


def test_records_after_seq_limit_and_log_size():
    clock, send = FakeClock(), FakeSend()
    r = make_runner([cue(i, 0.1 * i) for i in range(1, 6)], clock, send, log_size=4)
    r.start(tp(0.0, 0.0))
    for k in range(1, 20):
        step(r, clock, k * FRAME, k * FRAME)
    assert r.last_seq == 5
    assert [x.seq for x in r.records()] == [2, 3, 4, 5]          # bounded log
    assert [x.seq for x in r.records(after_seq=3)] == [4, 5]
    assert [x.seq for x in r.records(limit=2)] == [4, 5]
    assert r.records(after_seq=5) == [] and r.records(limit=0) == []
    d = r.records()[0].to_dict()
    assert json.loads(json.dumps(d))["cue_id"] == 2


def test_send_errors_are_recorded_and_do_not_stop_later_cues():
    clock, send = FakeClock(), FakeSend(fail={1}, boom={2})
    r = make_runner([cue(1, 0.1), cue(2, 0.2), cue(3, 0.3)], clock, send)
    r.start(tp(0.0, 0.0))
    for k in range(1, 10):
        step(r, clock, k * FRAME, k * FRAME)
    recs = r.records()
    assert send.calls == [1, 2, 3]
    assert (recs[0].sent, recs[0].error) == (False, "unreachable")
    assert not recs[1].sent and "boom" in recs[1].error
    assert recs[2].sent and recs[2].error is None
    assert r.counts() == {"sent": 1, "skipped": 0, "errors": 2, "dry": 0}


def test_skipped_records_are_logged_but_never_sent():
    clock, send = FakeClock(), FakeSend()
    r = make_runner([cue(1, 1.0), cue(2, 2.0)], clock, send, params=SchedulerParams(skip_policy="skip"))
    r.start(tp(10.0, 0.0))
    assert [(x.reason, x.sent, x.dry_run) for x in r.records()] == [("skipped", False, False)] * 2
    assert send.calls == []
    assert r.counts()["skipped"] == 2


def test_poll_fires_at_the_due_time_and_on_fire_gets_records_in_order():
    clock, send, got = FakeClock(), FakeSend(), []

    def on_fire(rec: FireRecord) -> None:
        got.append(rec.cue_id)
        if rec.cue_id == 1:
            raise RuntimeError("a broken callback must not stop the show")

    r = make_runner([cue(1, 0.05), cue(2, 0.07)], clock, send, on_fire=on_fire)
    r.start(tp(0.0, 0.0))
    clock.t = 0.049
    assert r.poll() == []
    clock.t = 0.05
    recs = r.poll()
    assert [(x.cue_id, x.reason) for x in recs] == [(1, "track")] and recs[0].ref_late_ms < 1e-3
    clock.t = 0.08
    assert [x.cue_id for x in r.poll()] == [2]
    assert got == [1, 2]
    assert r.position() == pytest.approx(0.08)
    assert [c.id for c, _ in r.upcoming()] == []


def test_runner_jump_set_cues_and_default_target():
    clock, send = FakeClock(), FakeSend()
    r = make_runner([cue(1, 1.0), cue(2, 5.0)], clock, send, default_target="desk")
    step(r, clock, 0.0, 0.0)                           # feed before start starts the pass
    assert r.running
    clock.t = 0.1
    r.jump(tp(5.0, 0.1, 0.0, "hold"))
    recs = r.records()
    assert [(x.cue_id, x.reason, x.target) for x in recs] == [(1, "chase", "desk"), (2, "track", "desk")]
    r.set_default_target("main")
    r.set_cues([cue(1, 1.0), cue(2, 5.0), cue(3, 5.5)])
    step(r, clock, 5.6, 0.2)
    assert [(x.cue_id, x.target) for x in r.records(after_seq=2)] == [(3, "main")]
    r.set_params(SchedulerParams(skip_policy="skip"))
    r.set_latency_of(zero)
    r.stop()
    step(r, clock, 0.0, 0.3)
    r.jump(tp(0.0, 0.3))
    assert r.last_seq == 3 and not r.running


def test_stop_joins_the_timer_thread():
    r = CueRunner(CueScheduler([cue(1, 50.0)], zero), FakeSend())
    r.start(tp(0.0, time.perf_counter()))
    thread = r._thread
    assert thread is not None and thread.is_alive()
    r.stop()
    assert not thread.is_alive() and not r.running
    r.feed(tp(60.0, time.perf_counter()))
    assert r.records() == []
    r.start(tp(0.0, time.perf_counter()))              # restartable
    assert r._thread is not None and r._thread.is_alive()
    r.stop()


def test_timer_thread_fires_cues_on_time():
    sent_at: list[tuple[int, float]] = []

    def send(c: Cue) -> SendResult:
        sent_at.append((c.id, time.perf_counter()))
        return SendResult(True, "main")

    s = CueScheduler([cue(1, 0.06), cue(2, 0.12)], zero, SchedulerParams(max_extrap_s=0.5))
    r = CueRunner(s, send)
    t0 = time.perf_counter()
    r.start(tp(0.0, t0))
    deadline = t0 + 1.0
    while len(r.records()) < 2 and time.perf_counter() < deadline:
        time.sleep(0.005)
    r.stop()
    assert [x.reason for x in r.records()] == ["track", "track"]
    assert [i for i, _ in sent_at] == [1, 2]
    for (_, t), ideal in zip(sent_at, (0.06, 0.12)):
        assert -1e-4 <= (t - t0) - ideal < 0.02


def test_feed_from_another_thread_while_the_timer_runs_fires_each_cue_once_per_pass():
    rng = random.Random(7)
    cues = [cue(i + 1, 0.05 + 0.1 * i, f"/c{i % 5}", "v" if i % 4 == 0 else None) for i in range(300)]
    sends: list[int] = []

    def send(c: Cue) -> SendResult:
        sends.append(c.id)
        return SendResult(True, "main")

    s = CueScheduler(cues, lat_map({"v": 0.03}), SchedulerParams(skip_policy="all"))
    r = CueRunner(s, send, log_size=5000)
    statuses = ["tracking"] * 8 + ["coasting", "hold"]
    errors: list[BaseException] = []

    def feeder() -> None:
        try:
            r.start(tp(0.0, time.perf_counter()))
            for k in range(1, 2001):
                base = 0.012 * k if k < 1000 else 8.0 + 0.012 * (k - 1000)   # pass 2 starts back at 8 s
                r.feed(tp(base + rng.uniform(-0.15, 0.15), time.perf_counter(), rng.choice([0.95, 1.0, 1.05]),
                          rng.choice(statuses)))
                time.sleep(0.002 if k % 50 == 0 else 0)
        except BaseException as e:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(e)

    th = threading.Thread(target=feeder)
    th.start()
    th.join(10.0)
    r.stop()
    assert not errors and not th.is_alive()
    count = Counter(sends)
    by_time = {c.id: c.time for c in cues}
    assert len(r.records(limit=10_000)) == len(sends) == r.counts()["sent"]
    assert max(count.values()) <= 2
    for cid, t in by_time.items():
        if t <= 7.5 or 12.5 <= t <= 19.5:
            assert count[cid] == 1, (cid, t, count[cid])
        elif 8.5 <= t <= 11.5:
            assert count[cid] == 2, (cid, t, count[cid])
        elif t >= 20.5:
            assert count[cid] == 0, (cid, t, count[cid])


def test_a_cue_due_before_a_jump_is_sent_before_the_cues_the_jump_chases():
    clock, send = FakeClock(), FakeSend()
    cues = [cue(1, 0.99, "/scene", args=[("s", "A")]), cue(2, 5.0, "/scene", args=[("s", "B")])]
    r = make_runner(cues, clock, send, params=SchedulerParams(skip_policy="latest"))
    r.start(tp(0.0, 0.0))
    for k in range(1, 25):
        step(r, clock, k * FRAME, k * FRAME)
    clock.t = 0.995                                    # cue 1 became due at 0.99; the timer has not woken yet
    r.feed(tp(10.0, 0.995))                            # and the tracker re-localises 9 s ahead
    assert send.calls == [1, 2]                        # the scene ends on B, as "latest" means
    assert [(x.cue_id, x.reason) for x in r.records()] == [(1, "track"), (2, "chase")]


def test_a_send_waiting_for_a_lock_held_around_feed_blocks_neither_feed_nor_the_views():
    # the session's send takes the session lock, which the audio thread holds while it calls feed
    session_lock = threading.RLock()
    in_send, holding, fed = threading.Event(), threading.Event(), threading.Event()
    sends: list[int] = []

    def send(c: Cue) -> SendResult:
        in_send.set()
        with session_lock:
            sends.append(c.id)
        return SendResult(True, "main")

    clock = FakeClock()
    r = make_runner([cue(1, 0.1, "/a"), cue(2, 0.2, "/b")], clock, send)
    r.start(tp(0.0, 0.0))

    def audio() -> None:
        with session_lock:
            holding.set()
            if not in_send.wait(2.0):
                return
            clock.t = 0.2
            r.feed(tp(0.2, 0.2))                       # cue 2 is due: queued behind the send in flight
            r.counts(), r.position(), r.upcoming(), r.records()
            fed.set()

    audio_th = threading.Thread(target=audio, daemon=True)
    audio_th.start()
    assert holding.wait(2.0)
    clock.t = 0.1
    timer_th = threading.Thread(target=r.poll, daemon=True)    # what the timer thread does at 0.1
    timer_th.start()
    assert fed.wait(2.0), "feed blocked behind a send (deadlock with the session lock)"
    audio_th.join(2.0)
    timer_th.join(2.0)
    assert not timer_th.is_alive()
    assert sends == [1, 2]                             # in order, both sent by the thread that was sending
    assert [(x.cue_id, x.reason, x.sent) for x in r.records()] == [(1, "track", True), (2, "track", True)]


def test_stop_returns_while_a_send_blocks_and_drops_what_was_still_queued():
    release = threading.Event()
    sends: list[int] = []

    def send(c: Cue) -> SendResult:
        sends.append(c.id)
        if c.id == 1:
            release.wait(5.0)
        return SendResult(True, "main")

    clock = FakeClock()
    r = make_runner([cue(1, 0.1, "/a"), cue(2, 0.1, "/b")], clock, send)
    r.start(tp(0.0, 0.0))
    clock.t = 0.1
    timer_th = threading.Thread(target=r.poll, daemon=True)
    timer_th.start()
    deadline = time.perf_counter() + 2.0
    while not sends and time.perf_counter() < deadline:
        time.sleep(0.001)
    stopper = threading.Thread(target=r.stop, daemon=True)
    stopper.start()
    stopper.join(1.0)
    stopped_while_blocked = not stopper.is_alive()
    release.set()
    timer_th.join(2.0)
    assert stopped_while_blocked, "stop() waited for a blocked send"
    assert sends == [1] and [x.cue_id for x in r.records()] == [1]


def test_stop_racing_the_start_of_the_timer_thread(monkeypatch):
    r = CueRunner(CueScheduler([cue(1, 50.0)], zero), FakeSend())
    real_start = threading.Thread.start
    errors: list[BaseException] = []
    stoppers: list[threading.Thread] = []

    def stop_quietly() -> None:
        try:
            r.stop()
        except BaseException as e:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(e)

    def start(self: threading.Thread) -> None:
        if self.name == "cue-runner" and not stoppers:
            # a stop() from another thread lands between creating the timer thread and starting it
            th = threading.Thread(target=stop_quietly, daemon=True)
            stoppers.append(th)
            real_start(th)
            th.join(0.2)
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    r.start(tp(0.0, time.perf_counter()))
    stoppers[0].join(2.0)
    assert not errors and not stoppers[0].is_alive()
    assert not r.running
