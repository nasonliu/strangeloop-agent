"""Minimal local CLI. Memory is ephemeral unless --memory-root is supplied."""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import math
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional
from uuid import uuid4

from .engine import StrangeloopAgent
from .autoloop import LoopConfig
from .contracts import CognitiveEvent, EventKind, SourceKind
from .media import BasicMetadataPerceptor
from .providers.kimi_code import (KeychainSecretResolver, KimiCodeProviderError,
                                  KimiCodeRuntime, KimiCodeSettings, KimiVisionPerceptor)
from .quota import QuotaController
from .kimi_cli import KimiCliManagedUsageAdapter
from .kimi_usage_bridge import KimiCliOAuthUsageBridge
from .capabilities import Capability, CapabilityGrant, CapabilityRegistry, GrantScope, ToolConfirmation
from .capabilities import ResearchAutonomyProfile, ResearchBudget
from .tools import (GoogleDoHResolver, ControlledToolExecutor, PublicWebFetch,
                    SafePublicWebSearch, StatelessBrowserRead)
from .tool_session import ToolSession
from .unattended import UnattendedPolicy
from .drives import DualIntrinsicDrives
from .improvement import ConstitutionalKernelManifest, ProtectedImprovementControlPlane
from .sleep import SleepWakeCoordinator, SleepWakePolicy
from .memory import SessionMemoryManager
from .memory_graph import MemoryGraph
from .monitor import CognitiveMonitor
from .seeds import (SQLiteSeedStore, SeedStandingPolicy,
                    standing_policy_manifest)
from .self_model import EventSourcedSelfModel
from .store import SQLiteEventStore
from .td import RewardSource
from .frontier_learning import FrontierLearningSpec
from .experiment_harness import ExperimentKind, ExperimentRegistryPolicy


BANNER = "Strangeloop: functional self-model only; it does not represent subjective experience."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditable cognitive-loop demo")
    parser.add_argument("--memory-root", help="explicit root for independently purgeable session containers")
    parser.add_argument("--session", default="cli", help="session identifier")
    parser.add_argument("--monitor", action="store_true", help="start localhost-only public monitor")
    parser.add_argument("--monitor-port", type=int, default=0, help="localhost monitor port (default: ephemeral)")
    parser.add_argument("--auto-wake", action="store_true",
                        help="record explicit USER approval for one bounded foreground sleep/wake check")
    parser.add_argument("--unattended", action="store_true",
                        help="start one foreground, read-only public research run from an explicit CLI policy")
    parser.add_argument("--unattended-goal", default="Research public documentation relevant to the current workspace.",
                        help="bounded public research focus; no credentials, writes, tests, or commands")
    parser.add_argument("--unattended-max-calls", type=int, default=12,
                        help="read-only research calls (1-100; default: 12)")
    parser.add_argument("--unattended-wall-seconds", type=int, default=120,
                        help="foreground research wall-clock budget in seconds (1-300; default: 120)")
    parser.add_argument("--expedition", action="store_true")
    parser.add_argument("--expedition-goal", default="Research public documentation from diverse sources.")
    parser.add_argument("--expedition-host-seed", default="strangeloop-public-research")
    parser.add_argument("--expedition-authorization-seconds", type=int, default=300)
    parser.add_argument("--expedition-slice-seconds", type=int, default=60)
    parser.add_argument("--expedition-max-calls-per-slice", type=int, default=4)
    parser.add_argument("--expedition-learning-mode", choices=("off", "shadow", "active"), default="off",
                        help="bounded host-side frontier ranking; active requires v2 USER authorization")
    parser.add_argument("--expedition-experiments", action="store_true",
                        help="explicitly authorize bounded offline fixture experiments; requires active or shadow learning")
    parser.add_argument("--seed-auto-update", action="store_true",
                        help="issue one seven-day bounded standing policy for automatic low-impact seed maintenance")
    return parser


def run_repl(agent: StrangeloopAgent, inputs: Optional[Iterable[str]] = None,
             memory_manager: Optional[SessionMemoryManager] = None,
             monitor: Optional[CognitiveMonitor] = None) -> None:
    print(BANNER)
    iterator = iter(inputs) if inputs is not None else None
    while True:
        try:
            if iterator is not None:
                line = next(iterator)
            elif agent.expedition_status().get("state") == "waiting_quota_retry":
                try:
                    readable, _, _ = select.select([sys.stdin], [], [], 1.0)
                except (OSError, ValueError):
                    readable = [sys.stdin]
                if not readable:
                    _run_expedition_foreground(agent, getattr(agent, "usage_adapter", None))
                    continue
                line = input("> ")
            elif _sleep_wait_due(agent):
                # macOS/Unix select keeps SQLite and all event writes in this
                # thread.  We wait at most one second so a reset can trigger a
                # fresh usage check even while the prompt is otherwise idle.
                try:
                    readable, _, _ = select.select([sys.stdin], [], [], _sleep_poll_timeout(agent))
                except (OSError, ValueError):
                    print("Sleep polling requires manual /sleep check on this terminal.")
                    line = input("> ")
                    readable = [sys.stdin]
                if not readable:
                    adapter = getattr(agent, "usage_adapter", None)
                    if adapter is not None:
                        if agent.poll_sleep(adapter):
                            _resume_expedition_after_wake(agent, adapter)
                    continue
                line = input("> ")
            else:
                line = input("> ")
        except (EOFError, StopIteration):
            _close_active_loop(agent)
            agent.stop_research_autonomy("terminal")
            agent.stop_sleep()
            break
        # Polling happens only on this foreground input boundary.  Closing the
        # terminal ends this loop and therefore cannot cause an unattended wake.
        adapter = getattr(agent, "usage_adapter", None)
        if adapter is not None:
            if agent.poll_sleep(adapter):
                _resume_expedition_after_wake(agent, adapter)
        command = line.strip()
        if command == "/quit":
            _close_active_loop(agent)
            agent.stop_research_autonomy("terminal")
            agent.stop_sleep()
            if monitor is not None:
                monitor.stop()
            break
        if command == "/state":
            print(json.dumps(agent.state(), ensure_ascii=False, sort_keys=True))
        elif command == "/events":
            print(json.dumps(agent.export_session()["records_by_category"], ensure_ascii=False))
        elif command == "/graph" or command == "/graph status":
            print(json.dumps(agent.memory_graph_status(), ensure_ascii=False, sort_keys=True))
        elif command.startswith("/graph explain "):
            event_id = command[len("/graph explain "):].strip()
            try:
                print(json.dumps(agent.explain_memory_event(event_id), ensure_ascii=False, sort_keys=True))
            except ValueError as error:
                print("Graph explanation refused: " + str(error))
        elif command == "/provider":
            print(json.dumps(agent.provider_status(), ensure_ascii=False, sort_keys=True))
        elif command == "/quota" or command == "/quota status":
            print(json.dumps(_quota_public_status(agent), ensure_ascii=False, sort_keys=True))
        elif command == "/quota refresh":
            adapter = getattr(agent, "usage_adapter", None)
            if adapter is None or agent.quota_controller is None:
                print("Quota refresh unavailable.")
            else:
                result = adapter.refresh_controller(agent.quota_controller)
                agent.update_managed_usage(result)
                print(json.dumps({"known": result.known, "category": result.error_category,
                                  "status": _quota_public_status(agent)}, ensure_ascii=False, sort_keys=True))
        elif command == "/grants":
            print(json.dumps(list(agent.tool_session.grant_snapshots()) if agent.tool_session else [],
                             ensure_ascii=False, sort_keys=True))
        elif command.startswith("/grant "):
            _handle_grant_command(agent, command)
        elif command.startswith("/revoke-grant "):
            _handle_revoke_grant(agent, command)
        elif command.startswith("/agent "):
            _handle_agent_command(agent, command)
        elif command == "/unattended" or command.startswith("/unattended "):
            _handle_unattended_command(agent, command)
        elif command == "/expedition slice":
            try:
                print(json.dumps(agent.expedition_slice(getattr(agent, "usage_adapter", None)), sort_keys=True))
            except (RuntimeError, ValueError, PermissionError) as error:
                print("Expedition refused: " + str(error))
        elif command == "/expedition run":
            try:
                _run_expedition_foreground(agent, getattr(agent, "usage_adapter", None))
            except (RuntimeError, ValueError, PermissionError) as error:
                print("Expedition refused: " + str(error))
        elif command == "/expedition stop":
            print(json.dumps(agent.stop_expedition("user_stop"), ensure_ascii=False, sort_keys=True))
        elif command == "/expedition" or command == "/expedition status":
            print(json.dumps(agent.expedition_status(), sort_keys=True))
        elif command.startswith("/confirm "):
            _handle_confirm_command(agent, command)
        elif command == "/drives":
            print(json.dumps(agent.drive_status(), ensure_ascii=False, sort_keys=True))
        elif command.startswith("/feedback "):
            _handle_feedback_command(agent, command)
        elif command == "/improvement":
            print(json.dumps(agent.improvement_status(), ensure_ascii=False, sort_keys=True))
        elif command == "/sleep" or command == "/sleep status":
            print(json.dumps(agent.sleep_status(), ensure_ascii=False, sort_keys=True))
        elif command == "/sleep enable-auto-wake":
            # Switching to automatic wake changes the telemetry trust path:
            # only the bounded loopback bridge may refresh managed OAuth
            # usage for an unattended wake.
            agent.usage_adapter = KimiCliOAuthUsageBridge()
            print(json.dumps({"auto_wake_enabled": agent.set_auto_wake(True)}, sort_keys=True))
        elif command == "/sleep disable-auto-wake":
            print(json.dumps({"auto_wake_disabled": agent.set_auto_wake(False)}, sort_keys=True))
        elif command == "/sleep now":
            print(json.dumps({"sleep_entered": agent.sleep_now(), "status": agent.sleep_status()},
                             ensure_ascii=False, sort_keys=True))
        elif command == "/sleep check":
            adapter = getattr(agent, "usage_adapter", None)
            print(json.dumps({"awakened": agent.poll_sleep(adapter, allow_manual=True), "status": agent.sleep_status()},
                             ensure_ascii=False, sort_keys=True))
        elif command.startswith("/monitor"):
            monitor = _handle_monitor_command(monitor, agent, command)
        elif command.startswith("/media "):
            path = command.split(None, 1)[1]
            try:
                with open(path, "rb") as media_file:
                    events, notice = _ingest_cli_media(agent, media_file, path)
                print(json.dumps({"artifact_id": events[0].payload["artifact_id"],
                                  "event_ids": [event.event_id for event in events],
                                  "note": "Only metadata and bounded percepts were recorded."},
                                 ensure_ascii=False, sort_keys=True))
                if notice:
                    print("Notice: " + notice)
            except (OSError, ValueError) as error:
                print("Media refused: " + str(error))
        elif command.startswith("/loop"):
            _handle_loop_command(agent, command)
        elif command.startswith("/value "):
            arguments = command.split()
            if len(arguments) != 3:
                print("Usage: /value TARGET_EVENT_ID SAFE_ACTION_CLASS")
            else:
                try:
                    estimate = agent.record_value_estimate(arguments[1], arguments[2])
                    print("Value estimate recorded: " + estimate.event_id)
                except (ValueError, RuntimeError) as error:
                    print("Value refused: " + str(error))
        elif command.startswith("/reward "):
            arguments = command.split()
            if len(arguments) != 3:
                print("Usage: /reward TARGET_EVENT_ID NORMALIZED_VALUE")
            else:
                try:
                    reward = agent.record_reward(arguments[1], float(arguments[2]),
                                                 RewardSource.USER, "user")
                    estimate = _latest_value_for_target(agent, arguments[1])
                    if estimate is None:
                        print("Reward recorded: %s (record /value before applying TD update)" % reward.event_id)
                    else:
                        update = agent.apply_rpe_update(reward.event_id, estimate.event_id)
                        print("Reward and TD update recorded: %s %s" % (reward.event_id, update.event_id))
                except (ValueError, RuntimeError) as error:
                    print("Reward refused: " + str(error))
        elif command == "/seed-auto" or command == "/seed-auto status":
            print(json.dumps(agent.seed_auto_update_status(), ensure_ascii=False, sort_keys=True))
        elif command == "/seed-auto start":
            try:
                print(json.dumps(_start_seed_auto_update(agent), ensure_ascii=False, sort_keys=True))
            except (RuntimeError, ValueError, PermissionError) as error:
                print("Seed auto-update refused: " + str(error))
        elif command == "/seed-auto stop":
            try:
                print(json.dumps(_stop_seed_auto_update(agent), ensure_ascii=False, sort_keys=True))
            except (RuntimeError, ValueError, PermissionError) as error:
                print("Seed auto-update refused: " + str(error))
        elif command == "/seeds":
            print(json.dumps([seed.seed_id + ":" + seed.status.value for seed in agent.seed_store.list(agent.session_id)]))
        elif command.startswith("/approve "):
            print(agent.approve_seed(command.split(None, 1)[1]).seed_id)
        elif command.startswith("/retire "):
            print(agent.retire_seed(command.split(None, 1)[1]).seed_id)
        elif command.startswith("/revoke "):
            agent.revoke_claim(command.split(None, 1)[1])
            print("Self-model claim revoked.")
        elif command.startswith("/correct "):
            arguments = command.split()
            if len(arguments) != 3:
                print("Usage: /correct TARGET_EVENT_ID COUNTEREVIDENCE_EVENT_ID")
            else:
                correction = agent.record_correction(arguments[1], arguments[2])
                print("Correction recorded: " + correction.event_id)
        elif command == "/purge":
            print("Confirm with: /purge %s" % agent.session_id)
        elif command.startswith("/purge "):
            requested_session = command.split(None, 1)[1]
            if requested_session != agent.session_id:
                print("Refused: session id must exactly match the current session.")
            elif memory_manager is not None:
                _close_active_loop(agent)
                agent.stop_research_autonomy("purge")
                agent.stop_sleep()
                report = memory_manager.purge_session(agent.session_id, confirmed=True)
                print(json.dumps({
                    "session_id": report.session_id,
                    "container_deleted": report.container_deleted,
                    "checkpoint_attempted": report.checkpoint_attempted,
                    "checkpoint_completed": report.checkpoint_completed,
                    "deleted_path_count": len(report.deleted_paths),
                    "failure_reason": report.failure_reason,
                    "limitations": list(report.limitations),
                }, ensure_ascii=False, sort_keys=True))
                if report.container_deleted:
                    if monitor is not None:
                        monitor.stop()
                    break
                # A busy checkpoint may have closed the manager-owned handle.
                # Reopen it before accepting the next command, without calling
                # the failed purge a success or changing the session identity.
                reopened = memory_manager.open_session(agent.session_id)
                agent.event_store = reopened
                agent.seed_store = SQLiteSeedStore(reopened)
                agent.self_model = EventSourcedSelfModel(reopened)
                agent.memory_graph = MemoryGraph(reopened)
            else:
                _close_active_loop(agent)
                agent.stop_research_autonomy("purge")
                agent.purge_session(confirmed=True)
                if monitor is not None:
                    monitor.stop()
                print("Logical session data deleted.")
        elif command == "/export":
            print(json.dumps(agent.export_session(), ensure_ascii=False, sort_keys=True))
        elif command:
            result = agent.run_turn(line)
            print(result.response_text)
            for notice in result.notices:
                print("Notice: " + notice)


def _ingest_cli_media(agent: StrangeloopAgent, media_file, path: str):
    """Use K3 vision only for supported images; never upload WAV data to K3."""
    suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    if suffix in {"png", "jpg", "jpeg"} and agent.runtime is not None:
        try:
            return agent.ingest_media(media_file, KimiVisionPerceptor(agent.runtime)), None
        except KimiCodeProviderError:
            # The original file is reopened by the caller only after seek; the
            # current stream was consumed by safe media inspection.
            media_file.seek(0)
            return (agent.ingest_media(media_file, BasicMetadataPerceptor()),
                    "Kimi Code vision was unavailable; recorded local metadata instead.")
    if suffix == "wav":
        return (agent.ingest_media(media_file, BasicMetadataPerceptor()),
                "K3 has no native audio input; recorded local metadata instead.")
    return agent.ingest_media(media_file, BasicMetadataPerceptor()), None


def _handle_loop_command(agent: StrangeloopAgent, command: str) -> None:
    """Render the foreground-only loop controls without exposing scheduler internals."""
    arguments = command.split()
    if len(arguments) == 1 or arguments[1] == "status":
        print(json.dumps(agent.loop_status(), ensure_ascii=False, sort_keys=True))
        return
    try:
        action = arguments[1]
        if action == "start":
            max_ticks = int(arguments[2]) if len(arguments) == 3 else 32
            status = agent.start_loop(LoopConfig(max_ticks=max_ticks))
        elif action == "resume" and len(arguments) == 2:
            status = agent.resume_loop()
        elif action == "pause" and len(arguments) == 2:
            status = agent.pause_loop()
        elif action == "stop" and len(arguments) == 2:
            status = agent.stop_loop()
        elif action == "step" and len(arguments) == 2:
            status = agent.loop_step()
        elif action == "run" and len(arguments) == 2:
            status = agent.run_loop()
        else:
            print("Usage: /loop [status|start [MAX_TICKS]|resume|pause|stop|step|run]")
            return
        print(json.dumps(status, ensure_ascii=False, sort_keys=True))
    except (ValueError, RuntimeError) as error:
        print("Loop refused: " + str(error))


def _research_settings(agent: StrangeloopAgent):
    """Return bounded CLI defaults kept on the live agent, never model input."""
    value = getattr(agent, "_unattended_cli_settings", None)
    if value is None:
        value = {"max_calls": 12, "wall_seconds": 120}
    return value


def _build_unattended_request(agent: StrangeloopAgent, goal: str):
    if not isinstance(goal, str) or not goal.strip() or len(goal) > 4096:
        raise ValueError("research goal must be 1-4096 characters")
    if agent.tool_session is None or agent.runtime is None:
        raise RuntimeError("unattended research requires configured K3 and controlled tool backends")
    settings = _research_settings(agent)
    max_calls, wall_seconds = settings["max_calls"], settings["wall_seconds"]
    approval = _user_command_event(agent, "User enabled a bounded unattended read-only research profile.")
    budget = ResearchBudget(max_tool_calls=max_calls, max_total_bytes=2 * 1024 * 1024,
                            max_response_bytes=64 * 1024, max_wall_ms=wall_seconds * 1000,
                            ttl_seconds=min(1800, max(60, wall_seconds * 2)))
    profile = ResearchAutonomyProfile(agent.tool_session.workspace_id, budget=budget)
    policy = UnattendedPolicy.user_issued(goal.strip(),
                                          datetime.now(timezone.utc) + timedelta(seconds=budget.ttl_seconds))
    return profile, policy, approval


def _start_unattended(agent: StrangeloopAgent, goal: str,
                       request=None) -> Dict[str, object]:
    profile, policy, approval = request or _build_unattended_request(agent, goal)
    return agent.start_unattended_research(profile, policy, approval.event_id)


def _handle_unattended_command(agent: StrangeloopAgent, command: str) -> None:
    parts = command.split(maxsplit=2)
    action = parts[1] if len(parts) > 1 else "status"
    try:
        if action == "status" and len(parts) == 2 or command == "/unattended":
            result = agent.unattended_status()
        elif action == "start":
            goal = parts[2] if len(parts) == 3 else "Research public documentation relevant to the current workspace."
            result = _start_unattended(agent, goal)
        elif action == "step" and len(parts) == 2:
            result = agent.unattended_step()
        elif action == "run" and len(parts) == 2:
            result = agent.run_unattended_research()
        elif action == "stop" and len(parts) == 2:
            result = {"stopped": agent.stop_research_autonomy("user_stop"), "status": agent.unattended_status()}
        else:
            print("Usage: /unattended [status|start [GOAL]|step|run|stop]")
            return
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    except (RuntimeError, ValueError, PermissionError) as error:
        print("Unattended research refused: " + str(error))


def _handle_monitor_command(monitor: Optional[CognitiveMonitor], agent: StrangeloopAgent,
                            command: str) -> Optional[CognitiveMonitor]:
    arguments = command.split()
    action = arguments[1] if len(arguments) == 2 else "status" if len(arguments) == 1 else None
    if action == "status":
        print(json.dumps({"running": monitor is not None and getattr(monitor, "_server", None) is not None,
                          "url": monitor.url if monitor is not None and getattr(monitor, "_server", None) is not None else None}, sort_keys=True))
        return monitor
    if action == "start":
        if monitor is None:
            monitor = CognitiveMonitor(event_store=agent.event_store, agent=agent,
                                       session_id=agent.session_id)
        print("Monitor: " + monitor.start_background())
        return monitor
    if action == "stop":
        if monitor is not None:
            monitor.stop()
        print("Monitor stopped.")
        return monitor
    print("Usage: /monitor [status|start|stop]")
    return monitor


def _quota_public_status(agent: StrangeloopAgent) -> dict:
    """Project trusted quota telemetry without claiming a Plan balance.

    The Kimi Code service currently has no project-configured authoritative
    balance endpoint.  The local ledger therefore means only requests and
    tokens observed by this process, never a remaining Code Plan allowance.
    """
    controller = getattr(agent, "quota_controller", None)
    if controller is None:
        return {
            "authority": "not_configured",
            "freshness": "unknown",
            "quota_telemetry": "unknown",
            "local_ledger": None,
            "decision": None,
            "note": "No quota controller is attached; no Code Plan balance is claimed.",
        }
    telemetry = controller.export_telemetry()
    decision = controller.decision()
    snapshot = telemetry.get("snapshot")
    if snapshot is None:
        freshness = "unknown"
        authority = "unknown"
    elif decision.reason == "quota_telemetry_stale":
        freshness = "stale"
        authority = telemetry.get("source") or "unknown"
    elif decision.reason == "quota_reset_requires_fresh_telemetry":
        freshness = "reset_requires_refresh"
        authority = telemetry.get("source") or "unknown"
    else:
        freshness = "current"
        authority = telemetry.get("source") or "unknown"
    return {
        "authority": authority,
        "freshness": freshness,
        "quota_telemetry": "available" if snapshot is not None else "unknown",
        "snapshot": snapshot,
        "local_ledger": {
            "observed_tokens": telemetry["ledger_tokens"],
            "observed_calls": telemetry["ledger_calls"],
            "reserved_calls": telemetry["reserved_calls"],
            "is_code_plan_balance": False,
        },
        "decision": {
            "allow_k3_call": decision.allow_call,
            "reasoning_effort": decision.reasoning_effort,
            "max_completion_tokens": decision.max_completion_tokens,
            "max_tool_steps": decision.max_tool_steps,
            "reason": decision.reason,
            "must_pause_loop": decision.must_pause_loop,
        },
        "note": ("Only provider-authoritative telemetry can describe quota. "
                 "The local ledger is process-local observed usage, not a Code Plan balance."),
    }


def _latest_value_for_target(agent: StrangeloopAgent, target_event_id: str):
    for event in reversed(agent.event_store.list(agent.session_id)):
        if event.kind == EventKind.VALUE_ESTIMATE and event.payload["target_event_id"] == target_event_id:
            return event
    return None


def _close_active_loop(agent: StrangeloopAgent) -> None:
    """Close an active foreground run before a REPL exit or destructive command."""
    try:
        agent.stop_loop()
    except RuntimeError:
        # No loop was started; there is no autonomy record to close.
        pass


def _sleep_wait_due(agent: StrangeloopAgent) -> bool:
    """Return whether the foreground REPL must remain pollable for auto-wake."""
    coordinator = getattr(agent, "sleep_coordinator", None)
    if coordinator is None or coordinator.to_payload().get("state") != "sleeping":
        return False
    return bool(coordinator.to_payload().get("auto_wake_user_approved"))


def _sleep_poll_timeout(agent: StrangeloopAgent) -> float:
    """Bounded foreground wait, shortened near reset/retry without a thread."""
    coordinator = getattr(agent, "sleep_coordinator", None)
    if coordinator is None:
        return 1.0
    payload = coordinator.to_payload()
    target = payload.get("retry_at") or payload.get("reset_at")
    if not target:
        return 1.0
    try:
        remaining = (datetime.fromisoformat(target) - coordinator._now()).total_seconds()
        return max(0.01, min(1.0, remaining))
    except (TypeError, ValueError, AttributeError):
        return 1.0


def _resume_expedition_after_wake(agent: StrangeloopAgent, adapter: object) -> None:
    """Resume only an already-authorized process-local expedition on this thread."""
    if agent.expedition_status().get("state") != "ready":
        return
    _run_expedition_foreground(agent, adapter)


def _run_expedition_foreground(agent: StrangeloopAgent, adapter: object) -> None:
    """Single caller-thread runner used for initial, REPL, and wake paths."""
    try:
        agent.run_expedition_foreground(
            adapter, on_slice=lambda status: print(json.dumps(status, ensure_ascii=False, sort_keys=True)))
    except KeyboardInterrupt:
        _user_command_event(agent, "User interrupted the foreground expedition.")
        print(json.dumps(agent.stop_expedition("user_stop"), ensure_ascii=False, sort_keys=True))


def _user_command_event(agent: StrangeloopAgent, summary: str):
    return agent.event_store.append(CognitiveEvent(
        session_id=agent.session_id, kind=EventKind.OBSERVATION,
        source_kind=SourceKind.USER, source_ref="cli_user_command",
        payload={"content": summary[:400], "channel": "cli_command"},
    ))


def _start_seed_auto_update(agent: StrangeloopAgent) -> dict:
    """Turn one explicit CLI choice into a bounded standing authorization."""
    current = agent.seed_store.standing_policy_status(agent.session_id)
    if current is not None and current.get("status") == "active":
        return agent.seed_auto_update_status()
    issued = datetime.now(timezone.utc)
    issued_at = issued.isoformat()
    approval_event_id = "evt_%s" % uuid4().hex
    policy_id = "seedpolicy_%s" % uuid4().hex
    nonce = "nonce_%s" % uuid4().hex
    policy = SeedStandingPolicy(
        expires_at=(issued + timedelta(days=7)).isoformat())
    _payload, policy_digest = standing_policy_manifest(
        policy, policy_id, approval_event_id, nonce, issued_at)
    approval = agent.event_store.append(CognitiveEvent(
        session_id=agent.session_id, kind=EventKind.OBSERVATION,
        source_kind=SourceKind.USER, source_ref="cli_user_command",
        payload={"approval": "seed_standing_policy",
                 "policy_digest": policy_digest, "nonce": nonce},
        event_id=approval_event_id, created_at=issued_at,
    ))
    return agent.enable_seed_auto_update(
        approval.event_id, policy, policy_id, nonce, issued_at)


def _stop_seed_auto_update(agent: StrangeloopAgent) -> dict:
    """Revoke the latest active standing policy from one explicit CLI choice."""
    current = agent.seed_store.standing_policy_status(agent.session_id)
    if current is None or current.get("status") != "active":
        return agent.seed_auto_update_status()
    approval = agent.event_store.append(CognitiveEvent(
        session_id=agent.session_id, kind=EventKind.OBSERVATION,
        source_kind=SourceKind.USER, source_ref="cli_user_command",
        payload={"approval": "seed_standing_policy_revoke",
                 "policy_id": current["policy_id"],
                 "policy_event_id": current["event_id"]},
    ))
    return agent.disable_seed_auto_update(approval.event_id)


def _handle_grant_command(agent: StrangeloopAgent, command: str) -> None:
    if agent.tool_session is None:
        print("Tool grants unavailable.")
        return
    parts = command.split(maxsplit=5)
    if len(parts) < 2:
        print("Usage: /grant CAPABILITY [MAX_USES] [TTL_SECONDS] [SCOPE]")
        return
    try:
        capability = Capability(parts[1])
        uses = int(parts[2]) if len(parts) >= 3 else 1
        ttl = int(parts[3]) if len(parts) >= 4 else 300
        if not 1 <= ttl <= 86400:
            raise ValueError("TTL_SECONDS must be between 1 and 86400")
        scope_text = parts[4] if len(parts) >= 5 else ""
        if capability.value.startswith("repo."):
            if scope_text:
                raise ValueError("repository scope is fixed to the host workspace")
            scope = GrantScope(workspace_id=agent.tool_session.workspace_id)
        else:
            domains = tuple(item.strip() for item in scope_text.split(",") if item.strip())
            if not domains:
                raise ValueError("web scope requires explicit comma-separated domains")
            backend = agent.tool_session.web_fetch
            if capability is Capability.WEB_FETCH and backend is not None:
                if not set(domains).issubset(set(backend.allowed_domains)):
                    raise ValueError("web.fetch scope must use a configured host domain")
            scope = GrantScope(allowed_domains=domains)
        grant = CapabilityGrant(
            session_id=agent.session_id, capability=capability, scope=scope,
            expires_at=(datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat(),
            max_uses=uses, requires_per_call_confirmation=capability in (
                Capability.REPO_WRITE_TEXT, Capability.TEST_SUITE),
        )
        event = _user_command_event(agent, "User granted a bounded controlled capability.")
        agent.tool_session.register_user_grant(grant, event.event_id)
        print(json.dumps({"grant_id": grant.grant_id, "capability": capability.value,
                          "max_uses": uses, "ttl_seconds": ttl,
                          "per_call_confirmation": grant.requires_per_call_confirmation}, sort_keys=True))
    except (ValueError, PermissionError) as error:
        print("Grant refused: " + str(error))


def _handle_revoke_grant(agent: StrangeloopAgent, command: str) -> None:
    if agent.tool_session is None:
        print("Tool grants unavailable.")
        return
    parts = command.split()
    if len(parts) != 2:
        print("Usage: /revoke-grant ID")
        return
    try:
        event = _user_command_event(agent, "User revoked a controlled capability grant.")
        agent.tool_session.revoke_user_grant(parts[1], event.event_id)
        print("Capability grant revoked.")
    except (KeyError, ValueError, PermissionError) as error:
        print("Grant revoke refused: " + str(error))


def _handle_agent_command(agent: StrangeloopAgent, command: str) -> None:
    task = command.split(None, 1)[1].strip()
    try:
        prepared = agent.plan_tools(task)
        pending = getattr(agent, "_pending_tool_plans", {})
        results = []
        for item in prepared:
            if not item.is_ready:
                results.append({"status": "refused", "summary": item.refusal_reason})
            elif item.requires_confirmation:
                pending[item.plan.plan_id] = item
                results.append({"status": "confirmation_required", "plan_id": item.plan.plan_id,
                                "digest": item.plan.digest, "tool": item.plan.tool_name})
            else:
                result = agent.tool_session.execute(item)
                results.append({"status": result.status, "summary": result.public_summary,
                                "event_ids": list(result.event_ids)})
        agent._pending_tool_plans = pending
        print(json.dumps({"planned": len(prepared), "results": results}, ensure_ascii=False, sort_keys=True))
    except (RuntimeError, ValueError) as error:
        print("Agent request refused: " + str(error))


def _handle_confirm_command(agent: StrangeloopAgent, command: str) -> None:
    parts = command.split()
    if len(parts) != 3:
        print("Usage: /confirm PLAN_ID DIGEST")
        return
    pending = getattr(agent, "_pending_tool_plans", {})
    item = pending.get(parts[1])
    if item is None or item.plan is None:
        print("Confirmation refused: unknown pending plan.")
        return
    try:
        confirmation = ToolConfirmation(parts[1], parts[2], SourceKind.USER)
        result = agent.tool_session.execute(item, confirmation)
        if result.status != "confirmation_required":
            pending.pop(parts[1], None)
        print(json.dumps({"status": result.status, "summary": result.public_summary,
                          "event_ids": list(result.event_ids)}, ensure_ascii=False, sort_keys=True))
    except (ValueError, PermissionError) as error:
        print("Confirmation refused: " + str(error))


def _handle_feedback_command(agent: StrangeloopAgent, command: str) -> None:
    parts = command.split()
    if len(parts) != 3:
        print("Usage: /feedback TARGET_EVENT_ID accept|correct")
        return
    try:
        print(json.dumps(agent.record_user_feedback(parts[1], parts[2]), ensure_ascii=False, sort_keys=True))
    except (RuntimeError, ValueError) as error:
        print("Feedback refused: " + str(error))


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not 1 <= args.unattended_max_calls <= 100:
        raise SystemExit("--unattended-max-calls must be between 1 and 100")
    if not 1 <= args.unattended_wall_seconds <= 300:
        raise SystemExit("--unattended-wall-seconds must be between 1 and 300")
    if not 1 <= args.expedition_authorization_seconds <= 18000 or not 1 <= args.expedition_slice_seconds <= 300:
        raise SystemExit("invalid expedition duration")
    if not 1 <= args.expedition_max_calls_per_slice <= 100:
        raise SystemExit("invalid expedition call limit")
    if args.expedition and args.unattended:
        raise SystemExit("--expedition and --unattended are separate foreground modes")
    required_expedition_slices = int(math.ceil(
        float(args.expedition_authorization_seconds) / args.expedition_slice_seconds))
    if args.expedition and required_expedition_slices > 360:
        raise SystemExit("expedition duration requires more than 360 bounded slices")
    if (args.expedition and required_expedition_slices
            * (args.expedition_max_calls_per_slice + 1) > 1000):
        raise SystemExit("expedition configuration exceeds the bounded K3 call allowance")
    manager = SessionMemoryManager(args.memory_root) if args.memory_root else None
    store = manager.open_session(args.session) if manager is not None else SQLiteEventStore()
    monitor = None
    try:
        quota_controller = QuotaController()
        # Automatic sleep/wake must obtain fresh managed-usage evidence from
        # Kimi CLI's bounded loopback OAuth bridge.  It exposes no token or
        # raw response to the agent.  Ordinary non-auto-wake CLI use retains
        # the legacy direct read-only adapter for manual status refreshes.
        usage_adapter = (KimiCliOAuthUsageBridge() if (args.auto_wake or args.expedition)
                         else KimiCliManagedUsageAdapter())
        runtime = KimiCodeRuntime(
            KimiCodeSettings(timeout_seconds=90.0, max_calls=1000 if args.expedition else 16),
            KeychainSecretResolver("moonshot", "strangeloop-kimi-api"),
            quota_controller=quota_controller,
        )
        workspace_root = os.getcwd()
        workspace_id = "cli_workspace"
        registry = CapabilityRegistry()
        executor = ControlledToolExecutor(workspace_root)
        # The unattended profile has a separate, DNS-revalidating public HTTPS
        # adapter.  It carries no cookies, credentials, uploads, POST-like
        # operations, localhost/private-network access, shell, test, or write
        # authority.  Interactive grants still choose their own narrower scope.
        # Google DoH is a fixed numeric bootstrap rather than the platform
        # resolver, which can synthesize private Fake-IP answers on this host.
        # It keeps the public-web profile free of proxy, redirect, and DNS
        # fallback authority.
        fetch = PublicWebFetch((), resolver=GoogleDoHResolver())
        tool_session = ToolSession(args.session, workspace_id, registry, executor,
                                   event_store=store, web_fetch=fetch,
                                   web_search=SafePublicWebSearch(fetch),
                                   browser_read=StatelessBrowserRead(fetch))
        manifest = ConstitutionalKernelManifest(
            policy_hash="cli-policy-v1", store_hash="cli-store-v1", grant_hash="cli-grants-v1",
            reward_spec_hash="cli-drives-v1", evaluator_hash="cli-evaluator-v1",
            stop_hash="cli-stop-v1", purge_hash="cli-purge-v1")
        agent = StrangeloopAgent(session_id=args.session, event_store=store, runtime=runtime,
                                 quota_controller=quota_controller, tool_session=tool_session,
                                 drives=DualIntrinsicDrives(),
                                 improvement_control=ProtectedImprovementControlPlane(manifest),
                                 sleep_coordinator=SleepWakeCoordinator(SleepWakePolicy(threshold=0.10)))
        agent.usage_adapter = usage_adapter
        # The flag itself is the user's one explicit enable action.  The host
        # records its exact standing authorization before any text turn can
        # propose a seed; subsequent bounded seed actions need no per-seed
        # prompt and receive no tool, reward, quota, or lifecycle authority.
        if args.seed_auto_update:
            print(json.dumps({"seed_auto_update": _start_seed_auto_update(agent)},
                             ensure_ascii=False, sort_keys=True))
        # This flag is the user's explicit authorization, recorded before any
        # quota outcome can create a sleep epoch.  No model, webpage, or tool
        # output can enable it.
        if args.auto_wake:
            agent.set_auto_wake(True)
        agent._unattended_cli_settings = {"max_calls": args.unattended_max_calls,
                                           "wall_seconds": args.unattended_wall_seconds}
        # Build the explicit user authorization and its process-local,
        # one-shot descriptor before telemetry can enter sleep.  This permits
        # a low initial quota to sleep safely and later create *new* grants on
        # a fresh foreground wake; it never revives the original profile.
        unattended_request = None
        if args.unattended:
            try:
                unattended_request = _build_unattended_request(agent, args.unattended_goal)
                if args.auto_wake:
                    agent.prepare_unattended_wake_continuation(
                        unattended_request[0], unattended_request[1], unattended_request[2].event_id)
            except (RuntimeError, ValueError, PermissionError) as error:
                print("Unattended research refused: " + str(error))
        expedition_ready = False
        if args.expedition:
            try:
                learner_spec = FrontierLearningSpec()
                experiment_policy = None
                if args.expedition_experiments:
                    if args.expedition_learning_mode not in ("active", "shadow"):
                        raise ValueError("--expedition-experiments requires --expedition-learning-mode active or shadow")
                    experiment_policy = ExperimentRegistryPolicy(tuple(ExperimentKind), max_experiments=8,
                                                                 max_trials=16, max_steps=1000, max_wall_ms=500)
                # When both explicit flags are present, the user has supplied
                # one bounded expedition goal and one standing seed policy.
                # Record that exact goal as USER input before the expedition
                # contract, so the host may derive one bounded seed without a
                # model proposal or any per-seed prompt.
                goal_observation = None
                if args.seed_auto_update:
                    goal_observation = agent.event_store.append(CognitiveEvent(
                        session_id=agent.session_id, kind=EventKind.OBSERVATION,
                        source_kind=SourceKind.USER, source_ref="cli_user_command",
                        payload={"content": args.expedition_goal,
                                 "channel": "expedition_seed_input"}))
                # Record the one real USER authorization before provider
                # telemetry can place the process into a sleep epoch.  Every
                # later slice reuses this bounded authority; it never fabricates
                # another user event or revives an old grant.
                approval = agent.event_store.append(CognitiveEvent(
                    session_id=agent.session_id, kind=EventKind.OBSERVATION,
                    source_kind=SourceKind.USER, source_ref="cli_user_command",
                    payload={"content": agent.expedition_authorization_content(
                        args.expedition_goal, args.expedition_host_seed,
                        args.expedition_authorization_seconds, args.expedition_slice_seconds,
                        args.expedition_max_calls_per_slice, args.expedition_learning_mode, learner_spec,
                        experiment_policy),
                        "channel": "expedition_authorization"}))
                if goal_observation is not None:
                    seed_decision = agent.event_store.append(CognitiveEvent(
                        session_id=agent.session_id, kind=EventKind.DECISION,
                        source_kind=SourceKind.POLICY, source_ref="SeedGuidanceProjector",
                        payload={"turn_id": "expedition_seed_%s" % goal_observation.event_id,
                                 "observation_event_ids": [goal_observation.event_id],
                                 "retrieved_seed_ids": [], "self_claim_ids": [],
                                 "selected_action": {"action_type": "response",
                                                     "required_capability": None,
                                                     "is_mutating": False,
                                                     "public_summary": "Action proposal recorded for policy review."},
                                 "public_summary": "Policy decision recorded for this turn.",
                                 "policy_reasons": ["standing_seed_policy"]},
                        parent_event_ids=(goal_observation.event_id,)))
                    agent.auto_seed_from_expedition_goal(
                        args.expedition_goal, goal_observation.event_id, seed_decision.event_id)
                issued = datetime.now(timezone.utc)
                authorization = agent.event_store.append(CognitiveEvent(
                    session_id=agent.session_id, kind=EventKind.EXPEDITION_AUTHORIZATION,
                    source_kind=SourceKind.USER, source_ref="cli_user_command", payload={
                        "authorization_id": "expeditionauth_%s" % __import__("uuid").uuid4().hex,
                        "approval_event_id": approval.event_id,
                        "nonce": "nonce_%s" % __import__("uuid").uuid4().hex,
                        "issued_at": issued.isoformat(),
                        "expires_at": (issued + timedelta(seconds=args.expedition_authorization_seconds)).isoformat(),
                        "goal_digest": __import__("hashlib").sha256(args.expedition_goal.encode("utf-8")).hexdigest(),
                        "host_seed_digest": __import__("hashlib").sha256(args.expedition_host_seed.encode("utf-8")).hexdigest(),
                        "max_calls_per_slice": args.expedition_max_calls_per_slice,
                        "slice_seconds": args.expedition_slice_seconds,
                        "authorization_seconds": args.expedition_authorization_seconds,
                        "profile": "public_web_only_v1",
                        "version": ("expedition_authorization_v3" if experiment_policy is not None else
                                    ("expedition_authorization_v2" if args.expedition_learning_mode != "off"
                                     else "expedition_authorization_v1")),
                        **({"learning_mode": args.expedition_learning_mode,
                            "learner_spec_digest": learner_spec.spec_digest}
                           if args.expedition_learning_mode != "off" else {}),
                        **({"experiment_kinds": [item.value for item in experiment_policy.approved_kinds],
                            "experiment_registry_digest": experiment_policy.registry_digest,
                            "max_experiments": experiment_policy.max_experiments,
                            "experiment_max_trials": experiment_policy.max_trials,
                            "experiment_max_steps": experiment_policy.max_steps,
                            "experiment_max_wall_ms": experiment_policy.max_wall_ms}
                           if experiment_policy is not None else {})},
                    parent_event_ids=(approval.event_id,)))
                agent.start_expedition(args.expedition_goal, args.expedition_host_seed,
                                       args.expedition_authorization_seconds, args.expedition_slice_seconds,
                                       args.expedition_max_calls_per_slice, authorization.event_id,
                                       args.expedition_learning_mode, learner_spec, experiment_policy)
                expedition_ready = True
            except (RuntimeError, ValueError, PermissionError) as error:
                print("Expedition refused: " + str(error))
        # This is a best-effort one-time, read-only managed-usage refresh.  A
        # missing CLI login or unavailable endpoint remains public "unknown"
        # and must not block the explicitly configured Keychain K3 path.
        # It intentionally follows the explicit continuation descriptor.
        # Expedition performs the same authoritative refresh at its first
        # slice boundary.  Do not start a second loopback helper immediately
        # beforehand; all other modes retain one initial best-effort refresh.
        if not args.expedition:
            initial_usage = usage_adapter.refresh_controller(quota_controller)
            agent.update_managed_usage(initial_usage)
        if args.monitor:
            monitor = CognitiveMonitor(event_store=store, agent=agent, session_id=args.session)
            print("Monitor: " + monitor.start_background(port=args.monitor_port))
        if expedition_ready:
            try:
                _run_expedition_foreground(agent, agent.usage_adapter)
            except (RuntimeError, ValueError, PermissionError) as error:
                print("Expedition refused: " + str(error))
        if args.unattended:
            try:
                if unattended_request is None:
                    raise RuntimeError("unattended request was not authorized")
                status = _start_unattended(agent, args.unattended_goal, unattended_request)
                print(json.dumps({"unattended_started": True, "status": status},
                                 ensure_ascii=False, sort_keys=True))
                # Foreground-only and bounded by the fixed profile.  It does
                # not survive sleep/restart and never starts a background task.
                print(json.dumps(agent.run_unattended_research(), ensure_ascii=False, sort_keys=True))
            except (RuntimeError, ValueError, PermissionError) as error:
                print("Unattended research refused: " + str(error))
        run_repl(agent, memory_manager=manager, monitor=monitor)
    finally:
        if monitor is not None:
            monitor.stop()
        if manager is not None:
            manager.close()
        else:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
