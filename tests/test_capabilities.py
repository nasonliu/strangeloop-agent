from datetime import datetime, timedelta, timezone
import threading
import unittest

from strangeloop.capabilities import (Capability, CapabilityGrant, CapabilityRegistry,
                                      GrantScope, GrantStatus, ResearchAutonomyProfile,
                                      ResearchBudget, ToolPlan)
from strangeloop.contracts import SourceKind


def future(seconds=60):
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


class CapabilityRegistryTests(unittest.TestCase):
    def repo_grant(self, uses=2):
        return CapabilityGrant("session", Capability.REPO_READ, GrantScope(workspace_id="main"),
                               future(), uses)

    def test_only_user_can_grant_or_revoke_and_restart_suspends(self):
        registry = CapabilityRegistry()
        grant = self.repo_grant()
        with self.assertRaises(PermissionError):
            registry.grant(grant, SourceKind.MODEL)
        registry.grant(grant, SourceKind.USER)
        with self.assertRaises(PermissionError):
            registry.revoke(grant.grant_id, SourceKind.TOOL)
        self.assertEqual(GrantStatus.RESTART_SUSPENDED,
                         registry.suspend_after_restart()[0].status)
        with self.assertRaises(PermissionError):
            registry.consume(ToolPlan("session", grant.grant_id, "repo.read",
                                      {"workspace_id": "main", "path": "README.md"}))

    def test_plan_digest_scope_ttl_and_max_uses_fail_closed(self):
        registry = CapabilityRegistry()
        grant = self.repo_grant(uses=1)
        registry.grant(grant, SourceKind.USER)
        plan = ToolPlan("session", grant.grant_id, "repo.read",
                        {"workspace_id": "main", "path": "README.md"})
        self.assertEqual(grant, registry.consume(plan))
        self.assertEqual(GrantStatus.EXHAUSTED, registry.snapshot(grant.grant_id).status)
        with self.assertRaises(PermissionError):
            registry.consume(plan)

        expired = CapabilityGrant("expired", Capability.REPO_STATUS, GrantScope(workspace_id="main"),
                                  future(1), 1)
        expired = CapabilityGrant(expired.session_id, expired.capability, expired.scope,
                                  expired.expires_at, expired.max_uses, False, expired.grant_id,
                                  (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat())
        registry.grant(expired, SourceKind.USER)
        with self.assertRaises(PermissionError):
            registry.consume(ToolPlan("expired", expired.grant_id, "repo.status", {"workspace_id": "main"}),
                             now=future(5))

        allowed = CapabilityGrant("web", Capability.WEB_FETCH,
                                  GrantScope(allowed_domains=("example.com",)), future(), 1)
        registry.grant(allowed, SourceKind.USER)
        with self.assertRaises(PermissionError):
            registry.consume(ToolPlan("web", allowed.grant_id, "web.fetch",
                                      {"url": "https://other.example/path"}))

    def test_plan_is_immutable_and_atomic_under_competing_consumers(self):
        registry = CapabilityRegistry()
        grant = self.repo_grant(uses=1)
        registry.grant(grant, SourceKind.USER)
        source = {"workspace_id": "main", "path": "README.md"}
        plan = ToolPlan("session", grant.grant_id, "repo.read", source)
        source["path"] = "changed.md"
        self.assertEqual("README.md", plan.arguments["path"])
        outcomes = []

        def consume():
            try:
                registry.consume(plan)
                outcomes.append("ok")
            except PermissionError:
                outcomes.append("denied")

        threads = [threading.Thread(target=consume) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(1, outcomes.count("ok"))
        self.assertEqual(7, outcomes.count("denied"))

    def test_scope_and_plan_shape_are_typed(self):
        with self.assertRaises(ValueError):
            GrantScope(allowed_domains=("https://example.com",))
        with self.assertRaises(ValueError):
            CapabilityGrant("s", Capability.WEB_FETCH, GrantScope(), future(), 1)
        with self.assertRaises(ValueError):
            ToolPlan("s", "g", "repo.read", {"private_reasoning": "no"})
        with self.assertRaises(ValueError):
            ToolPlan("s", "g", "not-real", {})
        registry = CapabilityRegistry()
        search = CapabilityGrant("search", Capability.WEB_SEARCH,
                                 GrantScope(allowed_domains=("search.example",)), future(), 1)
        registry.grant(search, SourceKind.USER)
        registry.consume(ToolPlan("search", search.grant_id, "web.search", {"query": "safe query"}))

    def test_research_profile_is_read_only_public_https_and_never_a_command_grant(self):
        profile = ResearchAutonomyProfile("main", ResearchBudget(
            max_tool_calls=3, max_total_bytes=4096, max_response_bytes=1024,
            max_wall_ms=2000, ttl_seconds=60))
        self.assertEqual((Capability.REPO_STATUS, Capability.REPO_SEARCH, Capability.REPO_READ,
                          Capability.WEB_FETCH, Capability.WEB_SEARCH, Capability.BROWSER_READ),
                         profile.capabilities)
        grants = profile.grants_for("session")
        self.assertNotIn(Capability.REPO_WRITE_TEXT, [grant.capability for grant in grants])
        self.assertNotIn(Capability.TEST_SUITE, [grant.capability for grant in grants])
        web = next(grant for grant in grants if grant.capability is Capability.WEB_FETCH)
        self.assertTrue(web.scope.public_https)
        registry = CapabilityRegistry()
        registry.grant(web, SourceKind.USER)
        registry.consume(ToolPlan("session", web.grant_id, "web.fetch", {"url": "https://example.com/"}))
        local = next(grant for grant in profile.grants_for("other") if grant.capability is Capability.WEB_FETCH)
        registry.grant(local, SourceKind.USER)
        with self.assertRaises(PermissionError):
            registry.consume(ToolPlan("other", local.grant_id, "web.fetch", {"url": "https://localhost/"}))

    def test_public_web_only_profile_mints_no_repository_grants_and_has_distinct_binding(self):
        budget = ResearchBudget(max_tool_calls=3, max_total_bytes=4096, max_response_bytes=1024,
                                max_wall_ms=2000, ttl_seconds=60)
        default = ResearchAutonomyProfile("main", budget, profile_id="default")
        public = ResearchAutonomyProfile("main", budget, profile_id="public", public_web_only=True)
        self.assertEqual((Capability.WEB_FETCH, Capability.WEB_SEARCH, Capability.BROWSER_READ),
                         public.capabilities)
        self.assertTrue(all(grant.scope.public_https for grant in public.grants_for("session")))
        self.assertFalse(any(grant.capability.value.startswith("repo.") for grant in public.grants_for("session")))
        self.assertNotEqual(default.binding_digest(), public.binding_digest())
        with self.assertRaises(ValueError):
            ResearchAutonomyProfile("main", budget, public_web_only=1)


if __name__ == "__main__":
    unittest.main()
