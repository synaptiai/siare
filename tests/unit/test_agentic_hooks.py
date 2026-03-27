"""Tests for agentic evolution hooks (Hybrid Evolution)"""

import asyncio

import pytest

from siare.core.hooks import (
    AgenticEvolutionHooks,
    HookContext,
    HookRegistry,
    HookRunner,
    fire_agentic_evolution_hook,
)
from siare.core.models import (
    MutationType,
    SupervisorDirective,
    VariationResult,
)


# ============================================================================
# Test Implementation of AgenticEvolutionHooks
# ============================================================================


class MockAgenticHooks:
    """Test implementation of AgenticEvolutionHooks protocol."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    async def on_variation_start(self, ctx, parent_sop, directive):
        self.calls.append(("on_variation_start", (ctx,), {
            "parent_sop": parent_sop,
            "directive": directive,
        }))

    async def on_variation_iteration(self, ctx, iteration, quality, improved):
        self.calls.append(("on_variation_iteration", (ctx,), {
            "iteration": iteration,
            "quality": quality,
            "improved": improved,
        }))

    async def on_variation_complete(self, ctx, result):
        self.calls.append(("on_variation_complete", (ctx,), {
            "result": result,
        }))

    async def on_supervisor_redirect(self, ctx, directive, stagnation_generations):
        self.calls.append(("on_supervisor_redirect", (ctx,), {
            "directive": directive,
            "stagnation_generations": stagnation_generations,
        }))
        return True


# ============================================================================
# Protocol Compliance
# ============================================================================


class TestAgenticEvolutionHooksProtocol:
    """Tests that the protocol is correctly defined."""

    def test_mock_implements_protocol(self):
        hooks = MockAgenticHooks()
        assert isinstance(hooks, AgenticEvolutionHooks)

    def test_incomplete_implementation_fails_protocol(self):
        class PartialHooks:
            async def on_variation_start(self, ctx, parent_sop, directive):
                pass

        hooks = PartialHooks()
        assert not isinstance(hooks, AgenticEvolutionHooks)


# ============================================================================
# HookRegistry Tests
# ============================================================================


class TestHookRegistryAgentic:
    """Tests for AgenticEvolutionHooks in the HookRegistry."""

    def setup_method(self):
        HookRegistry.clear_all()

    def teardown_method(self):
        HookRegistry.clear_all()

    def test_default_is_none(self):
        assert HookRegistry.get_agentic_evolution_hooks() is None

    def test_set_and_get(self):
        hooks = MockAgenticHooks()
        HookRegistry.set_agentic_evolution_hooks(hooks)
        assert HookRegistry.get_agentic_evolution_hooks() is hooks

    def test_clear_all_clears_agentic_hooks(self):
        hooks = MockAgenticHooks()
        HookRegistry.set_agentic_evolution_hooks(hooks)
        HookRegistry.clear_all()
        assert HookRegistry.get_agentic_evolution_hooks() is None

    def test_agentic_hooks_independent_of_evolution_hooks(self):
        """Agentic hooks don't interfere with regular evolution hooks."""
        agentic = MockAgenticHooks()
        HookRegistry.set_agentic_evolution_hooks(agentic)
        assert HookRegistry.get_evolution_hooks() is None
        assert HookRegistry.get_agentic_evolution_hooks() is agentic


# ============================================================================
# Fire Hook Tests
# ============================================================================


class TestFireAgenticEvolutionHook:
    """Tests for the fire_agentic_evolution_hook convenience function."""

    def setup_method(self):
        HookRegistry.clear_all()

    def teardown_method(self):
        HookRegistry.clear_all()

    @pytest.mark.asyncio
    async def test_fire_returns_none_when_no_hooks(self):
        ctx = HookContext()
        result = await fire_agentic_evolution_hook(
            "on_variation_start", ctx, None, None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_fire_on_variation_start(self):
        hooks = MockAgenticHooks()
        HookRegistry.set_agentic_evolution_hooks(hooks)
        ctx = HookContext()

        await fire_agentic_evolution_hook(
            "on_variation_start", ctx, "mock_sop", None
        )

        assert len(hooks.calls) == 1
        assert hooks.calls[0][0] == "on_variation_start"

    @pytest.mark.asyncio
    async def test_fire_on_variation_iteration(self):
        hooks = MockAgenticHooks()
        HookRegistry.set_agentic_evolution_hooks(hooks)
        ctx = HookContext()

        await fire_agentic_evolution_hook(
            "on_variation_iteration", ctx, 2, 0.85, True
        )

        assert len(hooks.calls) == 1
        assert hooks.calls[0][2]["iteration"] == 2
        assert hooks.calls[0][2]["quality"] == 0.85
        assert hooks.calls[0][2]["improved"] is True

    @pytest.mark.asyncio
    async def test_fire_on_variation_complete(self):
        hooks = MockAgenticHooks()
        HookRegistry.set_agentic_evolution_hooks(hooks)
        ctx = HookContext()
        result = VariationResult(reason="no_improvement")

        await fire_agentic_evolution_hook(
            "on_variation_complete", ctx, result
        )

        assert len(hooks.calls) == 1
        assert hooks.calls[0][2]["result"] is result

    @pytest.mark.asyncio
    async def test_fire_on_supervisor_redirect(self):
        hooks = MockAgenticHooks()
        HookRegistry.set_agentic_evolution_hooks(hooks)
        ctx = HookContext()
        directive = SupervisorDirective(
            strategy="explore minimal topologies",
            focusArea="topology",
            mutationTypes=[MutationType.REMOVE_ROLE],
            explorationTarget="low-complexity cells",
            rationale="test",
        )

        result = await fire_agentic_evolution_hook(
            "on_supervisor_redirect", ctx, directive, 5
        )

        assert result is True
        assert len(hooks.calls) == 1
        assert hooks.calls[0][2]["stagnation_generations"] == 5

    @pytest.mark.asyncio
    async def test_fire_nonexistent_hook_returns_none(self):
        hooks = MockAgenticHooks()
        HookRegistry.set_agentic_evolution_hooks(hooks)
        ctx = HookContext()

        result = await fire_agentic_evolution_hook(
            "on_nonexistent_hook", ctx
        )
        assert result is None


# ============================================================================
# Error Isolation Tests
# ============================================================================


class TestAgenticHookErrorIsolation:
    """Tests that hook errors don't break execution."""

    def setup_method(self):
        HookRegistry.clear_all()

    def teardown_method(self):
        HookRegistry.clear_all()

    @pytest.mark.asyncio
    async def test_error_in_hook_returns_none(self):
        class FailingHooks:
            async def on_variation_start(self, ctx, parent_sop, directive):
                raise RuntimeError("Hook crashed")

            async def on_variation_iteration(self, ctx, iteration, quality, improved):
                pass

            async def on_variation_complete(self, ctx, result):
                pass

            async def on_supervisor_redirect(
                self, ctx, directive, stagnation_generations
            ):
                return True

        HookRegistry.set_agentic_evolution_hooks(FailingHooks())
        ctx = HookContext()

        result = await fire_agentic_evolution_hook(
            "on_variation_start", ctx, None, None
        )
        assert result is None  # Error swallowed, returned None
