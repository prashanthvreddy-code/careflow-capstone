"""Provider-agnostic LLM client on top of LiteLLM, with structured output + repair loop (M1)
and native tool calling (M2).

The provider is chosen only by the model string from Settings (e.g. "gemini/...", "groq/...",
"anthropic/..."). Switching provider must never require a code change here.

M7 (phase 2 decisions): CareFlow now runs the fallback chain itself instead of LiteLLM's
`fallbacks=` option, so that each model gets its own timeout, retry and breaker, and the trace
can say which model ACTUALLY answered (the M6 gap). Per model, in order:
  breaker "llm:<model>" -> [seeded fault, scenarios only] -> litellm.completion(timeout=30 s,
  num_retries=0) -> 1 retry on transient errors -> next model.
While the primary's breaker is open, calls go straight to the fallback (no waiting).
One "llm.call" generation span per call: model requested and used, fallback_used, attempts,
token counts, latency. Never the prompt or the reply.
"""
import json
from typing import Any, TypeVar

import litellm  # call it as `litellm.completion(...)` so tests can patch it
from pydantic import BaseModel, ValidationError

from capstone.observability.tracing import event, get_tracer
from capstone.reliability.config import Reliability, get_reliability
from capstone.reliability.retries import RetryExhausted, call_with_retry
from capstone.settings import Settings

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Raised when the model's output still fails parsing/validation after all repair attempts."""

    def __init__(self, errors: list[str]):
        super().__init__(f"structured output failed after {len(errors)} attempt(s)")
        self.errors = errors


class LLMUnavailable(Exception):
    """Every model in the chain failed (or its breaker was open)."""

    def __init__(self, failures: list[str]):
        super().__init__("no model answered: " + "; ".join(failures))
        self.failures = failures


class ToolCall(BaseModel):
    """One tool request from the model. Bad argument JSON is kept as an error, not a crash."""

    id: str
    name: str
    arguments: dict[str, Any] = {}
    arguments_error: str | None = None


class AssistantTurn(BaseModel):
    """The model's reply in a tool loop: final text, or one or more tool requests."""

    content: str = ""
    tool_calls: list[ToolCall] = []

    def as_message(self) -> dict:
        """The assistant message to append to the history before the tool results."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in self.tool_calls
            ]
        return msg


def tool_result_message(call: ToolCall, content: str) -> dict:
    """The message that answers one tool call (matched to it by the call id)."""
    return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}


class LLMClient:
    """Thin wrapper: one place that talks to LiteLLM for the whole project."""

    def __init__(self, settings: Settings, reliability: Reliability | None = None):
        self.settings = settings
        self.reliability = reliability or get_reliability()

    def models(self) -> list[str]:
        """The primary model, then the comma-separated fallback chain, in order."""
        s = self.settings
        chain = [m.strip() for m in (s.llm_fallback_model or "").split(",") if m.strip()]
        return [s.llm_model, *[m for m in chain if m != s.llm_model]]

    def _completion(self, kwargs: dict):
        """Run one completion through the hardened chain; returns LiteLLM's response."""
        rel, models = self.reliability, self.models()
        failures: list[str] = []
        with get_tracer().span("llm.call", kind="generation", model_requested=models[0],
                               tools=bool(kwargs.get("tools")), json_mode="response_format" in kwargs) as span:
            for i, model in enumerate(models):
                breaker = rel.breakers.get(f"llm:{model}")

                def once(model=model, breaker=breaker):
                    return breaker.call(lambda: self._one_call(model, kwargs))
                try:
                    resp, attempts = call_with_retry(
                        once, rel.config.llm_retry, sleep=rel.sleep, rng=rel.rng,
                        on_retry=lambda n, e, w, m=model: event("retry", model=m, attempt=n, error=type(e).__name__,
                                                                wait_s=round(w, 2)))
                except Exception as e:  # noqa: BLE001 - any failure of this model -> try the next one
                    last = e.last if isinstance(e, RetryExhausted) else e
                    failures.append(f"{model}: {type(last).__name__}")
                    event("model_failed", model=model, error=type(last).__name__,
                          attempts=getattr(e, "attempts", 1), breaker=breaker.state)
                    continue
                usage = getattr(resp, "usage", None)
                span.set(model=model, response_model=getattr(resp, "model", None), fallback_used=i > 0,
                         attempts=attempts, failed_models=[f.split(":")[0] for f in failures],
                         usage={"input": getattr(usage, "prompt_tokens", None),
                                "output": getattr(usage, "completion_tokens", None)} if usage else None)
                return resp
            span.set(failed_models=[f.split(":")[0] for f in failures])
            raise LLMUnavailable(failures)

    def _one_call(self, model: str, kwargs: dict):
        rel = self.reliability
        if rel.injector:
            rel.injector.before_call(f"llm:{model}", "llm", timeout_s=rel.config.llm_timeout_s)
        return litellm.completion(**{**kwargs, "model": model})

    def complete(self, messages: list[dict], *, json_mode: bool = False) -> str:
        """Send `messages`, return the reply text.

        Contract (the tests check these):
        - call `litellm.completion(...)` with model=settings.llm_model, temperature and max_tokens
          from settings
        - M7: if the primary fails (after 1 retry) or its breaker is open, the same call goes to
          settings.llm_fallback_model (CareFlow runs the chain; no LiteLLM `fallbacks=`)
        - json_mode=True -> response_format={"type": "json_object"}
        - return `resp.choices[0].message.content`
        """
        kwargs = self._base_kwargs(messages)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._completion(kwargs)
        # content can be None (e.g. output cut off); "" then fails json.loads and enters repair
        return resp.choices[0].message.content or ""

    def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        """One round of native tool calling (M2). `tools` are OpenAI-style function schemas;
        LiteLLM translates them for Gemini/Claude. Same model, fallback and settings as complete().
        The model may answer in text or request tools; running the tools is the caller's job."""
        kwargs = self._base_kwargs(messages)
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
        message = self._completion(kwargs).choices[0].message
        calls = []
        for i, tc in enumerate(getattr(message, "tool_calls", None) or []):
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                if not isinstance(args, dict):
                    raise TypeError("arguments must be a JSON object")
                calls.append(ToolCall(id=tc.id or f"call_{i}", name=tc.function.name, arguments=args))
            except (ValueError, TypeError) as e:
                calls.append(ToolCall(id=tc.id or f"call_{i}", name=tc.function.name,
                                      arguments_error=f"invalid arguments: {e}"))
        return AssistantTurn(content=message.content or "", tool_calls=calls)

    def _base_kwargs(self, messages: list[dict]) -> dict:
        s = self.settings
        kwargs = {
            "model": s.llm_model,
            "messages": messages,
            "temperature": s.llm_temperature,
            "max_tokens": s.llm_max_tokens,
            # providers differ (e.g. some Claude models reject temperature=0): drop what a model
            # doesn't support instead of failing, so the fallback chain keeps working
            "drop_params": True,
            # M7: each attempt is bounded; retries and fallback are CareFlow's, not LiteLLM's
            "timeout": self.reliability.config.llm_timeout_s,
            "num_retries": 0,
        }
        return kwargs

    def complete_structured(self, messages: list[dict], schema: type[T]) -> T:
        """Return a validated `schema` instance, using the A1 repair loop.

        Steps:
        1. raw = self.complete(messages, json_mode=True)
        2. try: json.loads(raw) and schema.model_validate(data) -> return it
        3. on (json.JSONDecodeError, ValidationError): record str(e); if attempts are left
           (settings.llm_max_retries), append the bad reply as an assistant turn and a user turn
           containing the error + "return corrected JSON only", then call complete() again
        4. after the last failed attempt: raise StructuredOutputError(errors)
        Do not mutate the caller's `messages` list (work on a copy).
        """
        history = list(messages)  # copy: the caller's list stays untouched
        errors: list[str] = []
        for attempt in range(self.settings.llm_max_retries + 1):
            raw = self.complete(history, json_mode=True)
            try:
                return schema.model_validate(json.loads(_strip_fences(raw)))
            except (json.JSONDecodeError, ValidationError) as e:
                errors.append(str(e))
                if attempt == self.settings.llm_max_retries:
                    break
                history = history + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply was not valid. Error:\n{e}\n\n"
                            "Return corrected json only, with exactly the keys requested. "
                            "Do not invent values that are not in the message; use null instead."
                        ),
                    },
                ]
        raise StructuredOutputError(errors)


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers some models add despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()
