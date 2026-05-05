"""
llm_client.py — Provider-agnostic LLM wrapper used by all committee agents.

Normalises OpenAI, Ollama, Claude (Anthropic), and Gemini into a single
.chat() interface so agents don't need to know which backend is active.
"""

from __future__ import annotations


class LLMClient:
    """Thin wrapper around any supported LLM provider."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        base_url: str = "",
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self._base_url = base_url

        if self.provider in ("openai", "ollama"):
            from openai import OpenAI
            kwargs: dict = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)

        elif self.provider == "claude":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")

        elif self.provider == "gemini":
            try:
                from google.genai import Client
                self._client = Client(api_key=api_key)
            except ImportError:
                raise ImportError("google-genai package not installed. Run: pip install google-genai")

        else:
            raise ValueError(f"Unsupported provider for committee: '{provider}'")

    @property
    def endpoint(self) -> str:
        """Human-readable endpoint string for error diagnostics."""
        if self.provider in ("openai", "ollama"):
            return str(getattr(self._client, "base_url", self._base_url or "api.openai.com"))
        return f"{self.provider} API"

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 400,
        temperature: float = 0.2,
        json_mode: bool = True,
    ) -> tuple[str, dict]:
        """
        Make a chat completion.  Returns (response_text, usage_dict).
        json_mode requests JSON output where the provider supports it.
        """
        if self.provider in ("openai", "ollama"):
            kwargs: dict = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self._client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or "{}"
            usage = {
                "prompt": resp.usage.prompt_tokens,
                "completion": resp.usage.completion_tokens,
            }
            return text, usage

        elif self.provider == "claude":
            # Anthropic uses a separate system param; extract it from messages
            system = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            user_msgs = [m for m in messages if m["role"] != "system"]
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": user_msgs,
            }
            if system:
                kwargs["system"] = system
            resp = self._client.messages.create(**kwargs)
            text = resp.content[0].text
            usage = {
                "prompt": resp.usage.input_tokens,
                "completion": resp.usage.output_tokens,
            }
            return text, usage

        elif self.provider == "gemini":
            parts = [m["content"] for m in messages]
            prompt = "\n\n".join(parts)
            resp = self._client.models.generate_content(
                model=self.model, contents=prompt
            )
            return resp.text, {}

        return "{}", {}

    def chat_prose(
        self,
        messages: list[dict],
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> tuple[str, dict]:
        """
        Request a prose response in a provider-agnostic way.  Returns the
        reasoning/analysis text and usage dict.

        OpenAI: injects a JSON schema into the system message and uses
                json_mode=True — returns data["reasoning"] from the parsed JSON.
        Ollama:  plain text with json_mode=False.
        Claude/Gemini: plain text directly.
        """
        if self.provider == "openai":
            import json as _json
            _JSON_NOTE = (
                "\n\nIMPORTANT: Respond ONLY with valid JSON in exactly this format:\n"
                '{"action": "BUY|SELL|HOLD|STRONG_BUY|STRONG_SELL", '
                '"confidence": 0.0-1.0, '
                '"reasoning": "your complete 2-3 sentence analysis"}'
            )
            mod = [
                {**m, "content": m["content"] + _JSON_NOTE} if m["role"] == "system" else m
                for m in messages
            ]
            text, usage = self.chat(mod, max_tokens=max_tokens, temperature=temperature, json_mode=True)
            try:
                data = _json.loads(text)
                return str(data.get("reasoning", text)), usage
            except Exception:
                return text, usage

        elif self.provider == "ollama":
            return self.chat(messages, max_tokens=max_tokens,
                             temperature=temperature, json_mode=False)

        else:  # claude, gemini
            return self.chat(messages, max_tokens=max_tokens,
                             temperature=temperature, json_mode=False)
