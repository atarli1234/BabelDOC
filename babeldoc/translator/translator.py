import contextlib
import logging
import threading
import time
import unicodedata

from abc import ABC
from abc import abstractmethod

import httpx
import openai
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from babeldoc.translator.cache import TranslationCache
from babeldoc.utils.atomic_integer import AtomicInteger

# Correct imports for Vertex AI
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded, ServiceUnavailable, InternalServerError
import google.auth
import google.auth.transport.requests

logger = logging.getLogger(__name__)


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class RateLimiter:
    """
    A rate limiter using the leaky bucket algorithm to ensure a smooth, constant rate of requests.
    This implementation is thread-safe and robust against system clock changes.
    """

    def __init__(self, max_qps: int):
        if max_qps <= 0:
            raise ValueError("max_qps must be a positive number")
        self.max_qps = max_qps
        self.min_interval = 1.0 / max_qps
        self.lock = threading.Lock()
        # Use monotonic time to prevent issues with system time changes
        self.next_request_time = time.monotonic()

    def wait(self, _rate_limit_params: dict = None):
        """
        Blocks until the next request can be processed, ensuring the rate limit is not exceeded.
        """
        with self.lock:
            now = time.monotonic()

            wait_duration = self.next_request_time - now
            if wait_duration > 0:
                time.sleep(wait_duration)

            # Update the next allowed request time.
            # If the limiter has been idle, the next request should start from 'now'.
            now = time.monotonic()
            self.next_request_time = (
                max(self.next_request_time, now) + self.min_interval
            )

    def set_max_qps(self, max_qps: int):
        """
        Updates the maximum queries per second. This operation is thread-safe.
        """
        if max_qps <= 0:
            raise ValueError("max_qps must be a positive number")
        with self.lock:
            self.max_qps = max_qps
            self.min_interval = 1.0 / max_qps


_translate_rate_limiter = RateLimiter(5)


def set_translate_rate_limiter(max_qps):
    _translate_rate_limiter.set_max_qps(max_qps)


class BaseTranslator(ABC):
    # Due to cache limitations, name should be within 20 characters.
    # cache.py: translate_engine = CharField(max_length=20)
    name = "base"
    lang_map = {}

    def __init__(self, lang_in, lang_out, ignore_cache):
        self.ignore_cache = ignore_cache
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out

        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": lang_in,
                "lang_out": lang_out,
            },
        )

        self.translate_call_count = 0
        self.translate_cache_call_count = 0

    def __del__(self):
        with contextlib.suppress(Exception):
            logger.info(
                f"{self.name} translate call count: {self.translate_call_count}"
            )
            logger.info(
                f"{self.name} translate cache call count: {self.translate_cache_call_count}",
            )

    def add_cache_impact_parameters(self, k: str, v):
        """
        Add parameters that affect the translation quality to distinguish the translation effects under different parameters.
        :param k: key
        :param v: value
        """
        self.cache.add_params(k, v)

    def translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        self.translate_call_count += 1
        if not (self.ignore_cache or ignore_cache):
            try:
                cache = self.cache.get(text)
                if cache is not None:
                    self.translate_cache_call_count += 1
                    return cache
            except Exception as e:
                logger.debug(f"try get cache failed, ignore it: {e}")
        _translate_rate_limiter.wait()
        translation = self.do_translate(text, rate_limit_params)
        if not (self.ignore_cache or ignore_cache):
            self.cache.set(text, translation)
        return translation

    def llm_translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        self.translate_call_count += 1
        if not (self.ignore_cache or ignore_cache):
            try:
                cache = self.cache.get(text)
                if cache is not None:
                    self.translate_cache_call_count += 1
                    return cache
            except Exception as e:
                logger.debug(f"try get cache failed, ignore it: {e}")
        _translate_rate_limiter.wait()
        translation = self.do_llm_translate(text, rate_limit_params)
        if not (self.ignore_cache or ignore_cache):
            self.cache.set(text, translation)
        return translation

    @abstractmethod
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        raise NotImplementedError

    @abstractmethod
    def do_translate(self, text, rate_limit_params: dict = None):
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        logger.critical(
            f"Do not call BaseTranslator.do_translate. "
            f"Translator: {self}. "
            f"Text: {text}. ",
        )
        raise NotImplementedError

    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"

    def get_rich_text_left_placeholder(self, placeholder_id: int):
        return f"<b{placeholder_id}>"

    def get_rich_text_right_placeholder(self, placeholder_id: int):
        return f"</b{placeholder_id}>"

    def get_formular_placeholder(self, placeholder_id: int):
        return self.get_rich_text_left_placeholder(placeholder_id)


class OpenAITranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "openai"

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        ignore_cache=False,
        enable_json_mode_if_requested=False,
        send_dashscope_header=False,
        send_temperature=True,
    ):
        super().__init__(lang_in, lang_out, ignore_cache)
        self.options = {"temperature": 0}  # 随机采样可能会打断公式标记
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None
                ),
                timeout=60,  # Set a reasonable timeout
            ),
        )
        if send_temperature:
            self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.model = model
        self.enable_json_mode_if_requested = enable_json_mode_if_requested
        self.send_dashscope_header = send_dashscope_header
        self.send_temperature = send_temperature
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        if self.enable_json_mode_if_requested:
            self.add_cache_impact_parameters(
                "enable_json_mode_if_requested", self.enable_json_mode_if_requested
            )
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        options = {}
        if self.send_temperature:
            options.update(self.options)

        response = self.client.chat.completions.create(
            model=self.model,
            **options,
            messages=self.prompt(text),
        )
        self.update_token_count(response)
        return response.choices[0].message.content.strip()

    def prompt(self, text):
        return [
            {
                "role": "user",
                "content": f"{self.lang_in}→{self.lang_out}\n{text}",
            },
        ]

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None

        options = {}
        if self.send_temperature:
            options.update(self.options)
        if self.enable_json_mode_if_requested and rate_limit_params.get(
            "request_json_mode", False
        ):
            options["response_format"] = {"type": "json_object"}

        extra_headers = {}
        if self.send_dashscope_header:
            extra_headers["X-DashScope-DataInspection"] = (
                '{"input": "disable", "output": "disable"}'
            )

        response = self.client.chat.completions.create(
            model=self.model,
            **options,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
            extra_headers=extra_headers,
        )
        self.update_token_count(response)
        return response.choices[0].message.content.strip()

    def update_token_count(self, response):
        try:
            if response.usage and response.usage.total_tokens:
                self.token_count.inc(response.usage.total_tokens)
            if response.usage and response.usage.prompt_tokens:
                self.prompt_token_count.inc(response.usage.prompt_tokens)
            if response.usage and response.usage.completion_tokens:
                self.completion_token_count.inc(response.usage.completion_tokens)
        except Exception as e:
            logger.exception("Error updating token count")

    def get_formular_placeholder(self, placeholder_id: int):
        return "{v" + str(placeholder_id) + "}", f"{{\\s*v\\s*{placeholder_id}\\s*}}"
        return "{{" + str(placeholder_id) + "}}"

    def get_rich_text_left_placeholder(self, placeholder_id: int):
        return (
            f"<style id='{placeholder_id}'>",
            f"<\\s*style\\s*id\\s*=\\s*'\\s*{placeholder_id}\\s*'\\s*>",
        )

    def get_rich_text_right_placeholder(self, placeholder_id: int):
        return "</style>", r"<\s*\/\s*style\s*>"


class GeminiTranslator(BaseTranslator):
    """
    Google Gemini wrapper for BabelDoc, using Vertex AI.

    Configured via:
      - google_cloud_project (required)
      - google_cloud_location (required)
      - google_application_credentials_path (optional, falls back to default if not provided)
      - model (Vertex AI model resource path, e.g., "projects/PROJECT_ID/locations/LOCATION/endpoints/ENDPOINT_ID")
    """
    name = "gemini"

    def __init__(
        self,
        lang_in,
        lang_out,
        model: str, # Model is now required and expected to be a full resource path
        google_cloud_project: str, # Make required
        google_cloud_location: str, # Make required
        google_application_credentials_path: str | None = None,
        ignore_cache: bool = False,
        generation_config: dict | None = None,
    ):
        super().__init__(lang_in, lang_out, ignore_cache)

        # Ensure project and location are provided for Vertex AI
        if not google_cloud_project:
            raise ValueError("google_cloud_project must be provided for Vertex AI.")
        if not google_cloud_location:
            raise ValueError("google_cloud_location must be provided for Vertex AI.")

        self.model = model
        self.project_id = google_cloud_project
        self.location = google_cloud_location

        self.options = {
            "temperature": 0,  # deterministic; helps keep placeholders intact
        }
        if generation_config:
            self.options.update(generation_config)

        # Load credentials
        credentials, project = None, None
        if google_application_credentials_path:
            try:
                credentials, project = google.auth.load_credentials_from_file(
                    google_application_credentials_path
                )
                logger.info(f"Loaded Google Cloud credentials from: {google_application_credentials_path}")
            except Exception as e:
                logger.error(f"Failed to load credentials from {google_application_credentials_path}: {e}")
                raise RuntimeError("Failed to load Google Application Credentials for Vertex AI.") from e
        else:
            try:
                credentials, project = google.auth.default()
                logger.info("Using default Google Cloud credentials for Vertex AI.")
            except Exception as e:
                logger.warning(f"Could not find default Google Cloud credentials: {e}. "
                               "Vertex AI access might fail without proper authentication.")
                # If no credentials found, and no path was provided, Vertex AI will likely fail.

        # Initialize Vertex AI with project, location, and credentials
        try:
            logger.info(f"location: {self.location}")
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=credentials
            )
            logger.info(f"Vertex AI initialized for project '{self.project_id}' in location '{self.location}'.")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise RuntimeError("Failed to initialize Vertex AI client.") from e
        
        # Instantiate the GenerativeModel with the full model path
        # Note: The `model` parameter here is the actual model ID or path,
        # for Vertex AI it's often the full resource path like "projects/.../locations/.../endpoints/..."
        # or a public model name like "gemini-1.5-flash".
        self.client = GenerativeModel(self.model)

        # cache-impact parameters
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self._prompt_preview(""))

        # usage counters
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()

    # Retry on common 429/5xx/timeouts from Google
    @retry(
        retry=retry_if_exception_type((ResourceExhausted, DeadlineExceeded, ServiceUnavailable, InternalServerError)),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        """
        Deterministic MT-style translation: translate text to self.lang_out, output ONLY the translation.
        """
        prompt = self._build_prompt(text)
        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": self.options["temperature"],
                "max_output_tokens": 2048,
            },
        )
        self._update_token_count(response)
        return self._extract_text(response)

    @retry(
        retry=retry_if_exception_type((ResourceExhausted, DeadlineExceeded, ServiceUnavailable, InternalServerError)),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        """
        Free-form LLM call (keeps your BabelDoc contract): returns model output for arbitrary input.
        """
        if text is None:
            return None
        response = self.client.generate_content(
            text,
            generation_config={
                "temperature": self.options["temperature"],
                "max_output_tokens": 2048,
            },
        )
        self._update_token_count(response)
        return self._extract_text(response)

    # ---------- Helpers ----------

    def _build_prompt(self, text: str) -> str:
        # Match your OpenAI prompt behavior as closely as possible.
        return (
            f"{self.lang_in}→{self.lang_out}\n{text}"
        )

    def _prompt_preview(self, text: str) -> str:
        # a short stable signature to include in cache-impact parameters
        return "gemini_mt_prompt_v1"

    def _extract_text(self, response) -> str:
        # vertexai.preview.generative_models.GenerateContentResponse has .text
        try:
            output = (response.text or "").strip()
            return remove_control_characters(output)
        except Exception:
            # Fallback: stitch parts if .text is missing or fails
            try:
                parts = []
                # Access parts through the first candidate
                for c in getattr(response, "candidates", []) or []:
                    for part in getattr(c, "content", {}).get("parts", []) or []:
                        if "text" in part:
                            parts.append(part["text"])
                return remove_control_characters(("".join(parts)).strip()) if parts else ""
            except Exception as e:
                logger.exception("GeminiTranslator: failed to extract text from response")
                return ""

    def _update_token_count(self, response):
        try:
            # Vertex AI SDK provides usage metadata directly on the response
            # These attributes align with what was previously on response.usage_metadata
            if hasattr(response, "total_tokens"):
                self.token_count.inc(response.total_tokens)
            # Note: Vertex AI response might not expose prompt_token_count and candidates_token_count directly
            # at the top level like google.generativeai did, but rather within usage_metadata
            # or through count_tokens calls.
            # If these are crucial for your logging, you might need to make a separate count_tokens call.
            # For now, we'll try to get them if they exist directly on the response or a sub-attribute
            if hasattr(response, "prompt_token_count") and response.prompt_token_count is not None:
                self.prompt_token_count.inc(response.prompt_token_count)
            if hasattr(response, "candidates_token_count") and response.candidates_token_count is not None:
                self.completion_token_count.inc(response.candidates_token_count)
            elif hasattr(response, "usage_metadata") and response.usage_metadata is not None:
                meta = response.usage_metadata
                if hasattr(meta, "prompt_token_count") and meta.prompt_token_count is not None:
                    self.prompt_token_count.inc(meta.prompt_token_count)
                if hasattr(meta, "candidates_token_count") and meta.candidates_token_count is not None:
                    self.completion_token_count.inc(meta.candidates_token_count)

        except Exception:
            logger.exception("GeminiTranslator: error updating token counters from Vertex AI response")

    # Keep placeholders consistent with OpenAITranslator for downstream processors
    def get_formular_placeholder(self, placeholder_id: int):
        return "{v" + str(placeholder_id) + "}", f"{{\\s*v\\s*{placeholder_id}\\s*}}"

    def get_rich_text_left_placeholder(self, placeholder_id: int):
        return (
            f"<style id='{placeholder_id}'>",
            f"<\\s*style\\s*id\\s*=\\s*'\\s*{placeholder_id}\\s*'\\s*>",
        )

    def get_rich_text_right_placeholder(self, placeholder_id: int):
        return "</style>", r"<\s*\/\s*style\s*>"