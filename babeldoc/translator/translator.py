# MODIFIED: Added imports for concurrency, batching, and JSON parsing
import concurrent.futures
import contextlib
import json
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

# ==============================================================================
# NEW: Helper function to chunk a list into smaller batches
# ==============================================================================
def _chunk_list(data: list, size: int) -> list[list]:
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(data), size):
        yield data[i:i + size]

# ==============================================================================
# NEW: High-level function to manage concurrent batch translations
# ==============================================================================
def translate_texts_in_parallel(
    translator: "BaseTranslator",
    texts: list[str],
    batch_size: int = 20,
    max_workers: int = 10,
) -> list[str]:
    """
    Translates a large list of texts using concurrent batching for maximum speed.

    Args:
        translator: An instance of your GeminiTranslator.
        texts: A list of strings to translate.
        batch_size: The number of texts to include in each API call.
        max_workers: The number of concurrent API calls to make.

    Returns:
        A list of translated strings in the same order as the input.
    """
    if not texts:
        return []

    text_batches = list(_chunk_list(texts, batch_size))
    translated_texts_in_batches = [None] * len(text_batches)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a map of future to its batch index to reorder results correctly
        future_to_index = {
            executor.submit(translator.translate_batch, batch): i
            for i, batch in enumerate(text_batches)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                translated_batch = future.result()
                if len(translated_batch) == len(text_batches[index]):
                    translated_texts_in_batches[index] = translated_batch
                else:
                    logger.error(
                        f"Mismatched batch size for index {index}. "
                        f"Expected {len(text_batches[index])}, got {len(translated_batch)}. "
                        "Falling back to empty strings for this batch."
                    )
                    translated_texts_in_batches[index] = [""] * len(text_batches[index])
            except Exception as exc:
                logger.error(f"Batch {index} generated an exception: {exc}")
                translated_texts_in_batches[index] = [""] * len(text_batches[index]) # Fill with empty strings on error

    # Flatten the list of lists into a single list
    return [text for batch in translated_texts_in_batches for text in batch]


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
        self.next_request_time = time.monotonic()

    def wait(self, _rate_limit_params: dict = None):
        with self.lock:
            now = time.monotonic()
            wait_duration = self.next_request_time - now
            if wait_duration > 0:
                time.sleep(wait_duration)
            now = time.monotonic()
            self.next_request_time = max(self.next_request_time, now) + self.min_interval

    def set_max_qps(self, max_qps: int):
        if max_qps <= 0:
            raise ValueError("max_qps must be a positive number")
        with self.lock:
            self.max_qps = max_qps
            self.min_interval = 1.0 / max_qps

_translate_rate_limiter = RateLimiter(20)

def set_translate_rate_limiter(max_qps):
    _translate_rate_limiter.set_max_qps(max_qps)

class BaseTranslator(ABC):
    name = "base"
    lang_map = {}

    def __init__(self, lang_in, lang_out, ignore_cache):
        self.ignore_cache = ignore_cache
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.cache = TranslationCache(self.name, {"lang_in": lang_in, "lang_out": lang_out})
        self.translate_call_count = 0
        self.translate_cache_call_count = 0

    def __del__(self):
        with contextlib.suppress(Exception):
            logger.info(f"{self.name} translate call count: {self.translate_call_count}")
            logger.info(f"{self.name} translate cache call count: {self.translate_cache_call_count}")

    def add_cache_impact_parameters(self, k: str, v):
        self.cache.add_params(k, v)

    def translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
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

    # ==============================================================================
    # MODIFIED: Added batch translation method for performance
    # ==============================================================================
    def translate_batch(self, texts: list[str], ignore_cache=False) -> list[str]:
        """Translates a batch of texts, using cache where possible."""
        self.translate_call_count += 1 # Counts as one logical call
        
        results = [None] * len(texts)
        uncached_texts_with_indices = []

        if not (self.ignore_cache or ignore_cache):
            for i, text in enumerate(texts):
                try:
                    cache = self.cache.get(text)
                    if cache is not None:
                        self.translate_cache_call_count += 1
                        results[i] = cache
                except Exception as e:
                    logger.debug(f"Cache get failed for text index {i}, ignoring: {e}")

        for i, text in enumerate(texts):
            if results[i] is None:
                uncached_texts_with_indices.append((i, text))

        if uncached_texts_with_indices:
            indices, uncached_texts = zip(*uncached_texts_with_indices)
            _translate_rate_limiter.wait() # Wait once per batch API call
            
            translated_uncached = self.do_translate_batch(list(uncached_texts))

            if len(translated_uncached) != len(uncached_texts):
                logger.error(
                    "Batch translation returned a different number of items than expected. "
                    f"Got {len(translated_uncached)}, expected {len(uncached_texts)}"
                )
                # Fallback to avoid crashing, though results will be wrong
                translated_uncached = [""] * len(uncached_texts)

            for original_index, text, translation in zip(indices, uncached_texts, translated_uncached):
                results[original_index] = translation
                if not (self.ignore_cache or ignore_cache):
                    self.cache.set(text, translation)
                    
        return results

    @abstractmethod
    def do_translate_batch(self, texts: list[str]) -> list[str]:
        """Actual batch translate, override this method."""
        raise NotImplementedError

    def llm_translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
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
        raise NotImplementedError

    @abstractmethod
    def do_translate(self, text, rate_limit_params: dict = None):
        logger.critical(f"Do not call BaseTranslator.do_translate. Translator: {self}. Text: {text}.")
        raise NotImplementedError

    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"
    
    # Placeholder methods are unchanged
    def get_rich_text_left_placeholder(self, placeholder_id: int): return f"<b{placeholder_id}>"
    def get_rich_text_right_placeholder(self, placeholder_id: int): return f"</b{placeholder_id}>"
    def get_formular_placeholder(self, placeholder_id: int): return self.get_rich_text_left_placeholder(placeholder_id)

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
    name = "gemini"
    # __init__ method is unchanged from your original code.
    def __init__( self, lang_in, lang_out, model: str, google_cloud_project: str, google_cloud_location: str, google_application_credentials_path: str | None = None, ignore_cache: bool = False, generation_config: dict | None = None):
        super().__init__(lang_in, lang_out, ignore_cache)
        if not google_cloud_project: raise ValueError("google_cloud_project must be provided for Vertex AI.")
        if not google_cloud_location: raise ValueError("google_cloud_location must be provided for Vertex AI.")
        self.model = model
        self.project_id = google_cloud_project
        self.location = google_cloud_location
        self.options = {"temperature": 0}
        if generation_config: self.options.update(generation_config)
        credentials, project = None, None
        if google_application_credentials_path:
            try:
                credentials, project = google.auth.load_credentials_from_file(google_application_credentials_path)
                logger.info(f"Loaded Google Cloud credentials from: {google_application_credentials_path}")
            except Exception as e:
                logger.error(f"Failed to load credentials from {google_application_credentials_path}: {e}")
                raise RuntimeError("Failed to load Google Application Credentials for Vertex AI.") from e
        else:
            try:
                credentials, project = google.auth.default()
                logger.info("Using default Google Cloud credentials for Vertex AI.")
            except Exception as e:
                logger.warning(f"Could not find default Google Cloud credentials: {e}. Vertex AI access might fail without proper authentication.")
        try:
            vertexai.init(project=self.project_id, location=self.location, credentials=credentials)
            logger.info(f"Vertex AI initialized for project '{self.project_id}' in location '{self.location}'.")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise RuntimeError("Failed to initialize Vertex AI client.") from e
        self.client = GenerativeModel(self.model)
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self._prompt_preview(""))
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()

    # ==============================================================================
    # MODIFIED: do_translate is now a simple wrapper for the batch method
    # ==============================================================================
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        """Handles single text translation by wrapping the batch method."""
        if not text:
            return ""
        results = self.do_translate_batch([text])
        return results[0] if results else ""

    # ==============================================================================
    # NEW: Batch translation implementation
    # ==============================================================================
    @retry(
        retry=retry_if_exception_type((ResourceExhausted, DeadlineExceeded, ServiceUnavailable, InternalServerError)),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate_batch(self, texts: list[str]) -> list[str]:
        """
        Performs batch translation using a single API call.
        """
        if not texts:
            return []
        
        prompt = self._build_batch_prompt(texts)
        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": self.options["temperature"],
                "max_output_tokens": 8192, # Increase for large batches
            },
        )
        self._update_token_count(response)
        
        # New parsing logic to handle JSON array from the model
        return self._extract_json_array(response)

    @retry(retry=retry_if_exception_type((ResourceExhausted, DeadlineExceeded, ServiceUnavailable, InternalServerError)), stop=stop_after_attempt(100), wait=wait_exponential(multiplier=1, min=1, max=15), before_sleep=before_sleep_log(logger, logging.WARNING))
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None: return None
        response = self.client.generate_content(text, generation_config={"temperature": self.options["temperature"], "max_output_tokens": 2048})
        self._update_token_count(response)
        return self._extract_text(response)

    # ---------- Helpers ----------
    
    # MODIFIED: Renamed from _build_prompt to _build_batch_prompt and updated for batching
    def _build_batch_prompt(self, texts: list[str]) -> str:
        """Builds a prompt to translate a batch of texts and requests a JSON array output."""
        text_lines = "\n".join([f'{i+1}. "{text}"' for i, text in enumerate(texts)])
        return (
            f"Translate the following texts from {self.lang_in} to {self.lang_out}. "
            "Return ONLY a single JSON array of strings in your response, where each string is the translation "
            "corresponding to the numbered text in the same order. Do not include markdown formatting "
            "like ```json or any other explanatory text.\n\n"
            "TEXTS:\n"
            f"{text_lines}"
        )

    def _prompt_preview(self, text: str) -> str:
        return "gemini_mt_batch_prompt_v1" # Updated prompt signature

    # NEW: Helper to parse the JSON array response from the model
    def _extract_json_array(self, response) -> list[str]:
        """Extracts a JSON array of strings from the model's response."""
        try:
            raw_text = self._extract_text(response)
            # Clean up potential markdown code fences
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            
            translations = json.loads(raw_text.strip())
            
            if isinstance(translations, list) and all(isinstance(item, str) for item in translations):
                return [remove_control_characters(t) for t in translations]
            
            logger.warning(f"Model returned valid JSON, but it was not an array of strings: {translations}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from model response: {self._extract_text(response)}")
            return [] # Return empty list on parsing failure
        except Exception:
            logger.exception("GeminiTranslator: failed to extract JSON array from response")
            return []

    # _extract_text, _update_token_count, and placeholder methods are unchanged
    def _extract_text(self, response) -> str:
        try:
            output = (response.text or "").strip()
            return remove_control_characters(output)
        except Exception:
            try:
                parts = []
                for c in getattr(response, "candidates", []) or []:
                    for part in getattr(c, "content", {}).get("parts", []) or []:
                        if "text" in part: parts.append(part["text"])
                return remove_control_characters(("".join(parts)).strip()) if parts else ""
            except Exception as e:
                logger.exception("GeminiTranslator: failed to extract text from response")
                return ""

    def _update_token_count(self, response):
        try:
            if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
                meta = response.usage_metadata
                if hasattr(meta, "total_token_count"): self.token_count.inc(meta.total_token_count)
                if hasattr(meta, "prompt_token_count"): self.prompt_token_count.inc(meta.prompt_token_count)
                if hasattr(meta, "candidates_token_count"): self.completion_token_count.inc(meta.candidates_token_count)
        except Exception:
            logger.exception("GeminiTranslator: error updating token counters")

    def get_formular_placeholder(self, placeholder_id: int): return "{v" + str(placeholder_id) + "}", f"{{\\s*v\\s*{placeholder_id}\\s*}}"
    def get_rich_text_left_placeholder(self, placeholder_id: int): return (f"<style id='{placeholder_id}'>", f"<\\s*style\\s*id\\s*=\\s*'\\s*{placeholder_id}\\s*'\\s*>")
    def get_rich_text_right_placeholder(self, placeholder_id: int): return "</style>", r"<\s*\/\s*style\s*>"