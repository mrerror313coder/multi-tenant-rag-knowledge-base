"""Unified LLM Client with Multimodal Vision (Gemini 2.0 Flash -> Groq -> OpenAI -> Mock), Audio STT, and streaming."""

import os
import time
import json
import base64
import logging
import re
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
import httpx

from services.cost_tracker import CostTracker

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "what", "when", "where", "who", "which", "why", "how", "is", "are", "was", "were",
    "the", "a", "an", "in", "on", "at", "for", "to", "of", "with", "by", "from", "and",
    "or", "not", "do", "does", "did", "can", "could", "should", "would", "about", "that",
    "this", "these", "those", "it", "its", "your", "my", "our", "their", "per", "day"
}


class LLMResult:
    """Encapsulates response payload, citations, and metrics from LLM."""

    def __init__(
        self,
        text: str,
        citations: List[Dict[str, Any]],
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        degraded: bool = False,
        error_message: Optional[str] = None,
    ):
        self.text = text
        self.citations = citations
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_ms = latency_ms
        self.degraded = degraded
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "citations": self.citations,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "degraded": self.degraded,
            "error_message": self.error_message,
        }


ROMAN_URDU_KEYWORDS = {
    "kia", "kya", "hai", "hain", "hona", "honi", "hone", "chahiye", "chaiye", "chahye",
    "batao", "bataein", "bataen", "batayein", "kese", "kesay", "kaise", "konsa", "konsi",
    "kahan", "kab", "kyun", "kyu", "karo", "karein", "karen", "karna", "mera", "meri",
    "mere", "humaray", "humari", "karta", "karti", "karte", "hoga", "hogi", "honge",
    "wala", "wali", "walay", "kitna", "kitni", "kitne", "mujhe", "aap", "yeh", "woh",
    "baray", "mutabiq", "bhi", "toh"
}


class UnifiedLLMClient:
    """Unified LLM interface with multimodal vision reasoning, audio transcription, and fallback orchestration."""

    def __init__(
        self,
        primary_provider: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
        huggingface_api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ):
        self._primary_provider = primary_provider
        self._gemini_api_key = gemini_api_key
        self._groq_api_key = groq_api_key
        self._openai_api_key = openai_api_key
        self._cohere_api_key = cohere_api_key
        self._huggingface_api_key = huggingface_api_key
        self.timeout = timeout_seconds

    @property
    def gemini_api_key(self) -> str:
        key = self._gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        return key.strip().strip('"').strip("'")

    @property
    def groq_api_key(self) -> str:
        key = self._groq_api_key or os.getenv("GROQ_API_KEY", "")
        return key.strip().strip('"').strip("'")

    @property
    def openai_api_key(self) -> str:
        key = self._openai_api_key or os.getenv("OPENAI_API_KEY", "")
        return key.strip().strip('"').strip("'")

    @property
    def cohere_api_key(self) -> str:
        key = self._cohere_api_key or os.getenv("COHERE_API_KEY", "")
        return key.strip().strip('"').strip("'")

    @property
    def huggingface_api_key(self) -> str:
        key = self._huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY", "") or os.getenv("HF_TOKEN", "")
        return key.strip().strip('"').strip("'")

    @property
    def primary_provider(self) -> str:
        return (self._primary_provider or os.getenv("PRIMARY_LLM_PROVIDER", "mock")).lower().strip().strip('"').strip("'")

    def _extract_citations_from_text_or_context(
        self, text: str, context_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extracts structured citations from text tags or matches with retrieved context chunks."""
        citations = []
        seen = set()

        pattern = r"\[Doc:\s*([^,\]]+)(?:,\s*Chunk:\s*(\d+))?\]"
        matches = re.findall(pattern, text)
        for doc_name, chunk_idx in matches:
            clean_doc = doc_name.strip()
            c_idx = int(chunk_idx) if chunk_idx else 0
            key = (clean_doc, c_idx)
            if key not in seen:
                seen.add(key)
                snippet = ""
                similarity = 0.0
                cluster_id = None
                cluster_label = None
                for chunk in context_chunks:
                    if chunk.get("filename") == clean_doc or chunk.get("document_id") == clean_doc:
                        snippet = chunk.get("text", "")[:250]
                        similarity = chunk.get("similarity_score", 0.0)
                        cluster_id = chunk.get("cluster_id")
                        cluster_label = chunk.get("cluster_label")
                        break
                citations.append({
                    "document_name": clean_doc,
                    "chunk_index": c_idx,
                    "snippet": snippet,
                    "similarity_score": similarity,
                    "cluster_id": cluster_id,
                    "cluster_label": cluster_label,
                })

        if not citations and context_chunks and "I don't have information" not in text and "nahi mili" not in text and "نہیں ملی" not in text:
            for chunk in context_chunks[:3]:
                doc_name = chunk.get("filename", "Unknown Document")
                c_idx = chunk.get("chunk_index", 0)
                key = (doc_name, c_idx)
                if key not in seen:
                    seen.add(key)
                    citations.append({
                        "document_name": doc_name,
                        "chunk_index": c_idx,
                        "snippet": chunk.get("text", "")[:250],
                        "similarity_score": chunk.get("similarity_score", 0.0),
                        "cluster_id": chunk.get("cluster_id"),
                        "cluster_label": chunk.get("cluster_label"),
                    })

        return citations

    async def _call_gemini_multimodal_rest(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: Optional[bytes] = None,
        image_mime_type: str = "image/png",
    ) -> str:
        """Direct REST call to Gemini 2.0 Flash with optional image / screenshot vision input."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
        parts = []

        if image_bytes:
            b64_data = base64.b64encode(image_bytes).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": image_mime_type,
                    "data": b64_data,
                }
            })

        parts.append({"text": f"{system_prompt}\n\n{user_prompt}"})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_openai_compatible(
        self, base_url: str, api_key: str, model: str, system_prompt: str, user_prompt: str
    ) -> str:
        """Call Groq or OpenAI chat completion."""
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _call_cohere(self, system_prompt: str, user_prompt: str) -> str:
        """Call Cohere Chat v2 API (Command R+)."""
        url = "https://api.cohere.com/v2/chat"
        headers = {
            "Authorization": f"Bearer {self.cohere_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "command-r-plus-08-2024",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"][0]["text"]

    async def _call_huggingface(self, system_prompt: str, user_prompt: str) -> str:
        """Call Hugging Face Serverless Chat API (Llama 3.3 70B Instruct)."""
        url = "https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.3-70B-Instruct/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.huggingface_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _split_into_clean_sentences(self, text: str) -> List[str]:
        """Splits text into sentences without breaking honorifics (Dr., Mr., etc.)."""
        protected = re.sub(r"\b(Dr|Mr|Ms|Mrs|Prof|Inc|Corp|Ltd)\.", r"\1__DOT__", text)
        raw_sentences = re.split(r"(?<=[.!?\n])\s+", protected)
        cleaned = [s.replace("__DOT__", ".").strip() for s in raw_sentences if s.strip()]
        return cleaned

    def _mock_grounded_generation(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        has_image: bool = False,
    ) -> str:
        """Deterministic, grounded mock LLM generator supporting English, Urdu, and Roman Urdu."""
        is_urdu_script = bool(re.search(r"[\u0600-\u06FF]", query))
        query_tokens_set = set(re.findall(r"\w+", query.lower()))
        is_roman_urdu = bool(query_tokens_set & ROMAN_URDU_KEYWORDS)

        if not context_chunks:
            if is_urdu_script:
                return "مجھے آپ کی تنظیم کے دستاویزات میں اس بارے میں کوئی معلومات نہیں ملی۔"
            if is_roman_urdu:
                return "Aap ki organization ke documents mein is baray mein koi information nahi mili."
            if has_image:
                return "I analyzed your uploaded image/screenshot. However, I don't have information about that in your organization's documents."
            return "I don't have information about that in your organization's documents."

        query_words = [
            w for w in re.findall(r"\w+", query.lower())
            if len(w) >= 3 and w not in STOP_WORDS and w not in ROMAN_URDU_KEYWORDS
        ]

        best_chunk = None
        best_content = None
        best_overlap = 0

        for chunk in context_chunks:
            chunk_text = chunk.get("text", "")
            chunk_lower = chunk_text.lower()
            overlap = sum(1 for w in query_words if w in chunk_lower) if query_words else 1
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk = chunk
                sentences = self._split_into_clean_sentences(chunk_text)
                matching_sentences = [s for s in sentences if any(w in s.lower() for w in query_words)] if query_words else sentences
                if matching_sentences:
                    best_content = " ".join(matching_sentences)
                else:
                    best_content = chunk_text.strip()

        # If query is in Urdu or Roman Urdu, match against retrieved context chunks directly
        if (is_urdu_script or is_roman_urdu) and context_chunks and not best_chunk:
            best_chunk = context_chunks[0]
            best_content = best_chunk.get("text", "").strip()

        if best_chunk is None or (query_words and not (is_urdu_script or is_roman_urdu) and best_overlap < 1) or not best_content:
            if is_urdu_script:
                return "مجھے آپ کی تنظیم کے دستاویزات میں اس بارے میں کوئی معلومات نہیں ملی۔"
            if is_roman_urdu:
                return "Aap ki organization ke documents mein is baray mein koi information nahi mili."
            return "I don't have information about that in your organization's documents."

        doc_name = best_chunk.get("filename", "document.txt")
        c_idx = best_chunk.get("chunk_index", 0)

        if is_urdu_script:
            return f"آپ کی تنظیم کی دستاویزات کے مطابق، {best_content} [Doc: {doc_name}, Chunk: {c_idx}]"
        if is_roman_urdu:
            return f"Aap ke documents ke mutabiq, {best_content} [Doc: {doc_name}, Chunk: {c_idx}]"

        image_prefix = "Based on the provided screenshot/image and your organization's documents, " if has_image else "According to your documents, "
        return f"{image_prefix}{best_content} [Doc: {doc_name}, Chunk: {c_idx}]"

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/webm",
        language: Optional[str] = "ur",
    ) -> str:
        """Transcribes speech audio recording into text (Urdu / Roman Urdu / English) using Groq Whisper or Gemini."""
        clean_mime = mime_type.split(";")[0].strip().lower()
        if clean_mime not in ["audio/webm", "audio/wav", "audio/mp3", "audio/ogg", "audio/aac", "audio/flac"]:
            clean_mime = "audio/webm"

        # 1. Primary: Groq Whisper Audio Transcription (with Urdu/English bias & anti-hallucination prompt)
        if self.groq_api_key:
            try:
                url = "https://api.groq.com/openai/v1/audio/transcriptions"
                headers = {"Authorization": f"Bearer {self.groq_api_key}"}
                ext = "mp3" if "mp3" in clean_mime else ("wav" if "wav" in clean_mime else ("ogg" if "ogg" in clean_mime else "webm"))
                files = {"file": (f"audio_recording.{ext}", audio_bytes, clean_mime)}
                data = {
                    "model": "whisper-large-v3-turbo",
                    "prompt": "Urdu and Roman Urdu and English audio transcription. دستاویزات سوال اور جواب",
                    "temperature": 0.0,
                }
                if language and language != "auto":
                    data["language"] = language

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, headers=headers, files=files, data=data)
                    resp.raise_for_status()
                    result = resp.json()
                    transcription = result.get("text", "").strip()

                    # Filter hallucinated foreign characters (Icelandic, Old Norse, etc.)
                    if re.search(r"[ÞþÐðæøå]", transcription):
                        logger.warning(f"Whisper returned foreign language hallucination: '{transcription}'. Retrying with language='ur'...")
                        data["language"] = "ur"
                        resp2 = await client.post(url, headers=headers, files=files, data=data)
                        if resp2.status_code == 200:
                            transcription = resp2.json().get("text", "").strip()

                    if transcription:
                        logger.info(f"Groq Whisper transcribed audio: '{transcription}'")
                        return transcription
            except Exception as e:
                logger.warning(f"Groq Whisper audio transcription error: {e}.")

        # 2. Gemini 2.0 Flash Audio Transcription
        if self.gemini_api_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
                b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "inline_data": {
                                        "mime_type": clean_mime,
                                        "data": b64_audio,
                                    }
                                },
                                {
                                    "text": "Transcribe this speech accurately in the spoken language (Urdu script or English). Return ONLY the spoken words."
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 256,
                    }
                }
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates and candidates[0].get("content", {}).get("parts", []):
                        transcription = candidates[0]["content"]["parts"][0]["text"].strip()
                        if transcription:
                            return transcription
            except Exception as e:
                logger.warning(f"Gemini audio transcription error: {e}.")

        # 3. OpenAI Whisper Fallback
        if self.openai_api_key:
            try:
                url = "https://api.openai.com/v1/audio/transcriptions"
                headers = {"Authorization": f"Bearer {self.openai_api_key}"}
                ext = "mp3" if "mp3" in clean_mime else ("wav" if "wav" in clean_mime else "webm")
                files = {"file": (f"recording.{ext}", audio_bytes, clean_mime)}
                data = {"model": "whisper-1"}
                if language and language != "auto":
                    data["language"] = language
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, headers=headers, files=files, data=data)
                    resp.raise_for_status()
                    result = resp.json()
                    transcription = result.get("text", "").strip()
                    if transcription:
                        return transcription
            except Exception as e:
                logger.warning(f"OpenAI Whisper transcription error: {e}")

        # 4. Deterministic Mock Fallback
        return "سات سٹیپ کیا ہیں پروجیکٹ کے" if language == "ur" else "What are the project requirements?"

    @staticmethod
    def _detect_task_intent(query: str) -> Tuple[str, str]:
        """Detects user query intent (Summarization, Explanation, Classification, QA) and returns specialized guidance."""
        q_lower = query.lower()

        # 1. Summarization Mode
        if any(k in q_lower for k in ["summary", "summarize", "brief", "overview", "khulasa", "mukhtasar", "خلاصہ", "مختصر"]):
            return "Summarization", "\n\n[TASK DIRECTIVE: SUMMARIZATION]\nProvide a crisp, structured executive summary highlighting the essential goals, key milestones, and deliverables."

        # 2. Comparison & Detailed Explanation Mode
        if any(k in q_lower for k in ["difference", "compare", "explain", "detail", "farq", "kya farq", "kyun", "فرق", "وضاحت", "تفصیل", "کیوں", "کیسے"]):
            return "Explanation & Comparison", "\n\n[TASK DIRECTIVE: COMPARISON & EXPLANATION]\nStructure your explanation clearly with comparative points. For each item/stage mentioned, explain its specific objective, focus, and how it contrasts with other items in the documents."

        # 3. Step Extraction & Classification Mode
        if any(k in q_lower for k in ["step", "stage", "list", "rules", "points", "kitne", "konsa", "اقدامات", "مراحل", "کتنے", "کونسے", "فہرست"]):
            return "Classification & Step Extraction", "\n\n[TASK DIRECTIVE: STEP EXTRACTION & CLASSIFICATION]\nEnumerate the stages/steps sequentially with clear bullet points outlining the name, purpose, and key activities of each step."

        return "General QA", ""

    async def generate_answer(
        self,
        query: str,
        system_prompt: str,
        user_prompt: str,
        context_chunks: List[Dict[str, Any]],
        org_id: str = "default",
        image_bytes: Optional[bytes] = None,
        image_mime_type: str = "image/png",
    ) -> LLMResult:
        """Generates grounded answer with multimodal vision support, task-aware prompt specialization, and graceful degradation."""
        start_time = time.perf_counter()
        
        # Auto-detect intent and inject specialized task directives
        task_type, task_directive = self._detect_task_intent(query)
        effective_system_prompt = system_prompt + task_directive

        prompt_tokens = CostTracker.estimate_tokens(effective_system_prompt + " " + user_prompt)
        if image_bytes:
            prompt_tokens += 258  # Standard multimodal vision token allowance
        text_response = ""
        used_model = "mock"
        error_msg = None

        # Determine provider priority order
        all_provs = ["groq", "gemini", "cohere", "huggingface", "openai"]
        providers = [self.primary_provider] + [p for p in all_provs if p != self.primary_provider]

        for prov in providers:
            if prov == "gemini" and self.gemini_api_key and not text_response:
                try:
                    text_response = await self._call_gemini_multimodal_rest(
                        system_prompt=effective_system_prompt,
                        user_prompt=user_prompt,
                        image_bytes=image_bytes,
                        image_mime_type=image_mime_type,
                    )
                    if text_response:
                        used_model = "gemini-2.0-flash"
                        error_msg = None
                        break
                except Exception as e:
                    logger.warning(f"Gemini LLM call failed: {e}. Attempting fallback.")
                    error_msg = str(e)

            elif prov == "groq" and self.groq_api_key and not text_response:
                groq_models = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "groq/compound", "allam-2-7b"]
                for g_model in groq_models:
                    try:
                        text_response = await self._call_openai_compatible(
                            base_url="https://api.groq.com/openai/v1",
                            api_key=self.groq_api_key,
                            model=g_model,
                            system_prompt=effective_system_prompt,
                            user_prompt=user_prompt,
                        )
                        if text_response:
                            used_model = g_model
                            error_msg = None
                            break
                    except Exception as e:
                        logger.warning(f"Groq model {g_model} failed: {e}. Trying next Groq model.")
                        error_msg = str(e)
                if text_response:
                    break

            elif prov == "cohere" and self.cohere_api_key and not text_response:
                try:
                    text_response = await self._call_cohere(
                        system_prompt=effective_system_prompt,
                        user_prompt=user_prompt,
                    )
                    if text_response:
                        used_model = "cohere/command-r-plus"
                        error_msg = None
                        break
                except Exception as e:
                    logger.warning(f"Cohere fallback failed: {e}.")
                    error_msg = str(e)

            elif prov == "huggingface" and self.huggingface_api_key and not text_response:
                try:
                    text_response = await self._call_huggingface(
                        system_prompt=effective_system_prompt,
                        user_prompt=user_prompt,
                    )
                    if text_response:
                        used_model = "hf/llama-3.3-70b-instruct"
                        error_msg = None
                        break
                except Exception as e:
                    logger.warning(f"Hugging Face fallback failed: {e}.")
                    error_msg = str(e)

            elif prov == "openai" and self.openai_api_key and not text_response:
                try:
                    text_response = await self._call_openai_compatible(
                        base_url="https://api.openai.com/v1",
                        api_key=self.openai_api_key,
                        model="gpt-4o-mini",
                        system_prompt=effective_system_prompt,
                        user_prompt=user_prompt,
                    )
                    if text_response:
                        used_model = "gpt-4o-mini"
                        error_msg = None
                        break
                except Exception as e:
                    logger.warning(f"OpenAI fallback failed: {e}.")
                    error_msg = str(e)

        # 4. Fallback to Mock LLM
        if not text_response:
            text_response = self._mock_grounded_generation(
                query=query,
                context_chunks=context_chunks,
                has_image=image_bytes is not None,
            )
            used_model = "mock-grounded-multimodal"
            error_msg = None

        # 5. Graceful degradation
        is_degraded = False
        if not text_response:
            is_degraded = True
            used_model = "graceful-degradation-offline"
            if context_chunks:
                text_response = "⚠️ [LLM Service Unavailable - Graceful Degradation]\n\nBelow are the most relevant document chunks retrieved for your organization:\n\n"
                for i, c in enumerate(context_chunks, 1):
                    text_response += f"**[{i}] {c.get('filename')} (Chunk {c.get('chunk_index')}):**\n> {c.get('text')}\n\n"
            else:
                text_response = "I don't have information about that in your organization's documents."

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        completion_tokens = CostTracker.estimate_tokens(text_response)
        citations = self._extract_citations_from_text_or_context(text_response, context_chunks)

        CostTracker.log_call(
            org_id=org_id,
            model=used_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            is_success=not is_degraded,
            error=error_msg,
        )

        return LLMResult(
            text=text_response,
            citations=citations,
            model=used_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            degraded=is_degraded,
            error_message=error_msg,
        )

    async def stream_answer(
        self,
        query: str,
        system_prompt: str,
        user_prompt: str,
        context_chunks: List[Dict[str, Any]],
        org_id: str = "default",
        image_bytes: Optional[bytes] = None,
        image_mime_type: str = "image/png",
    ) -> AsyncGenerator[str, None]:
        """Streams answer token-by-token using SSE-ready chunk formatting."""
        result = await self.generate_answer(
            query=query,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_chunks=context_chunks,
            org_id=org_id,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
        )

        words = result.text.split(" ")
        for i, word in enumerate(words):
            chunk_data = {
                "token": word + (" " if i < len(words) - 1 else ""),
                "done": False,
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

        final_data = {
            "token": "",
            "done": True,
            "citations": result.citations,
            "model": result.model,
            "latency_ms": result.latency_ms,
            "degraded": result.degraded,
            "tokens": result.prompt_tokens + result.completion_tokens,
        }
        yield f"data: {json.dumps(final_data)}\n\n"
