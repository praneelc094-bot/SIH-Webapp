"""
MediKiosk: AI Clinical History Software Platform
Backend entrypoint (FastAPI).

Implements the /extract-history and /extract-from-image endpoints per
claude.md.md:
- Pydantic v2 BaseModel data contracts with explicit Field(description=...)
- OpenAI native Structured Outputs (client.beta.chat.completions.parse) —
  never hand-parse LLM string output.
- async def routing.
- SOCRATES pain framework + AYUSH Dashavidha Pariksha parameters captured
  in the schema below.
- Emergency red-flag detection (cardiac / neurological) forces
  red_flags_detected = True.
- Database Rules: supabase-py for all DB operations, extracted histories are
  stored in patient_histories, and prescription images are uploaded to the
  medical_documents Storage bucket before their public URL is sent to the
  OpenAI Vision model.
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Optional
from urllib import request

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from openai import OpenAI, OpenAIError
from supabase import Client, create_client

load_dotenv()

logger = logging.getLogger("medikiosk")

# ---------------------------------------------------------------------------
# Pydantic Data Contracts (Section 4 of claude.md.md)
# ---------------------------------------------------------------------------


class Medication(BaseModel):
    """A single current medication entry."""

    name: str = Field(description="Name of the medication as reported by the patient.")
    dosage: str = Field(description="Dosage of the medication, e.g. '500mg'.")
    frequency: str = Field(description="How often the medication is taken, e.g. 'twice daily'.")


class AyushParameters(BaseModel):
    """
    Full AYUSH Dashavidha Pariksha ('tenfold examination') parameters, captured
    from the conversation where the transcript provides relevant information.
    """

    prakriti: str = Field(
        description="Patient's baseline constitutional type (Vata/Pitta/Kapha balance) as assessed from the conversation."
    )
    vikriti: str = Field(
        description="Patient's current state of doshic imbalance as assessed from the conversation."
    )
    sara: str = Field(description="Tissue (dhatu) quality and excellence assessed from the conversation.")
    samhanana: str = Field(description="Body compactness/build (physical compactness of the frame) assessed from the conversation.")
    pramana: str = Field(description="Body measurements and proportion assessed from the conversation.")
    satmya: str = Field(description="Patient's suitability/adaptability to different foods, climates, and conditions.")
    sattva: str = Field(description="Patient's psychic strength and mental resilience assessed from the conversation.")
    ahara_shakti: str = Field(description="Patient's digestive/appetite capacity (power of food intake and digestion).")
    vyayama_shakti: str = Field(description="Patient's exercise capacity and physical stamina.")
    vaya: str = Field(description="Patient's age-related constitutional stage (e.g. growth, adult, or decline phase).")


class ClinicalHistorySummary(BaseModel):
    """
    Primary JSON data contract for backend-to-frontend clinical history
    communication (Section 4 of claude.md.md).
    """

    chief_complaint: str = Field(
        description="The patient's primary reason for the visit, in 1-2 sentences."
    )
    hpi_socrates: str = Field(
        description=(
            "Detailed narrative of the History of Present Illness, structured using the "
            "SOCRATES framework: Site, Onset, Character, Radiation, Associated symptoms, "
            "Time course, Exacerbating/relieving factors, and Severity."
        )
    )
    past_medical_history: List[str] = Field(
        description="List of past medical history items reported by the patient."
    )
    current_medications: List[Medication] = Field(
        description="List of medications the patient is currently taking."
    )
    ayush_parameters: AyushParameters = Field(
        description="AYUSH Dashavidha Pariksha parameters (Prakriti, Vikriti) captured from the conversation."
    )
    red_flags_detected: bool = Field(
        description=(
            "True if the transcript contains markers for acute cardiac events "
            "(e.g. chest pain, dyspnoea) or neurological deficits (e.g. stroke symptoms)."
        )
    )


class TranscriptRequest(BaseModel):
    """Request body for the /extract-history endpoint."""

    transcript: str = Field(description="Raw patient conversation transcript to extract structured history from.")
    patient_id: Optional[str] = Field(default=None, description="Optional patient identifier for retrieving known allergy and previous report context.")


class ConversationalQuestionResponse(BaseModel):
    """A single bilingual, non-prescriptive clinical intake response."""

    reply: str = Field(description="The next question for the patient, in the patient's language.")
    language: str = Field(description="Detected language, such as Hindi, English, or Hinglish.")
    red_flags_detected: bool = Field(
        description="True when the patient's message suggests an urgent emergency symptom."
    )
    transcript: Optional[str] = Field(
        default=None,
        description="Recognized patient speech when the response came from a voice request.",
    )


class VoiceTranscriptionResult(BaseModel):
    """Speech-to-text result for patient or doctor dictation."""

    text: str = Field(description="The text recovered from the audio input.")
    provider: str = Field(description="Speech provider used: openai or bhashini.")
    language: str = Field(description="Detected or requested language code, e.g. en, hi, or hi-IN.")
    confidence: Optional[float] = Field(default=None, description="Optional confidence score if the provider provides one.")


class VoiceSynthesisResult(BaseModel):
    """Text-to-speech result containing audio payload."""

    provider: str = Field(description="Voice provider used for synthesis.")
    language: str = Field(description="Language used for synthesis.")
    mime_type: str = Field(description="Audio MIME type returned by the provider.")
    audio_base64: str = Field(description="Base64-encoded audio data for playback in a mobile or web app.")


class PatientProfile(BaseModel):
    """Long-term patient medical profile used to avoid repeated allergy and medication questions."""

    patient_id: str = Field(description="Stable identifier for the person or login account.")
    name: Optional[str] = Field(default=None, description="Patient name if available.")
    age: Optional[int] = Field(default=None, description="Patient age in years, if known.")
    allergies: List[str] = Field(default_factory=list, description="Known allergies, including medicines, food, and environmental triggers.")
    medication_allergies: List[str] = Field(default_factory=list, description="Medicine-specific allergies already known.")
    chronic_conditions: List[str] = Field(default_factory=list, description="Known chronic medical conditions.")
    ongoing_medications: List[str] = Field(default_factory=list, description="Current medications the patient is already taking.")
    notes: Optional[str] = Field(default=None, description="Doctor or patient notes to remember across visits.")
    last_updated: Optional[str] = Field(default=None, description="Last updated timestamp, if provided by the client.")


class PatientReport(BaseModel):
    """A saved previous report or record that the model can reuse for follow-up questioning."""

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Stable identifier for the stored report.")
    patient_id: str = Field(description="Patient identifier associated with the saved report.")
    report_type: str = Field(description="Type of report, such as lab, prescription, previous-visit, or doctor-note.")
    report_date: Optional[str] = Field(default=None, description="Report date, if known.")
    summary: str = Field(description="Condensed summary of the past report or medical history.")
    notes: Optional[str] = Field(default=None, description="Detailed notes from the report.")
    created_at: Optional[str] = Field(default=None, description="Saved timestamp.")


class ApprovedTrainingCase(BaseModel):
    """A doctor-reviewed medical case that may be added to a curated training dataset."""

    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Stable identifier for the approved case.")
    patient_id: str = Field(description="Patient identifier for the case.")
    transcript: str = Field(description="Conversation transcript or patient story to preserve for future model learning.")
    summary: str = Field(description="Doctor-reviewed summary of the case.")
    diagnoses: List[str] = Field(default_factory=list, description="Approved diagnoses associated with the case.")
    treatment_plan: List[str] = Field(default_factory=list, description="Treatment or management notes approved by the doctor.")
    approved_by: str = Field(description="Doctor or reviewing clinician identifier.")
    tags: List[str] = Field(default_factory=list, description="Labels such as 'cardiology', 'dermatology', 'pain', 'follow_up'.")


class PatientHistoryRecord(BaseModel):
    """A row from the patient_histories Supabase table, as returned after insert."""

    id: str = Field(description="UUID primary key of the patient_histories row.")
    created_at: str = Field(description="Timestamp the row was created, as returned by Postgres.")
    chief_complaint: str = Field(description="The patient's primary reason for the visit.")
    hpi_socrates: str = Field(description="Detailed narrative of the History of Present Illness (SOCRATES).")
    current_medications: List[Medication] = Field(description="Medications the patient is currently taking.")
    ayush_parameters: AyushParameters = Field(description="AYUSH Dashavidha Pariksha parameters.")
    red_flags_detected: bool = Field(description="True if emergency red flags were detected.")
    alert_acknowledged: bool = Field(
        default=False, description="True once triage staff have acknowledged a red-flag alert for this history."
    )


class InvestigationResult(BaseModel):
    """A single lab or imaging investigation result extracted from a medical document."""

    test_name: str = Field(description="Name of the lab test or investigation, e.g. 'Hemoglobin' or 'Chest X-Ray'.")
    value: str = Field(description="The measured value or finding, e.g. '10.2 g/dL'.")
    reference_range: str = Field(description="The normal reference range for this test, e.g. '13.0-17.0 g/dL'. Empty if not applicable/available.")
    is_abnormal: bool = Field(description="True if the value falls outside the normal reference range.")


class ExtractedDocument(BaseModel):
    """Structured clinical data extracted from a medical document image via the OpenAI Vision model."""

    diagnoses: List[str] = Field(description="Diagnoses mentioned in the document.")
    medications: List[Medication] = Field(description="Medications prescribed in the document.")
    investigations: List[InvestigationResult] = Field(
        description="Lab or imaging investigation results found in the document."
    )
    procedures: List[str] = Field(description="Procedures or surgeries mentioned in the document.")


class DocumentUploadResult(BaseModel):
    """Response for /upload-document: the Supabase Storage path plus the extracted document data."""

    storage_path: str = Field(
        description="Path of the uploaded file within the medical_documents Supabase Storage bucket."
    )
    extracted_document: ExtractedDocument = Field(
        description="Structured clinical data extracted by the OpenAI Vision model."
    )


class WaitingRoomResponse(BaseModel):
    """Response for GET /api/v1/waiting-room: the most recently created patient histories."""

    histories: List[PatientHistoryRecord] = Field(
        description="The 10 most recently created patient histories, ordered newest first."
    )


class ConversationTurn(BaseModel):
    """One turn in a patient interview conversation."""

    role: str = Field(description="Either 'assistant' (the kiosk's question) or 'patient' (the patient's answer).")
    content: str = Field(description="The text of this conversation turn.")


class ConverseRequest(BaseModel):
    """Request body for POST /converse. Stateless: the caller resends the full history each turn."""

    history: List[ConversationTurn] = Field(
        default_factory=list, description="The conversation so far, oldest first. Empty on the first call."
    )


class ConversationStep(BaseModel):
    """The kiosk's next move in an adaptive patient interview."""

    next_question: str = Field(description="The next question to ask the patient. Empty string once is_complete is true.")
    quick_reply_options: List[str] = Field(
        description="0-4 short tap-friendly answer options for next_question, for touch-based input. Empty if open-ended."
    )
    is_complete: bool = Field(
        description="True once enough history has been gathered to generate a full clinical summary."
    )
    is_red_flag_urgent: bool = Field(
        description="True if the patient's most recent answer indicates an emergency requiring immediate triage."
    )


class DocumentExtractionInput(BaseModel):
    """A previously-extracted document (from /upload-document) to merge into a unified summary."""

    storage_path: str = Field(description="Supabase Storage path of the source document, for reference/traceability.")
    extracted_document: ExtractedDocument = Field(description="Previously extracted structured data from this document.")


class GenerateSummaryRequest(BaseModel):
    """Request body for POST /generate-summary."""

    patient_id: Optional[str] = Field(default=None, description="Patient ID for associating prior reports and allergy context to the generated summary.")
    transcript: Optional[str] = Field(
        default=None, description="Conversational history transcript, if a voice/touch interview was conducted."
    )
    documents: List[DocumentExtractionInput] = Field(
        default_factory=list, description="Previously extracted documents to merge into the unified summary."
    )


class PatientHistoryUpdate(BaseModel):
    """Request body for PATCH /patient-histories/{id}. Only supplied fields are updated."""

    chief_complaint: Optional[str] = Field(default=None, description="Updated chief complaint.")
    hpi_socrates: Optional[str] = Field(default=None, description="Updated HPI narrative.")
    current_medications: Optional[List[Medication]] = Field(default=None, description="Updated medications list.")
    ayush_parameters: Optional[AyushParameters] = Field(default=None, description="Updated AYUSH parameters.")
    red_flags_detected: Optional[bool] = Field(default=None, description="Updated red-flag status.")


class AbhaVerificationRequest(BaseModel):
    """Request body for the mock POST /abdm/verify-abha endpoint."""

    abha_id: str = Field(description="The patient's ABHA (Ayushman Bharat Health Account) ID or address.")


class AbhaVerificationResult(BaseModel):
    """Mock response for POST /abdm/verify-abha, standing in for a real ABDM Gateway call."""

    abha_id: str = Field(description="The ABHA ID that was verified.")
    verified: bool = Field(description="Whether the ABHA ID was successfully verified.")
    patient_name: str = Field(description="Mock patient name returned by the ABDM Gateway.")
    date_of_birth: str = Field(description="Mock patient date of birth (YYYY-MM-DD) returned by the ABDM Gateway.")
    gender: str = Field(description="Mock patient gender returned by the ABDM Gateway.")


class HisPushRequest(BaseModel):
    """Request body for the mock POST /abdm/push-to-his endpoint."""

    history_id: str = Field(description="UUID of the patient_histories row to push to the Hospital Information System.")
    abha_id: str = Field(description="The patient's ABHA ID to link this record to in the HIS/ABDM Personal Health Record.")


class HisPushResult(BaseModel):
    """Mock response for POST /abdm/push-to-his, standing in for a real FHIR-based HIS/ABDM push."""

    history_id: str = Field(description="UUID of the patient_histories row that was pushed.")
    abha_id: str = Field(description="The ABHA ID the record was linked to.")
    his_record_id: str = Field(description="Mock record ID assigned by the Hospital Information System.")
    status: str = Field(description="Mock push status, e.g. 'submitted'.")


# ---------------------------------------------------------------------------
# App & OpenAI client setup
# ---------------------------------------------------------------------------

app = FastAPI(title="MediKiosk API", version="0.1.0")

# OpenAI-compatible providers can be selected without changing endpoint code.
# Set AI_PROVIDER=xai and GROK_API_KEY to use xAI's Grok models.
AI_PROVIDER = os.environ.get("AI_PROVIDER", "openai").lower()
if AI_PROVIDER == "xai":
    AI_API_KEY = os.environ.get("GROK_API_KEY") or "not-set"
    AI_BASE_URL = "https://api.x.ai/v1"
    AI_MODEL = os.environ.get("AI_MODEL", "grok-4.6")
else:
    AI_API_KEY = os.environ.get("OPENAI_API_KEY") or "not-set"
    AI_BASE_URL = os.environ.get("AI_BASE_URL") or None
    AI_MODEL = os.environ.get("AI_MODEL", "gpt-4o-2024-08-06")

client = OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)

VOICE_PROVIDER = os.environ.get("VOICE_PROVIDER", "openai").lower()
LOCAL_WHISPER_MODEL = os.environ.get("LOCAL_WHISPER_MODEL", "small")
LOCAL_WHISPER_DEVICE = os.environ.get("LOCAL_WHISPER_DEVICE", "auto")
BHASHINI_API_KEY = os.environ.get("BHASHINI_API_KEY")
BHASHINI_ASR_URL = os.environ.get("BHASHINI_ASR_URL")
BHASHINI_TTS_URL = os.environ.get("BHASHINI_TTS_URL")
BHASHINI_USER_ID = os.environ.get("BHASHINI_USER_ID")
OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "sage")

_local_whisper = None
_local_whisper_lock = Lock()


def _load_local_whisper():
    global _local_whisper
    if _local_whisper is None:
        with _local_whisper_lock:
            if _local_whisper is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise RuntimeError(
                        "Local voice requires faster-whisper. Install it with 'pip install faster-whisper'."
                    ) from exc
                device = (
                    "cuda"
                    if LOCAL_WHISPER_DEVICE == "auto" and torch.cuda.is_available()
                    else LOCAL_WHISPER_DEVICE
                )
                if device == "auto":
                    device = "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                _local_whisper = WhisperModel(
                    LOCAL_WHISPER_MODEL,
                    device=device,
                    compute_type=compute_type,
                )
    return _local_whisper


def _transcribe_locally(audio_bytes: bytes, language: str, filename: str) -> VoiceTranscriptionResult:
    model = _load_local_whisper()
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
        audio_file.write(audio_bytes)
        audio_path = audio_file.name
    try:
        language_code = language.split("-")[0] if language and language.lower() not in ("auto", "hinglish") else None
        segments, info = model.transcribe(
            audio_path,
            language=language_code,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        Path(audio_path).unlink(missing_ok=True)
    if not text:
        raise RuntimeError("Local speech recognition returned no transcription text.")
    detected_language = getattr(info, "language", None) or language or "en"
    return VoiceTranscriptionResult(text=text, provider="local", language=detected_language)


def _synthesize_locally(text: str, language: str) -> VoiceSynthesisResult:
    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError("Local voice requires pyttsx3. Install it with 'pip install pyttsx3'.") from exc
    engine = pyttsx3.init()
    engine.setProperty("rate", int(os.environ.get("LOCAL_TTS_RATE", "155")))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_path = audio_file.name
    try:
        engine.save_to_file(text, audio_path)
        engine.runAndWait()
        audio_bytes = Path(audio_path).read_bytes()
    finally:
        Path(audio_path).unlink(missing_ok=True)
    if not audio_bytes:
        raise RuntimeError("Local text-to-speech returned no audio.")
    return VoiceSynthesisResult(
        provider="local",
        language=language,
        mime_type="audio/wav",
        audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
    )

DEVANAGARI_MAP = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ट": "t", "ठ": "th",
    "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph",
    "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ं": "n", "ः": "h", "।": ".", "़": "",
}
HINGLISH_NORMALIZATIONS = {
    "mere pet me": "mere pet me",
    "mere pet mei": "mere pet me",
    "pet me dard": "pet me dard",
    "pet me pain": "pet me pain",
    "hue": "hai",
    "mai": "main",
    "mujhe": "mujhe",
    "haii": "hai",
    "kaise": "kaise",
    "kab": "kab",
    "kya": "kya",
    "medicine se": "medicine se",
    "allergy": "allergy",
    "allergies": "allergies",
}


def romanize_hindi_text(text: str) -> str:
    result = []
    for char in text:
        result.append(DEVANAGARI_MAP.get(char, char))
    normalized = "".join(result)
    normalized = normalized.replace("aa p", "aap").replace(" hai", " hai")
    return normalized


def normalize_multilingual_voice_text(text: str, language_hint: Optional[str] = None) -> tuple[str, str]:
    if not text:
        return "", "en-IN"
    cleaned = text.strip()
    has_devanagari = any("\u0900" <= char <= "\u097f" for char in cleaned)
    if has_devanagari:
        cleaned = romanize_hindi_text(cleaned)
        language = "hi-IN"
    else:
        lower = cleaned.lower()
        for source, target in HINGLISH_NORMALIZATIONS.items():
            lower = lower.replace(source, target)
        cleaned = lower
        if re.search(r"\b(mere|pet|dard|hai|kaise|mujhe|kya|kab)\b", cleaned):
            language = "hi-IN"
        else:
            language = language_hint or "en-IN"
    return cleaned, language


def _safe_supabase_insert(table_name: str, payload: dict) -> Optional[dict]:
    try:
        response = supabase.table(table_name).insert(payload).execute()
        if response.data:
            return response.data[0]
    except Exception as exc:
        logger.warning("Supabase table %s not available or insert failed: %s", table_name, exc)
    return None


def _safe_supabase_select(table_name: str, key: str, value: str) -> Optional[dict]:
    try:
        response = supabase.table(table_name).select("*").eq(key, value).limit(1).execute()
        if response.data:
            return response.data[0]
    except Exception as exc:
        logger.warning("Supabase table %s not available or select failed: %s", table_name, exc)
    return None


def _safe_supabase_query(table_name: str, key: str, value: str) -> List[dict]:
    try:
        response = supabase.table(table_name).select("*").eq(key, value).execute()
        return response.data or []
    except Exception as exc:
        logger.warning("Supabase table %s not available or list query failed: %s", table_name, exc)
        return []


def _get_patient_context(patient_id: Optional[str]) -> str:
    if not patient_id:
        return ""

    profile = _safe_supabase_select("patient_profiles", "patient_id", patient_id)
    reports = _safe_supabase_query("patient_reports", "patient_id", patient_id)

    parts = []
    if profile:
        allergies = profile.get("allergies") or []
        medication_allergies = profile.get("medication_allergies") or []
        chronic = profile.get("chronic_conditions") or []
        meds = profile.get("ongoing_medications") or []
        if allergies or medication_allergies or chronic or meds:
            parts.append(
                "Known patient profile: "
                + "; ".join(
                    filter(
                        None,
                        [
                            f"Allergies: {', '.join(allergies)}" if allergies else None,
                            f"Medicine allergies: {', '.join(medication_allergies)}" if medication_allergies else None,
                            f"Chronic conditions: {', '.join(chronic)}" if chronic else None,
                            f"Current medications: {', '.join(meds)}" if meds else None,
                        ],
                    )
                )
            )
    if reports:
        summaries = []
        for report in reports[:5]:
            summary = report.get("summary")
            if summary:
                summaries.append(summary)
        if summaries:
            parts.append("Previous medical reports: " + " | ".join(summaries))
    return "\n".join(parts)


def _normalize_voice_text(payload: object) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("text", "transcript", "output_text", "final", "result"):
            if key in payload and isinstance(payload[key], str):
                return payload[key].strip()
        if "output" in payload:
            nested = _normalize_voice_text(payload["output"])
            if nested:
                return nested
        if "data" in payload:
            nested = _normalize_voice_text(payload["data"])
            if nested:
                return nested
        if "result" in payload and isinstance(payload["result"], dict):
            return _normalize_voice_text(payload["result"])
    if isinstance(payload, list):
        parts = [_normalize_voice_text(item) for item in payload]
        text = " ".join(part for part in parts if part)
        if text:
            return text
    return ""


def _transcribe_with_bhashini(audio_bytes: bytes, language: str, filename: str) -> VoiceTranscriptionResult:
    if not BHASHINI_ASR_URL:
        raise RuntimeError("VOICE_PROVIDER=bhashini requires BHASHINI_ASR_URL to be set in the environment.")
    if not BHASHINI_API_KEY:
        raise RuntimeError("VOICE_PROVIDER=bhashini requires BHASHINI_API_KEY to be set in the environment.")

    payload = {
        "language": language,
        "task": "asr",
        "audio_b64": base64.b64encode(audio_bytes).decode("utf-8"),
        "filename": filename,
    }
    req = request.Request(
        BHASHINI_ASR_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BHASHINI_API_KEY}",
            **({"x-user-id": BHASHINI_USER_ID} if BHASHINI_USER_ID else {}),
        },
        method="POST",
    )
    with request.urlopen(req, timeout=60) as response:
        body = response.read()
    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception:
        parsed = {"text": body.decode("utf-8", errors="ignore")}
    text = _normalize_voice_text(parsed)
    if not text:
        raise RuntimeError("Bhashini ASR returned no transcription text.")
    return VoiceTranscriptionResult(text=text, provider="bhashini", language=language)


def _synthesize_with_bhashini(text: str, language: str) -> VoiceSynthesisResult:
    if not BHASHINI_TTS_URL:
        raise RuntimeError("VOICE_PROVIDER=bhashini requires BHASHINI_TTS_URL to be set in the environment.")
    if not BHASHINI_API_KEY:
        raise RuntimeError("VOICE_PROVIDER=bhashini requires BHASHINI_API_KEY to be set in the environment.")

    payload = {
        "text": text,
        "language": language,
        "gender": "female",
    }
    req = request.Request(
        BHASHINI_TTS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BHASHINI_API_KEY}",
            **({"x-user-id": BHASHINI_USER_ID} if BHASHINI_USER_ID else {}),
        },
        method="POST",
    )
    with request.urlopen(req, timeout=60) as response:
        body = response.read()
    if isinstance(body, bytes):
        return VoiceSynthesisResult(
            provider="bhashini",
            language=language,
            mime_type="audio/mpeg",
            audio_base64=base64.b64encode(body).decode("utf-8"),
        )
    raise RuntimeError("Bhashini TTS response was not binary audio data.")


async def _transcribe_voice_upload(file: UploadFile, language: str) -> VoiceTranscriptionResult:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")

    if VOICE_PROVIDER == "local":
        return _transcribe_locally(contents, language, file.filename or "voice.wav")

    if VOICE_PROVIDER in ("openai", "default"):
        if AI_API_KEY in (None, "not-set"):
            raise RuntimeError("OPENAI_API_KEY is required to use the OpenAI voice pipeline.")
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=(file.filename or "voice.wav", contents, file.content_type or "audio/wav"),
            language=language,
        )
        return VoiceTranscriptionResult(
            text=(transcription.text or "").strip(),
            provider="openai",
            language=language,
            confidence=getattr(transcription, "confidence", None),
        )

    if VOICE_PROVIDER == "bhashini":
        return _transcribe_with_bhashini(contents, language, file.filename or "voice.wav")

    raise RuntimeError(f"Unsupported VOICE_PROVIDER: {VOICE_PROVIDER}. Set it to 'openai' or 'bhashini'.")


async def _synthesize_voice_text(text: str, language: str) -> VoiceSynthesisResult:
    if VOICE_PROVIDER == "local":
        return _synthesize_locally(text, language)

    if VOICE_PROVIDER in ("openai", "default"):
        if AI_API_KEY in (None, "not-set"):
            raise RuntimeError("OPENAI_API_KEY is required to use the OpenAI voice synthesis pipeline.")
        response = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
        )
        audio_bytes = response.read()
        return VoiceSynthesisResult(
            provider="openai",
            language=language,
            mime_type=response.content_type or "audio/mpeg",
            audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
        )

    if VOICE_PROVIDER == "bhashini":
        return _synthesize_with_bhashini(text, language)

    raise RuntimeError(f"Unsupported VOICE_PROVIDER: {VOICE_PROVIDER}. Set it to 'openai' or 'bhashini'.")


async def _local_voice_assistant(file: UploadFile, language: str) -> ConversationalQuestionResponse:
    transcription = await _transcribe_voice_upload(file, language)
    text_for_model, detected_language = normalize_multilingual_voice_text(transcription.text, language)
    try:
        from local_bilingual_model import ask

        reply = await asyncio.to_thread(ask, text_for_model)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        logger.exception("Local bilingual model inference failed for voice input.")
        raise HTTPException(status_code=503, detail=f"Local bilingual model unavailable: {exc}") from exc
    lower = text_for_model.lower()
    hinglish = any(word in lower for word in ("mere", "pet", "dard", "hai", "hue", "kaise"))
    hindi = any("\u0900" <= char <= "\u097f" for char in transcription.text)
    urgent = any(word in lower for word in ("chest pain", "breathing", "faint", "stroke", "बेहोश", "सीने"))
    return ConversationalQuestionResponse(
        reply=reply,
        language="Hinglish" if hinglish else ("Hindi" if hindi else ("Hindi" if detected_language.startswith("hi") else "English")),
        red_flags_detected=urgent,
        transcript=transcription.text,
    )


# Same fallback pattern for Supabase: allow the app to start before
# SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are configured; real DB/Storage
# calls will fail with a clear 502 until valid credentials are set in the
# environment.
#
# The backend authenticates with the service_role key (not the anon key) so
# it bypasses Row Level Security entirely. patient_histories and the
# medical_documents object policies deny anon/authenticated access outright
# (see supabase_schema.sql) — this backend is the only client allowed to
# read or write patient data. Never expose this key to the frontend.
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL") or "https://not-set.supabase.co",
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "not-set",
)

PATIENT_HISTORIES_TABLE = "patient_histories"
MEDICAL_DOCUMENTS_BUCKET = "medical_documents"

SYSTEM_PROMPT = (
    "You are a clinical history extraction engine for MediKiosk, an AI clinical "
    "history intake kiosk. Given a patient conversation transcript, extract a "
    "structured clinical history summary.\n\n"
    "When the patient reports pain, apply the SOCRATES framework (Site, Onset, "
    "Character, Radiation, Associated symptoms, Time course, Exacerbating/relieving "
    "factors, Severity) in the hpi_socrates narrative.\n\n"
    "Capture the full AYUSH Dashavidha Pariksha ('tenfold examination') whenever the "
    "transcript provides relevant information: Prakriti, Vikriti, Sara, Samhanana, "
    "Pramana, Satmya, Sattva, Ahara Shakti, Vyayama Shakti, and Vaya. If a parameter "
    "cannot be assessed from the transcript, state that explicitly rather than "
    "inventing a value.\n\n"
    "Set red_flags_detected to true if the transcript contains any markers of acute "
    "cardiac events (e.g. chest pain, dyspnoea) or neurological deficits (e.g. stroke "
    "symptoms such as facial droop, slurred speech, sudden weakness). Otherwise set it "
    "to false."
)

CONVERSE_SYSTEM_PROMPT = (
    "You are the adaptive conversational history-taking engine for MediKiosk, an AI "
    "clinical intake kiosk used in Indian hospital OPDs. You conduct a structured "
    "patient interview one question at a time, mirroring how an experienced physician "
    "elicits a history.\n\n"
    "Ask ONE short, plain-language question per turn (suitable for an elderly or "
    "low-literacy patient). Cover, in a natural adaptive order driven by the patient's "
    "answers: chief complaint; if pain or a symptom is reported, drill into it using "
    "SOCRATES (Site, Onset, Character, Radiation, Associated symptoms, Time course, "
    "Exacerbating/relieving factors, Severity); past medical/surgical history; current "
    "medications and allergies; family history; personal/lifestyle history; a brief "
    "review of systems; and, where relevant, AYUSH Dashavidha Pariksha cues (Prakriti, "
    "Vikriti, Sara, Samhanana, Pramana, Satmya, Sattva, Ahara Shakti, Vyayama Shakti, "
    "Vaya).\n\n"
    "For each question, if it has a small set of natural short answers (e.g. yes/no, a "
    "severity scale, common durations), propose up to 4 quick_reply_options so the "
    "patient can tap instead of speaking. Leave quick_reply_options empty for genuinely "
    "open-ended questions.\n\n"
    "Set is_red_flag_urgent to true the moment any answer indicates an emergency (acute "
    "cardiac symptoms, stroke symptoms, etc.), independent of whether the interview is "
    "otherwise complete.\n\n"
    "Set is_complete to true, and next_question to an empty string, once you have "
    "gathered enough history to produce a complete clinical summary — do not drag the "
    "interview out longer than necessary."
)

GENERATE_SUMMARY_SYSTEM_PROMPT = (
    "You are a clinical history synthesis engine for MediKiosk. You will be given a "
    "patient's conversational history transcript (if a voice/touch interview was "
    "conducted) and/or structured data extracted from their prior medical documents "
    "(if any were scanned). Synthesize everything provided into a single, unified, "
    "physician-ready ClinicalHistorySummary — do not produce separate summaries per "
    "source.\n\n"
    "Fold document-derived diagnoses and procedures into past_medical_history, "
    "document-derived medications into current_medications (merging with any "
    "conversation-derived medications, avoiding duplicates), and mention clinically "
    "significant abnormal investigation results in the hpi_socrates or "
    "past_medical_history narrative as appropriate.\n\n"
    "When the patient reports pain, apply the SOCRATES framework in the hpi_socrates "
    "narrative. Capture the full AYUSH Dashavidha Pariksha (Prakriti, Vikriti, Sara, "
    "Samhanana, Pramana, Satmya, Sattva, Ahara Shakti, Vyayama Shakti, Vaya) wherever "
    "information is available; state explicitly where a parameter cannot be assessed.\n\n"
    "Set red_flags_detected to true if anything provided indicates acute cardiac events "
    "or neurological deficits. Otherwise set it to false."
)
CONVERSATION_SYSTEM_PROMPT = (
    "You are MediKiosk's clinical intake interviewer. Your only job is to ask the patient "
    "the next useful question; do not diagnose, recommend treatment, or prescribe medicine. "
    "Detect whether the patient uses Hindi, English, or Hinglish and reply in that same style. "
    "Use simple, respectful language and ask one focused question at a time. For pain, ask "
    "follow-up questions covering location, onset, character, severity, duration, radiation, "
    "and what makes it better or worse, adapting to answers already provided. Ask relevant "
    "questions about associated symptoms, medical history, and current medicines only when "
    "needed. If emergency warning signs are reported (severe chest pain, trouble breathing, "
    "fainting, sudden weakness, facial drooping, or confusion), set red_flags_detected true "
    "and tell the patient to seek emergency care immediately; do not provide a prescription. "
    "The reply must always be a question, except for that emergency instruction followed by "
    "a question."
)


# ---------------------------------------------------------------------------
# Persistence (Database Rules, Section 5 of claude.md.md)
# ---------------------------------------------------------------------------


def _persist_history(summary: ClinicalHistorySummary) -> dict:
    """
    Insert an extracted clinical history into the patient_histories table via
    the official supabase-py client, using the parsed ClinicalHistorySummary's
    exact JSON shape for the JSONB columns, and return the inserted row as
    Supabase returns it (including its generated id and created_at).
    """
    try:
        response = (
            supabase.table(PATIENT_HISTORIES_TABLE)
            .insert(
                {
                    "chief_complaint": summary.chief_complaint,
                    "hpi_socrates": summary.hpi_socrates,
                    "current_medications": [m.model_dump() for m in summary.current_medications],
                    "ayush_parameters": summary.ayush_parameters.model_dump(),
                    "red_flags_detected": summary.red_flags_detected,
                }
            )
            .execute()
        )
    except Exception as exc:
        logger.exception("Failed to persist clinical history to Supabase.")
        raise HTTPException(status_code=502, detail=f"Failed to persist clinical history to Supabase: {exc}") from exc

    if not response.data:
        raise HTTPException(status_code=502, detail="Supabase insert returned no data.")

    return response.data[0]


def _store_patient_report(patient_id: Optional[str], summary: ClinicalHistorySummary, report_type: str = "summary") -> None:
    if not patient_id:
        return
    report_data = {
        "patient_id": patient_id,
        "report_type": report_type,
        "report_date": datetime.utcnow().date().isoformat(),
        "summary": f"{summary.chief_complaint} | {summary.hpi_socrates}",
        "notes": json.dumps({
            "current_medications": [m.model_dump() for m in summary.current_medications],
            "red_flags_detected": summary.red_flags_detected,
            "ayush_parameters": summary.ayush_parameters.model_dump(),
        }, ensure_ascii=False),
        "created_at": datetime.utcnow().isoformat(),
    }
    _safe_supabase_insert("patient_reports", report_data)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/voice/transcribe", response_model=VoiceTranscriptionResult)
async def transcribe_voice(file: UploadFile = File(...), language: str = "en") -> VoiceTranscriptionResult:
    """Convert spoken audio into text for patient intake or doctor dictation."""
    try:
        return await _transcribe_voice_upload(file, language)
    except (RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"Voice transcription failed: {exc}") from exc


@app.post("/voice/speak", response_model=VoiceSynthesisResult)
async def speak_text(text: str = "", language: str = "en") -> VoiceSynthesisResult:
    """Convert text back into audio for a patient or doctor-facing voice assistant."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text to speak cannot be empty.")
    try:
        return await _synthesize_voice_text(text, language)
    except (RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"Voice synthesis failed: {exc}") from exc


@app.post("/voice/patient-assistant", response_model=ConversationalQuestionResponse)
async def patient_voice_assistant(file: UploadFile = File(...), language: str = "en") -> ConversationalQuestionResponse:
    """Transcribe a spoken patient answer and return the next question to ask."""
    try:
        return await _local_voice_assistant(file, language)
    except HTTPException:
        raise
    except (RuntimeError, ValueError, OSError) as exc:
        logger.exception("Voice patient assistant failed.")
        raise HTTPException(status_code=502, detail=f"Voice patient assistant failed: {exc}") from exc


@app.post("/doctor/voice-note", response_model=VoiceTranscriptionResult)
async def doctor_voice_note(file: UploadFile = File(...), language: str = "en") -> VoiceTranscriptionResult:
    """Transcript doctor dictation to text so the physician can review or save notes."""
    return await _transcribe_voice_upload(file, language)


@app.post("/patient/profile", response_model=PatientProfile)
async def save_patient_profile(profile: PatientProfile) -> PatientProfile:
    """Save or update a patient's long-term allergies and medical profile for future visits."""
    payload = profile.model_dump()
    payload["last_updated"] = payload.get("last_updated") or datetime.utcnow().isoformat()
    stored = _safe_supabase_insert("patient_profiles", payload)
    if stored:
        return PatientProfile(**stored)
    return profile


@app.get("/patient/profile/{patient_id}", response_model=PatientProfile)
async def get_patient_profile(patient_id: str) -> PatientProfile:
    """Fetch the stored patient profile to avoid asking repeatedly for known allergies or medication issues."""
    stored = _safe_supabase_select("patient_profiles", "patient_id", patient_id)
    if stored:
        return PatientProfile(**stored)
    raise HTTPException(status_code=404, detail=f"Patient profile not found for {patient_id}.")


@app.post("/patient/report", response_model=PatientReport)
async def save_patient_report(report: PatientReport) -> PatientReport:
    """Store a previous patient report so the model can reuse that history during follow-up questioning."""
    payload = report.model_dump()
    payload["created_at"] = payload.get("created_at") or datetime.utcnow().isoformat()
    stored = _safe_supabase_insert("patient_reports", payload)
    if stored:
        return PatientReport(**stored)
    return report


@app.get("/patient/reports/{patient_id}", response_model=List[PatientReport])
async def get_patient_reports(patient_id: str) -> List[PatientReport]:
    """Fetch earlier patient reports so the model can contextually cross-question and advise."""
    records = _safe_supabase_query("patient_reports", "patient_id", patient_id)
    return [PatientReport(**record) for record in records]


@app.post("/patient/intake-start", response_model=ConversationalQuestionResponse)
async def patient_intake_start(patient_id: str, has_previous_history: bool = False) -> ConversationalQuestionResponse:
    """Begin intake by asking whether the patient has prior reports or medical history, first and before deeper questioning."""
    if has_previous_history:
        return ConversationalQuestionResponse(
            reply="Please tell me about your previous medical reports, past illnesses, surgeries, allergies, or recent medications. If you have a lab report or prescription, you can describe it here.",
            language="English",
            red_flags_detected=False,
        )
    return ConversationalQuestionResponse(
        reply="Do you have any previous medical history or reports? If yes, tell me about them. If not, just say 'No previous history' and we will start fresh.",
        language="English",
        red_flags_detected=False,
    )


@app.post("/doctor/approved-case", response_model=ApprovedTrainingCase)
async def doctor_approved_case(case: ApprovedTrainingCase) -> ApprovedTrainingCase:
    """Store a doctor-reviewed medical case for a curated learning dataset. This is a review-controlled path, not automatic live training."""
    payload = case.model_dump()
    stored = _safe_supabase_insert("doctor_approved_cases", payload)
    if stored:
        return ApprovedTrainingCase(**stored)
    local_path = Path("training/approved_cases.jsonl")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return case


@app.post("/ask-clinical-question", response_model=ConversationalQuestionResponse)
async def ask_clinical_question(request: TranscriptRequest) -> ConversationalQuestionResponse:
    """Ask the next bilingual intake question while using prior patient reports and medical context when available."""
    if not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript must not be empty.")

    patient_context = _get_patient_context(request.patient_id)
    if request.patient_id and not patient_context:
        return ConversationalQuestionResponse(
            reply="Do you have any previous medical history or reports? If yes, tell me about past illnesses, surgeries, allergies, or previous reports. If not, say 'No previous history' and we will start fresh.",
            language="English",
            red_flags_detected=False,
        )

    prompt_text = request.transcript
    if patient_context:
        prompt_text = f"Known patient context:\n{patient_context}\n\nCurrent patient message:\n{request.transcript}"

    try:
        from local_bilingual_model import ask

        reply = await asyncio.to_thread(ask, prompt_text)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        logger.exception("Local bilingual model inference failed.")
        raise HTTPException(status_code=503, detail=f"Local bilingual model unavailable: {exc}") from exc
    lower = prompt_text.lower()
    hindi = any("\u0900" <= char <= "\u097f" for char in request.transcript)
    hinglish = any(word in lower for word in ("mere", "pet", "dard", "hai", "hue", "kaise"))
    urgent = any(word in lower for word in ("chest pain", "breathing", "faint", "stroke", "बेहोश", "सीने"))
    return ConversationalQuestionResponse(
        reply=reply,
        language="Hinglish" if hinglish else ("Hindi" if hindi else "English"),
        red_flags_detected=urgent,
    )


@app.post("/extract-history", response_model=PatientHistoryRecord)
async def extract_history(request: TranscriptRequest) -> PatientHistoryRecord:
    """
    Extract a structured ClinicalHistorySummary from a raw patient transcript
    using OpenAI's native Structured Outputs, insert it as a new row in the
    patient_histories Supabase table, and return the inserted database
    record. The LLM's JSON output is never hand-parsed: the SDK guarantees
    the response conforms to the Pydantic schema via
    client.beta.chat.completions.parse.
    """
    try:
        completion = client.beta.chat.completions.parse(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.transcript},
            ],
            response_format=ClinicalHistorySummary,
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    message = completion.choices[0].message

    if message.refusal:
        raise HTTPException(status_code=422, detail=f"Model refused to process transcript: {message.refusal}")

    parsed = message.parsed

    if parsed is None:
        raise HTTPException(status_code=502, detail="Model did not return a parsed structured output.")

    record = _persist_history(parsed)
    if request.patient_id:
        _store_patient_report(request.patient_id, parsed, report_type="summary")

    return record


@app.post("/extract-from-image", response_model=ClinicalHistorySummary)
async def extract_from_image(file: UploadFile = File(...)) -> ClinicalHistorySummary:
    """
    Extract a structured ClinicalHistorySummary from a prescription/document
    image. Per the Database Rules, the image is uploaded to the
    medical_documents Storage bucket first, and only its public URL is sent
    to the OpenAI Vision model — the raw image bytes are never sent to OpenAI.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    extension = os.path.splitext(file.filename or "")[1] or ".jpg"
    object_path = f"{uuid.uuid4()}{extension}"

    try:
        supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).upload(
            object_path,
            contents,
            {"content-type": file.content_type or "application/octet-stream"},
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase Storage upload failed: {exc}") from exc

    public_url = supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).get_public_url(object_path)

    try:
        completion = client.beta.chat.completions.parse(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract the structured clinical history from this prescription/document image.",
                        },
                        {"type": "image_url", "image_url": {"url": public_url}},
                    ],
                },
            ],
            response_format=ClinicalHistorySummary,
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    message = completion.choices[0].message

    if message.refusal:
        raise HTTPException(status_code=422, detail=f"Model refused to process image: {message.refusal}")

    parsed = message.parsed
    if parsed is None:
        raise HTTPException(status_code=502, detail="Model did not return a parsed structured output.")

    _persist_history(parsed)

    return parsed


@app.post("/upload-document", response_model=DocumentUploadResult)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResult:
    """
    Upload a medical document image (prescription, lab report, or discharge
    summary) to the medical_documents Supabase Storage bucket, then extract
    its diagnoses, medications, investigation results, and procedures via the
    OpenAI gpt-4o Vision model using Structured Outputs. Only the image's
    public Storage URL is sent to OpenAI — never the raw image bytes.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    extension = os.path.splitext(file.filename or "")[1] or ".jpg"
    storage_path = f"{uuid.uuid4()}{extension}"

    # Step 1: upload the raw file bytes to Supabase Storage.
    try:
        supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).upload(
            storage_path,
            contents,
            {"content-type": file.content_type or "application/octet-stream"},
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase Storage upload failed: {exc}") from exc

    # Step 2: retrieve the public URL for the uploaded image.
    public_url = supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).get_public_url(storage_path)

    # Step 3: pass the public URL to the OpenAI gpt-4o Vision model, forcing
    # Structured Outputs to conform to ExtractedDocument.
    try:
        completion = client.beta.chat.completions.parse(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical document extraction engine. Extract every diagnosis, "
                        "medication (with dosage and frequency), lab/imaging investigation result "
                        "(with its value and reference range, flagging is_abnormal when the value "
                        "falls outside that range), and procedure or surgery mentioned in the "
                        "provided medical document image."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all structured clinical data from this document image."},
                        {"type": "image_url", "image_url": {"url": public_url}},
                    ],
                },
            ],
            response_format=ExtractedDocument,
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    message = completion.choices[0].message

    if message.refusal:
        raise HTTPException(status_code=422, detail=f"Model refused to process image: {message.refusal}")

    parsed = message.parsed
    if parsed is None:
        raise HTTPException(status_code=502, detail="Model did not return a parsed structured output.")

    return DocumentUploadResult(storage_path=storage_path, extracted_document=parsed)


@app.get("/api/v1/waiting-room", response_model=WaitingRoomResponse)
async def get_waiting_room() -> WaitingRoomResponse:
    """
    Return the 10 most recently created clinical histories from the
    patient_histories table, ordered by created_at descending.
    """
    try:
        response = (
            supabase.table(PATIENT_HISTORIES_TABLE)
            .select("*")
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to query patient_histories: {exc}") from exc

    return WaitingRoomResponse(histories=response.data)


@app.post("/converse", response_model=ConversationStep)
async def converse(request: ConverseRequest) -> ConversationStep:
    """
    Stateless adaptive interview engine: given the conversation so far, return
    the next question to ask (with optional touch quick-reply options), or
    signal that enough history has been gathered. The caller (frontend) owns
    conversation state and resends the full history each turn; nothing is
    persisted server-side until the interview is complete and
    /generate-summary is called with the resulting transcript.
    """
    messages = [{"role": "system", "content": CONVERSE_SYSTEM_PROMPT}]
    if not request.history:
        messages.append({"role": "user", "content": "[Interview starting. Ask the first question.]"})
    else:
        for turn in request.history:
            role = "assistant" if turn.role == "assistant" else "user"
            messages.append({"role": role, "content": turn.content})

    try:
        completion = client.beta.chat.completions.parse(
            model=AI_MODEL,
            messages=messages,
            response_format=ConversationStep,
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    message = completion.choices[0].message

    if message.refusal:
        raise HTTPException(status_code=422, detail=f"Model refused to continue interview: {message.refusal}")

    parsed = message.parsed
    if parsed is None:
        raise HTTPException(status_code=502, detail="Model did not return a parsed structured output.")

    return parsed


@app.post("/generate-summary", response_model=PatientHistoryRecord)
async def generate_summary(request: GenerateSummaryRequest) -> PatientHistoryRecord:
    """
    Synthesize a conversational history transcript and/or previously-extracted
    document data into a single unified ClinicalHistorySummary, persist it to
    patient_histories, and return the inserted record. This is the Module C
    'Structured History Summary Generator' step: it runs after /converse
    completes and/or after one or more /upload-document calls.
    """
    if not request.transcript and not request.documents:
        raise HTTPException(status_code=400, detail="At least one of transcript or documents must be provided.")

    user_content_parts = []
    if request.transcript:
        user_content_parts.append(f"Conversational history transcript:\n{request.transcript}")
    if request.documents:
        docs_json = json.dumps([d.model_dump() for d in request.documents], indent=2)
        user_content_parts.append(f"Extracted data from {len(request.documents)} prior medical document(s):\n{docs_json}")

    try:
        completion = client.beta.chat.completions.parse(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": GENERATE_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(user_content_parts)},
            ],
            response_format=ClinicalHistorySummary,
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    message = completion.choices[0].message

    if message.refusal:
        raise HTTPException(status_code=422, detail=f"Model refused to synthesize summary: {message.refusal}")

    parsed = message.parsed
    if parsed is None:
        raise HTTPException(status_code=502, detail="Model did not return a parsed structured output.")

    record = _persist_history(parsed)
    if request.patient_id:
        _store_patient_report(request.patient_id, parsed, report_type="summary")

    return record


@app.get("/patient-histories/{history_id}", response_model=PatientHistoryRecord)
async def get_patient_history(history_id: str) -> PatientHistoryRecord:
    """Fetch a single patient history row, for the physician review screen to load before editing."""
    try:
        response = supabase.table(PATIENT_HISTORIES_TABLE).select("*").eq("id", history_id).execute()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to query patient_histories: {exc}") from exc

    if not response.data:
        raise HTTPException(status_code=404, detail="Patient history not found.")

    return response.data[0]


@app.patch("/patient-histories/{history_id}", response_model=PatientHistoryRecord)
async def update_patient_history(history_id: str, update: PatientHistoryUpdate) -> PatientHistoryRecord:
    """
    Apply physician edits to a saved patient history (Module C: 'the summary
    is a draft to accept, amend, or reject'). Only fields explicitly supplied
    in the request body are updated.
    """
    payload = update.model_dump(exclude_unset=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No fields provided to update.")

    try:
        response = (
            supabase.table(PATIENT_HISTORIES_TABLE)
            .update(payload)
            .eq("id", history_id)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to update patient_histories: {exc}") from exc

    if not response.data:
        raise HTTPException(status_code=404, detail="Patient history not found.")

    return response.data[0]


@app.get("/api/v1/priority-alerts", response_model=WaitingRoomResponse)
async def get_priority_alerts() -> WaitingRoomResponse:
    """
    Return unacknowledged patient histories with detected red flags, oldest
    first, for a triage dashboard to poll — the backend half of the 'AI flags
    emergency symptoms and triggers immediate priority alert to triage staff'
    requirement.
    """
    try:
        response = (
            supabase.table(PATIENT_HISTORIES_TABLE)
            .select("*")
            .eq("red_flags_detected", True)
            .eq("alert_acknowledged", False)
            .order("created_at", desc=False)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to query priority alerts: {exc}") from exc

    return WaitingRoomResponse(histories=response.data)


@app.post("/patient-histories/{history_id}/acknowledge-alert", response_model=PatientHistoryRecord)
async def acknowledge_alert(history_id: str) -> PatientHistoryRecord:
    """Mark a red-flag alert as acknowledged by triage staff."""
    try:
        response = (
            supabase.table(PATIENT_HISTORIES_TABLE)
            .update({"alert_acknowledged": True})
            .eq("id", history_id)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to acknowledge alert: {exc}") from exc

    if not response.data:
        raise HTTPException(status_code=404, detail="Patient history not found.")

    return response.data[0]


# ---------------------------------------------------------------------------
# Mock ABDM / Hospital Information System integrations
#
# Per claude.md.md: "Implement mock endpoints for ABDM/Hospital integrations.
# Do not build real database connections during the hackathon." These stand
# in for the real ABDM Gateway (ABHA verification) and hospital HIS/FHIR push
# until real ABDM sandbox credentials are available.
# ---------------------------------------------------------------------------


@app.post("/abdm/verify-abha", response_model=AbhaVerificationResult)
async def verify_abha(request: AbhaVerificationRequest) -> AbhaVerificationResult:
    """Mock ABHA ID verification, standing in for a real ABDM Gateway call."""
    return AbhaVerificationResult(
        abha_id=request.abha_id,
        verified=True,
        patient_name="Mock Patient",
        date_of_birth="1990-01-01",
        gender="unspecified",
    )


@app.post("/abdm/push-to-his", response_model=HisPushResult)
async def push_to_his(request: HisPushRequest) -> HisPushResult:
    """
    Mock push of a patient_histories record to the Hospital Information
    System and ABHA Personal Health Record, standing in for a real FHIR-based
    integration. Confirms the history exists before returning a mock
    confirmation.
    """
    try:
        response = supabase.table(PATIENT_HISTORIES_TABLE).select("id").eq("id", request.history_id).execute()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to look up patient_histories: {exc}") from exc

    if not response.data:
        raise HTTPException(status_code=404, detail="Patient history not found.")

    return HisPushResult(
        history_id=request.history_id,
        abha_id=request.abha_id,
        his_record_id=str(uuid.uuid4()),
        status="submitted",
    )
