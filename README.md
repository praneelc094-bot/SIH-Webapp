# MediKiosk

AI-assisted clinical history intake platform for collecting patient information,
reading medical documents, asking safe follow-up questions, and preparing a
doctor-reviewable summary.

Full architecture rules, tech stack, and the clinical data contracts live in [`claude.md.md`](claude.md.md) — read that first.

## Technology and models

- **Backend:** Python, FastAPI, Uvicorn, Pydantic v2
- **Conversation model:** Qwen/Qwen2.5-1.5B-Instruct, loaded locally for
  question generation using ChatML formatting
- **Fine-tuning:** Optional LoRA adapter trained from the clinical question
  workbooks; the base Qwen model is used by default for dynamic transcripts
- **Document understanding:** OpenAI-compatible structured-output and vision
  API, configurable for OpenAI or xAI/Grok
- **Speech recognition:** Local `faster-whisper`, OpenAI transcription, or
  configurable Bhashini ASR
- **Speech synthesis:** Local `pyttsx3`, OpenAI TTS, or configurable Bhashini
  TTS
- **Database and storage:** Supabase PostgreSQL and Supabase Storage
- **Frontend handoff:** Three independent frontend workstreams are documented
  in [`FRONTEND_README.md`](FRONTEND_README.md)

The backend is device-independent: web and mobile frontends call these APIs;
the local AI models run on the backend machine.

## Getting Started

1. **Clone and enter the project**
   ```bash
   git clone <this-repo-url>
   cd Webapp
   ```

2. **Create and activate a virtual environment**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   pip install -r training\requirements.txt
   ```

4. **Configure environment variables**
   ```powershell
   Copy-Item .env.example .env
   ```
   Fill in `.env` with real values if using document/history extraction:
   - `OPENAI_API_KEY` — from OpenAI, or set `AI_PROVIDER=xai` and `GROK_API_KEY` for xAI
   - `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` — from the shared Supabase project (ask a teammate for access, or see below). The backend requires the **service_role** key, not the anon key — patient data is locked down to service_role-only access (see `supabase_schema.sql`). Never share this key outside the team or commit it.

5. **Choose a voice provider**

   For API-key-free local voice:

   ```env
   VOICE_PROVIDER=local
   LOCAL_WHISPER_MODEL=small
   LOCAL_WHISPER_DEVICE=auto
   ```

   OpenAI and Bhashini settings are also available in `.env.example`.

6. **Run the API**
   ```powershell
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```
   API docs available at `http://127.0.0.1:8000/docs`.

   The AI provider defaults to OpenAI. To use Grok for the existing extraction
   endpoints, set `AI_PROVIDER=xai`,
   `GROK_API_KEY`, and optionally `AI_MODEL` (for example,
   `grok-4.6`) in `.env`. Both providers use the same
   OpenAI-compatible client; no API key is committed to the repository.

   The local question endpoint loads Qwen lazily on its first request. The
   base model does not require an external AI API key, but its model weights
   must be available.

## Application workflow

1. Patient logs in with a patient ID.
2. The app loads saved allergies, medicines, conditions, and previous reports.
3. The app asks whether previous history exists before deeper questioning.
4. The patient uploads a prescription/report and describes the problem by text
   or voice.
5. The AI asks one focused follow-up question at a time in English, Hindi, or
   Hinglish.
6. The app sends the complete transcript and extracted documents to
   `/generate-summary`.
7. The summary is saved to Supabase and appears in the doctor's waiting room.
8. The doctor reviews, edits, and adds diagnosis/treatment information.

## API usage

Open Swagger UI at `http://127.0.0.1:8000/docs`.

- `POST /extract-history` — body `{"transcript": "..."}`, extracts structured clinical history from a text transcript.
- `POST /ask-clinical-question` — uses local Qwen to ask one safe follow-up question.
- `POST /extract-from-image` — multipart file upload, extracts structured clinical history from a prescription/document image (uploaded to Supabase Storage first, then read by the OpenAI Vision model).
- `POST /converse` — maintains an adaptive one-question-at-a-time interview.
- `POST /generate-summary` — combines transcript and document data into a saved doctor-reviewable summary.
- `POST /voice/transcribe` — converts audio to text.
- `POST /voice/patient-assistant` — converts patient audio and returns the next question.
- `POST /voice/speak` — converts text to playable audio.
- `POST /doctor/voice-note` — converts doctor dictation to text.
- `POST /patient/profile` and `GET /patient/profile/{patient_id}` — save/load allergies and long-term context.
- `POST /patient/report` and `GET /patient/reports/{patient_id}` — save/load previous reports.
- `POST /patient/intake-start` — asks about previous history first.
- `GET /api/v1/waiting-room` — doctor waiting room.
- `GET /api/v1/priority-alerts` — unacknowledged urgent cases.
- `PATCH /patient-histories/{history_id}` — doctor edits.
- `POST /doctor/approved-case` — stores a doctor-reviewed learning case.
- `GET /health` — liveness check.

Example:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/ask-clinical-question `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"transcript":"mere pet mae dard hae"}'
```

The first local-model request loads Qwen and may take a few seconds. The adapter is trained for question phrasing, not diagnosis, treatment, or clinical decision-making.

## Voice assistant support

The backend includes voice helpers for patient and doctor workflows:

- `POST /voice/transcribe` — upload an audio file and get text transcription
- `POST /voice/speak` — convert text into spoken audio
- `POST /voice/patient-assistant` — speak the patient's problem, transcribe it, and return the next question
- `POST /doctor/voice-note` — transcribe a doctor's voice note or dictation

Use `VOICE_PROVIDER=local` for a self-hosted assistant with faster-whisper speech recognition and pyttsx3 speech synthesis. Install the dependencies with `pip install -r requirements.txt`; the first Whisper request downloads the selected model. Set `LOCAL_WHISPER_MODEL=small` (or `base` for less memory) and optionally `LOCAL_WHISPER_DEVICE=cuda`. OpenAI and Bhashini remain available with their respective provider settings.

This repository also includes a safer, controlled profile and review flow:

- `POST /patient/profile` — store allergies, chronic conditions, and medicine history for a patient
- `GET /patient/profile/{patient_id}` — fetch the known profile to avoid repeating allergy questions
- `POST /patient/report` — save a previous medical report or past-visit summary
- `GET /patient/reports/{patient_id}` — fetch earlier reports so follow-up questions can use past context
- `POST /patient/intake-start` — ask whether previous medical history exists before deeper questioning
- `POST /doctor/approved-case` — store a doctor-reviewed case for a curated training dataset
- `python training/build_doctor_case_dataset.py` — generate a small JSONL dataset from approved cases for future model improvement

Important: automatic self-training from every live patient interaction is not enabled in this repo. Approved, sanitized cases are stored for controlled review and retraining, which is the safer healthcare pattern.

## Learn how training works

For a beginner-friendly, step-by-step explanation of the dataset, tokenizer,
fine-tuning, LoRA adapter, training settings, inference flow, and limitations,
see [`training/README.md`](training/README.md#how-the-model-was-trained-class-10-explanation).

For the three-person frontend implementation plan, screen layouts, API
contracts, voice integration, and end-to-end acceptance test, see
[`FRONTEND_README.md`](FRONTEND_README.md).

## Datasets and training

The repository includes:

- `trilingual_clinical_conversation_questions.xlsx` — English, Hindi, and
  Hinglish question examples.
- `bilingual_clinical_conversation_questions.xlsx` — earlier English/Hindi
  workbook with generated Hinglish variants.
- `training/dataset/*.jsonl` — prescription image/transcription splits and
  curated doctor-approved examples.
- `training/train_qwen_qlora.py` — LoRA fine-tuning script.
- `training/build_doctor_case_dataset.py` — builds data only from
  doctor-reviewed cases.

Training is not performed automatically on every live patient interaction.
Doctor approval and sanitization are required before cases enter a future
training run.

## Database (Supabase)

The database schema is defined in [`supabase_schema.sql`](supabase_schema.sql):
- `patient_histories` table — stores every extracted clinical history.
- `medical_documents` Storage bucket — stores uploaded prescription/document images.
- RLS is enabled on both with no anon/authenticated policies — only the backend's `service_role` key can read or write. The frontend must go through the FastAPI backend, never call Supabase directly.

To stand up a Supabase project from scratch: create a project at [supabase.com](https://supabase.com), open the SQL Editor, and run the contents of `supabase_schema.sql`. Then put that project's URL and **service_role** key (Dashboard -> Settings -> API) into your `.env` as `SUPABASE_SERVICE_ROLE_KEY`.

If the team is sharing a single Supabase project, ask whoever created it for the `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` instead of creating your own — this key is sensitive, so share it privately (not in Slack/GitHub), not by committing it anywhere.

## Project Structure

```
claude.md.md          Architecture rules, tech stack, and JSON data contracts
main.py                FastAPI app and API routes
local_bilingual_model.py Lazy loader for the local Qwen bilingual adapter
bilingual_clinical_conversation_questions.xlsx / trilingual_clinical_conversation_questions.xlsx Training question datasets
training/               Dataset preparation, training scripts, adapters, and metrics
frontend/README.md     Three-person frontend implementation and API integration guide
supabase_schema.sql     Database schema, storage bucket, and RLS policies
requirements.txt       Python dependencies
.env.example           Required environment variables (copy to .env)
```
