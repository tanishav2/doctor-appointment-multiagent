# Doctor Appointment Multi‑Agent

A conversational doctor appointment system built with FastAPI and Streamlit, powered by LangChain/LangGraph and a Groq LLM. It acts like a friendly receptionist: checks availability, proposes alternatives, and books/cancels/reschedules appointments against a CSV schedule.

## Features
- Natural language conversation with conversation history
- Availability checks by doctor or specialization
- Specific slot validation with nearby alternatives
- Booking, cancellation, and rescheduling actions
- Loop protection and routing via a supervisor → information/booking nodes

## Tech Stack
- API: FastAPI (`main.py`)
- UI: Streamlit (`streamlit_ui.py`)
- Agents/Graph: LangChain + LangGraph (`agent.py`)
- LLM: Groq via `langchain_groq` (`utils/llms.py`)
- Tools: Pandas over `data/doctor_availability.csv` (`toolkit/toolkits.py`)

## Prerequisites
- Python 3.10+
- A Groq API key

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirments.txt`
   - Optionally: `pip install -e .` for editable local package
3. Create a `.env` file at the project root with:
   - `GROQ_API_KEY="<your_key_here>"`

Note: `.env` is already ignored by Git.

## Run
- Start the API (Streamlit expects port 8002):
  - `uvicorn main:app --reload --port 8002`
- Start the Streamlit UI in another terminal:
  - `streamlit run streamlit_ui.py`

## Using the API
POST `http://127.0.0.1:8002/execute`

Example body:
```json
{
  "id_number": 1000001,
  "message": "book me for 8 am with dr john doe on 21-08-2024",
  "conversation_history": [
    {"role": "user", "content": "is 8 am available?"},
    {"role": "assistant", "content": "yes, 8 AM is available with Dr. John Doe"}
  ]
}
```
Response includes the assistant’s latest message and normalized conversation history.

## Data Notes
- Availability is stored in `data/doctor_availability.csv` with columns like `date_slot`, `doctor_name`, `specialization`, `is_available`, `patient_to_attend`.
- Booking/cancel/reschedule operations mutate the CSV. If you need to reset, restore the file from a clean copy.

## Project Structure
- `main.py` — FastAPI endpoint `POST /execute`
- `streamlit_ui.py` — simple chat UI pointing to the API
- `agent.py` — supervisor + `information_node` + `booking_node` workflow
- `toolkit/toolkits.py` — tools for availability checks and appointment actions
- `utils/llms.py` — Groq client via `langchain_groq`
- `data/doctor_availability.csv` — demo schedule data
- `setup.py`, `requirments.txt` — packaging and dependencies

## Development Tips
- The UI’s `API_URL` is `http://127.0.0.1:8002/execute`. Keep the API on that port or update `streamlit_ui.py`.
- Ensure `GROQ_API_KEY` is set before starting the API/UI.
- When running locally, disable SSL verification is already handled for development.

## Credits
- Author: Sunny Savita (`setup.py`)