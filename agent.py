from typing import Literal, List, Any
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from prompt_library.prompt import system_prompt
from utils.llms import LLMModel
from toolkit.toolkits import *
import json
import re
try:
    from groq import BadRequestError as GroqBadRequestError
except ImportError:
    GroqBadRequestError = Exception

# router typed dict unchanged
class Router(TypedDict):
    next: Literal["information_node", "booking_node", "FINISH"]
    reasoning: str

# AgentState now includes a 'turns' counter to avoid infinite recursion
class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    id_number: int
    next: str
    query: str
    current_reasoning: str
    turns: int  # new field to count supervisor entries
    last_node: str  # track last node visited to detect loops

class DoctorAppointmentAgent:
    def __init__(self):
        llm_model = LLMModel()
        self.llm_model = llm_model.get_model()

    def supervisor_node(self, state: AgentState) -> Command[Literal['information_node', 'booking_node', '__end__']]:
        """
        Supervisor decides which node to call next based on the LLM structured output.
        - preserves conversation history (doesn't clobber it)
        - increments a turns counter and ends if too many iterations
        """

        print("**************************below is my state right after entering****************************")
        print(state)

        # initialize turns if not present
        turns = state.get("turns", 0)
        turns += 1
        last_node = state.get("last_node", "")

        # Safety stop: avoid infinite loops from LLM uncertainty
        MAX_TURNS = 8
        if turns > MAX_TURNS:
            print(f"Reached max turns ({MAX_TURNS}). Ending.")
            return Command(goto=END, update={"turns": turns, "last_node": "supervisor"})
        
        # Check if we've been to the same node 3+ times in a row (likely a loop)
        if turns > 3 and last_node and last_node == state.get("next", ""):
            print(f"Detected loop: visited {last_node} multiple times. Finishing.")
            return Command(
                goto=END,
                update={
                    "turns": turns,
                    "last_node": "supervisor",
                    "current_reasoning": "Loop detected"
                }
            )

        # Build prompt messages (system + user id + current conversation)
        # Convert LangChain messages to dict format for structured output
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user's identification number is {state['id_number']}"}
        ]
        
        # Convert LangChain message objects to dict format
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, dict):
                messages.append(msg)

        print("***********************this is my message*****************************************")
        print(messages)

        # Extract the user's query (only if the last message is a user message)
        query = ""
        if len(state['messages']) >= 1:
            # If messages contains HumanMessage objects, try to extract last human content
            last = state['messages'][-1]
            if isinstance(last, HumanMessage):
                query = last.content
            else:
                # fallback: if list contains dicts or others, try best-effort extraction
                try:
                    query = state['messages'][-1].content
                except Exception:
                    query = ""

        print("************below is my query********************")
        print(query)

        # Check if the last AI message already answered the user's query
        # This prevents infinite loops when the same response is repeated
        if len(state["messages"]) >= 2:
            last_msg = state["messages"][-1]
            second_last_msg = state["messages"][-2] if len(state["messages"]) >= 2 else None
            
            # If last message is AI and second last is also AI (same response repeated), finish
            if isinstance(last_msg, AIMessage) and isinstance(second_last_msg, AIMessage):
                if last_msg.content.strip() == second_last_msg.content.strip():
                    print("Detected duplicate response. Finishing.")
                    return Command(
                        goto=END,
                        update={
                            "next": "FINISH",
                            "query": query,
                            "current_reasoning": "Duplicate response detected",
                            "messages": state["messages"],
                            "turns": turns,
                        },
                    )
            
            # If last message is AI and contains availability info, check if we should finish
            if isinstance(last_msg, AIMessage):
                last_content_upper = last_msg.content.upper()
                
                # Check if appointment was successfully booked/cancelled/rescheduled
                # The tool returns "Appointment successfully booked." - detect this pattern
                booking_complete = False
                
                # Check for success patterns (order matters - check most specific first)
                if "APPOINTMENT SUCCESSFULLY BOOKED" in last_content_upper:
                    booking_complete = True
                elif "SUCCESSFULLY BOOKED" in last_content_upper:
                    booking_complete = True
                elif "SUCCESSFULLY CANCELLED" in last_content_upper or "CANCELLED SUCCESSFULLY" in last_content_upper:
                    booking_complete = True
                elif "SUCCESSFULLY RESCHEDULED" in last_content_upper or "RESCHEDULED SUCCESSFULLY" in last_content_upper:
                    booking_complete = True
                elif "YOUR APPOINTMENT IS BOOKED" in last_content_upper:
                    booking_complete = True
                elif "APPOINTMENT BOOKED" in last_content_upper and "SUCCESSFULLY" in last_content_upper:
                    booking_complete = True
                elif "SEE YOU THEN" in last_content_upper:  # Indicates booking completion
                    booking_complete = True
                
                # Also check if query signals booking completion
                if query == "BOOKING_COMPLETE" or booking_complete:
                    print("Booking action completed successfully. Finishing.")
                    return Command(
                        goto=END,
                        update={
                            "next": "FINISH",
                            "query": query,
                            "current_reasoning": "Booking action completed",
                            "messages": state["messages"],
                            "turns": turns,
                            "last_node": "supervisor",
                        },
                    )
                
                # Check if the AI already provided availability information or booking confirmation
                has_availability_info = any(keyword in last_content_upper for keyword in [
                    "AVAILABLE", "NOT AVAILABLE", "ISN'T AVAILABLE", "ISN'T", 
                    "ALTERNATIVES", "SLOTS", "OPTIONS", "BOOKED", "SUCCESSFULLY",
                    "WHICH ONE", "WOULD YOU LIKE", "PREFER"
                ])
                
                # If we have availability info and turns > 3, likely already answered
                if has_availability_info and turns > 3:
                    # Check if user hasn't provided a new question (query is empty or same as before)
                    if not query or query == state.get("query", ""):
                        print("Availability info provided and no new query. Finishing.")
                        return Command(
                            goto=END,
                            update={
                                "next": "FINISH",
                                "query": query,
                                "current_reasoning": "Availability info already provided",
                                "messages": state["messages"],
                                "turns": turns,
                            },
                        )
                
                # If we've been through supervisor many times and last message is AI, finish
                if turns > 5 and isinstance(last_msg, AIMessage):
                    print(f"High turn count ({turns}) with AI response. Finishing to prevent loop.")
                    return Command(
                        goto=END,
                        update={
                            "next": "FINISH",
                            "query": query,
                            "current_reasoning": "High turn count - preventing loop",
                            "messages": state["messages"],
                            "turns": turns,
                        },
                    )

        # Ask the LLM for routing decision using structured output
        # Use a fallback approach for Groq models that don't support tool calling reliably
        try:
            response = self.llm_model.with_structured_output(Router).invoke(messages)
            goto = response["next"]
            reasoning = response.get("reasoning", "")
        except (GroqBadRequestError, Exception) as e:
            # Fallback: Use simple inference based on conversation
            print(f"Structured output failed: {e}. Using inference fallback.")
            
            # Simple inference based on last messages
            goto = "FINISH"  # Default to finish
            reasoning = "Structured output failed, defaulting to FINISH"
            
            # Check last few messages to infer intent
            if len(state["messages"]) > 0:
                last_user_msg = None
                last_ai_msg = None
                
                for msg in reversed(state["messages"]):
                    if isinstance(msg, HumanMessage) and last_user_msg is None:
                        last_user_msg = msg.content.lower()
                    elif isinstance(msg, AIMessage) and last_ai_msg is None:
                        last_ai_msg = msg.content.lower()
                    if last_user_msg and last_ai_msg:
                        break
                
                # Infer routing based on content
                if last_user_msg:
                    if any(word in last_user_msg for word in ["book", "appointment", "schedule", "cancel", "reschedule"]):
                        goto = "booking_node"
                        reasoning = "User wants to book/cancel/reschedule"
                    elif any(word in last_user_msg for word in ["available", "availability", "slot", "time", "when"]):
                        goto = "information_node"
                        reasoning = "User asking about availability"
                    elif "successfully" in last_ai_msg or "booked" in last_ai_msg:
                        goto = "FINISH"
                        reasoning = "Appointment already booked"
                elif last_ai_msg and ("available" in last_ai_msg or "slot" in last_ai_msg):
                    # If AI just provided availability, check if user wants to book
                    goto = "FINISH"
                    reasoning = "Availability provided, waiting for user response"
            
            # Use the inferred routing (goto and reasoning are already set above)
            print(f"Using inference fallback: goto={goto}, reasoning={reasoning}")

        print("********************************this is my goto*************************")
        print(goto)
        print("********************************")
        print(reasoning)

        # Map "FINISH" from LLM to END for langgraph
        if goto == "FINISH":
            return Command(
                goto=END,
                update={
                    "next": "FINISH",
                    "query": query,
                    "current_reasoning": reasoning,
                    # preserve full conversation and update turns
                    "messages": state["messages"],
                    "turns": turns,
                    "last_node": "supervisor",
                },
            )

        # Otherwise, route to the requested node. Preserve conversation history;
        # include the query in state so downstream nodes can read it.
        return Command(
            goto=goto,
            update={
                "next": goto,
                "query": query,
                "current_reasoning": reasoning,
                "messages": state["messages"],  # <-- very important: DO NOT overwrite
                "turns": turns,
                "last_node": goto,  # track which node we're going to
            },
        )

    def information_node(self, state: AgentState) -> Command[Literal['supervisor']]:
        """
        Specialized node to check availability (uses tools).
        Appends the agent's reply to messages and returns to supervisor.
        """
        print("*****************called information node************")
        
        # Check if we've already answered this query to prevent repetition
        if len(state["messages"]) >= 2:
            last_ai_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and hasattr(msg, 'name') and msg.name == "information_node":
                    last_ai_msg = msg
                    break
            
            # If we already provided availability info and user hasn't asked a new question, skip
            if last_ai_msg and ("AVAILABLE" in last_ai_msg.content.upper() or "NOT AVAILABLE" in last_ai_msg.content.upper()):
                # Check if the last user message is the same as before
                last_user_msg = None
                for msg in reversed(state["messages"]):
                    if isinstance(msg, HumanMessage):
                        last_user_msg = msg
                        break
                
                # If user hasn't provided new input, return to supervisor to finish
                if last_user_msg and last_user_msg.content == state.get("query", ""):
                    return Command(
                        update={"messages": state["messages"]},
                        goto="supervisor",
                    )

        system_prompt_local = (
            "You're a friendly receptionist helping someone book an appointment. "
            "Write like you're sending a text message - simple, casual, and easy to read.\n\n"
            
            "EXAMPLES OF GOOD RESPONSES:\n\n"
            
            "If time NOT available:\n"
            "'Sorry, 8 PM isn't available on that day. But I have these times: Dr. Emily Johnson at 8 AM, 8:30 AM, or 10 AM. Dr. John Doe at 8 AM, 9 AM, or 9:30 AM. Which one works for you?'\n\n"
            
            "If doctor not specified:\n"
            "'I have two doctors available: Dr. Emily Johnson has slots at 8 AM, 8:30 AM, or 10 AM. Dr. John Doe has 8 AM, 9 AM, or 9:30 AM. Which doctor do you prefer?'\n\n"
            
            "If time IS available:\n"
            "'Yes! That time is available. So that's [date] at [time] with Dr. [name], right? Should I go ahead and book it?'\n\n"
            
            "CRITICAL RULES:\n"
            "- Write in plain text only - NO markdown, NO bold, NO dashes, NO bullet points\n"
            "- Write like a text message - short sentences, casual tone\n"
            "- Times: Use '8 AM' not '8:00 AM', '8 PM' not '20:00'\n"
            "- Keep it under 80 words\n"
            "- Be friendly but not overly formal\n"
            "- Current year is 2024\n"
            "- Don't repeat information you already gave\n"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_local),
                ("placeholder", "{messages}"),
            ]
        )

        information_agent = create_react_agent(
            model=self.llm_model,
            tools=[check_availability_by_doctor, check_availability_by_specialization, check_specific_slot],
            prompt=prompt,
        )

        # call the agent; it expects the same state shape (we've preserved it)
        result = information_agent.invoke(state)

        # Pull the assistant message (best effort)
        assistant_msg = ""
        try:
            assistant_msg = result["messages"][-1].content
        except Exception:
            assistant_msg = str(result)

        # Append the assistant's reply to conversation and go back to supervisor
        new_messages = state["messages"] + [AIMessage(content=assistant_msg, name="information_node")]

        return Command(
            update={
                "messages": new_messages,
                # clear query after processing so supervisor doesn't re-process the same user input
                "query": "",
            },
            goto="supervisor",
        )

    def booking_node(self, state: AgentState) -> Command[Literal['supervisor']]:
        """
        Specialized node to set/cancel/reschedule appointments.
        """
        print("*****************called booking node************")

        patient_id = state.get('id_number', '')
        system_prompt_local = (
            f"You are a friendly assistant helping patients book appointments. "
            f"The patient's ID number is: {patient_id}\n"
            f"Your job is to EXTRACT information from the conversation and BOOK the appointment IMMEDIATELY when user confirms.\n\n"
            
            "**CRITICAL - EXTRACT INFORMATION FROM CONVERSATION:**\n"
            "- Read the ENTIRE conversation history CAREFULLY to find: date, time, and doctor name\n"
            f"- The patient ID is: {patient_id} (ALWAYS use this in set_appointment tool)\n"
            "- Look for time mentions like '10:30 AM', '2 PM', '10:30', etc. in EARLIER messages\n"
            "- Look for date mentions like 'August 7th', '07-08-2024', '08-07-2024', etc.\n"
            "- Look for doctor name mentions like 'Dr. John Doe', 'John Doe', etc.\n"
            "- DO NOT ask for information that's already in the conversation\n"
            "- DO NOT ask for patient ID - you already have it\n\n"
            
            "**WHEN USER SAYS 'YES' OR CONFIRMS:**\n"
            "1. If the conversation mentions a specific time (e.g., '10:30 AM', '2 PM'):\n"
            "   - Extract: date, time, doctor name from EARLIER in the conversation\n"
            "   - IMMEDIATELY call set_appointment tool - DO NOT ask for time again\n"
            "   - Format: date as 'DD-MM-YYYY HH:MM' (e.g., '07-08-2024 10:30' for 10:30 AM)\n"
            "   - Convert time: '10:30 AM' = '10:30', '2 PM' = '14:00', '8 AM' = '08:00'\n"
            "   - After booking, confirm: 'Great! Your appointment is booked for [date] at [time] with Dr. [name]. See you then!'\n\n"
            
            "2. If user confirms but time is unclear:\n"
            "   - Look back in conversation for the MOST RECENT time mentioned\n"
            "   - Use that time to book\n"
            "   - DO NOT ask user to choose again\n\n"
            
            "**EXAMPLES:**\n"
            "- Conversation: '10:30 AM on August 7th' + user says 'yes':\n"
            f"  → Call set_appointment(date='07-08-2024 10:30', id_number='{patient_id}', doctor_name='john doe')\n"
            "- Conversation: '2PM' + user says 'yes':\n"
            f"  → Call set_appointment(date='08-08-2024 14:00', id_number='{patient_id}', doctor_name='john doe')\n\n"
            
            "**ABSOLUTE RULES:**\n"
            "- If user says 'yes' or confirms, BOOK IMMEDIATELY - do not check availability again\n"
            "- DO NOT call check_specific_slot or check_availability tools if user already confirmed\n"
            "- DO NOT ask for time again if time was already mentioned\n"
            "- DO NOT ask 'which time' if time was already confirmed\n"
            "- Extract time from EARLIER conversation messages, not from asking again\n"
            "- When user confirms, look BACK in conversation for the time they agreed to\n"
            "- If you have date, time, doctor, and patient ID, call set_appointment IMMEDIATELY\n"
            "- Only check availability if user is ASKING about availability, not if they're CONFIRMING a booking\n"
            "- Current year is 2024\n"
            "- After successful booking, conversation is complete\n"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_local),
                ("placeholder", "{messages}"),
            ]
        )

        booking_agent = create_react_agent(
            model=self.llm_model,
            tools=[set_appointment, cancel_appointment, reschedule_appointment],
            prompt=prompt,
        )

        result = booking_agent.invoke(state)

        try:
            assistant_msg = result["messages"][-1].content
        except Exception:
            assistant_msg = str(result)

        new_messages = state["messages"] + [AIMessage(content=assistant_msg, name="booking_node")]
        
        # Check if booking was successful - if so, add a signal for supervisor to finish
        assistant_msg_upper = assistant_msg.upper()
        booking_success = any(keyword in assistant_msg_upper for keyword in [
            "SUCCESSFULLY BOOKED", "APPOINTMENT SUCCESSFULLY BOOKED", 
            "YOUR APPOINTMENT IS BOOKED", "BOOKED SUCCESSFULLY",
            "SUCCESSFULLY CANCELLED", "SUCCESSFULLY RESCHEDULED"
        ])
        
        # Update query to signal completion if booking was successful
        query_update = "BOOKING_COMPLETE" if booking_success else ""

        return Command(
            update={
                "messages": new_messages,
                "query": query_update,  # Signal to supervisor that booking is complete
            },
            goto="supervisor",
        )

    def workflow(self):
        """
        Build the state graph.
        START -> supervisor -> (information_node | booking_node | END)
        information_node/booking_node -> supervisor
        """
        self.graph = StateGraph(AgentState)
        self.graph.add_node("supervisor", self.supervisor_node)
        self.graph.add_node("information_node", self.information_node)
        self.graph.add_node("booking_node", self.booking_node)
        self.graph.add_edge(START, "supervisor")
        # information_node and booking_node implicitly return to supervisor via goto
        self.app = self.graph.compile()
        return self.app
