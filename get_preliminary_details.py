import time
from google import genai
from google.genai import types 
from datetime import datetime, timezone
import re
import os
import json
import requests
import streamlit as st
from json_repair import repair_json
from utils import llm_response_gemini_genai
from dotenv import load_dotenv
load_dotenv()

# @st.cache_resource
# def setup_gcp_credentials():
#     """Sets up Google Cloud credentials from Streamlit Secrets."""
#     try:
#         # Check if running in Streamlit Cloud
#         if "GCP_CREDENTIALS" in st.secrets:
#             # Create a temporary file with the credentials
#             creds_json_str = st.secrets["GCP_CREDENTIALS"]
#             with open("gcp_key.json", "w") as f:
#                 f.write(creds_json_str)
#             # Point the library to this temporary file
#             os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"
#         # If not on Streamlit Cloud, it will fall back to local gcloud auth
#     except Exception as e:
#         st.error(f"Failed to set up GCP credentials: {e}")

# # Call the setup function at the start
# setup_gcp_credentials()


GOOGLE_CLOUD_PROJECT="traverse-project-421916"
GOOGLE_GENAI_USE_VERTEXAI=True
GOOGLE_CLOUD_LOCATION="global"

genai_client = genai.Client(
    http_options=types.HttpOptions(api_version="v1"), 
    vertexai=True, 
    project=GOOGLE_CLOUD_PROJECT, 
    location=GOOGLE_CLOUD_LOCATION
)

# --- NEW AUTHENTICATION FUNCTION ---


def convert_to_json(inp, response_format: dict):

    conversation = [
        {"role": "system", "content": f"Convert the input text to json in the exact same format as below:\n\n{response_format}"},
        {"role": "user", "content": f"INPUT TEXT:\n{inp}"}
    ]

    response = llm_response_gemini_genai(conversation=conversation)
    # print(f"Response from LLM: {json.dumps(response, indent=4, default=str)}")
    return response


def llm_response_pplx(conversation, error_response=None, max_retries=3, retry_delay=1):
    """Sends a request to Perplexity AI with an automatic retry mechanism."""
    pplx_url = "https://api.perplexity.ai/chat/completions"
    pplx_api_key = os.getenv("PERPLEXITY_API_KEY")
    # print(f"perplexity api key: {pplx_api_key}")

    payload = {
        "model": "sonar-pro",
        "messages": conversation,
        "temperature": 0.2,
        "search_recency_filter": "month",
        "stream": False
    }
    headers = {"Authorization": f"Bearer {pplx_api_key}", "Content-Type": "application/json"}
    retries = 0
    while retries <= max_retries:
        try:
            response = requests.post(pplx_url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            openai_response = result["choices"][0]["message"]["content"]
            print("Successfully received response from Perplexity LLM.")
            return openai_response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
            retries += 1
            if retries > max_retries:
                print(f"Max retries ({max_retries}) exceeded. Last error: {str(e)}")
                return error_response
            backoff_delay = retry_delay * (2 ** (retries - 1))
            print(f"Attempt {retries}/{max_retries} failed: {str(e)}. Retrying in {backoff_delay} seconds...")
            time.sleep(backoff_delay)
        except Exception as e:
            print(f"An unexpected error occurred in llm_response_pplx: {str(e)}")
            return error_response


def llm_response_gemini_with_google_search_grounding(
    conversation,
    model_args={},
    error_response="error",
    max_retries=3,
    retry_delay=1,
    structured=True,
    response_format=None,
):
    """
    Send a request to Gemini API with Google Search grounding and retry mechanism.

    Returns:
        dict with {"status": "success"/"error", "response": <raw text or error>}
    """
    start_time = time.time()
    retries = 0
    response = None

    try:
        while retries <= max_retries:
            try:
                # Build system + user prompts
                system_prompt = conversation[0].get("content")
                contents = "\n".join([conv.get("content") for conv in conversation])

                # Attach Google search grounding tool
                grounding_tool = types.Tool(google_search=types.GoogleSearch())

                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                    temperature=model_args.get("temperature") or 0.2,
                    max_output_tokens=model_args.get("max_tokens", 8191),
                    response_mime_type="text/plain",
                    tools=[grounding_tool],
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=model_args.get("thinking_budget") or 128,
                        include_thoughts=model_args.get("include_thoughts") or False
                    )
                )

                response = genai_client.models.generate_content(
                    model=model_args.get("model") or "gemini-2.5-flash",
                    config=config,
                    contents=contents,
                )

                print(f"raw response: {response}")

                raw_text = response.text
                print(f"response text from gemini:\n{raw_text}")

                if structured:
                    matched_json = re.search(r"{.*}", raw_text, re.DOTALL)
                    if not matched_json and response_format:
                        json_response = convert_to_json(inp=raw_text, response_format=response_format)
                    else:
                        extracted_json = matched_json.group(0)
                        json_response = repair_json(extracted_json, return_objects=True)
                else:
                    json_response = raw_text
                return json_response

            except Exception as e:
                retries += 1
                if retries > max_retries:
                    print(f"Max retries ({max_retries}) exceeded. Last error: {str(e)}")
                    return error_response

                # exponential backoff
                backoff_delay = retry_delay * (2 ** (retries - 1))
                print(f"Attempt {retries}/{max_retries} failed: {str(e)}. Retrying in {backoff_delay} seconds...")
                time.sleep(backoff_delay)
    
    finally:
        end_time = time.time()
        print(f"Time taken in llm_response_gemini_with_google_search_grounding: {end_time - start_time} seconds")


def get_system_prompt(DESTINATION):
    """Generates the detailed system prompt for the LLM."""
    current_date = datetime.now(timezone.utc)
    current_date_str = current_date.strftime("%B %d, %Y")
    # This prompt is the same as before
    prompt = """You are an expert travel guide and your task is to find interesting events happening in a particular place.
    **Objective:** Today is {CURRENT_DATE}You are an automated data extraction agent. Your sole purpose is to find upcoming events happening only in future in {DESTINATION} that are suitable for tourists and return this information in a structured JSON format. You should try to get as many events which fall into the types of events mentioned below.

    **Potential Sources that can be used to collect the information:**
    * News Websites
    * City or country official tourism websites 
    * Tourism Calendars 
    * Event booking sites (Music tickets, sports tickets, etc) like BookMyShow, eventbrite, etc.
    * social media handles

    **Event Categories to Target:**
    *   Concerts and live music performances
    *   Major and local sporting events
    *   Cultural, arts, food,film, and music festivals
    *   Adventure sports tournaments (surfing competitions, paragliding festivals)
    *   Traditional ceremonies open to visitors
    *   Local theatre productions or musicals
    *   Stand-up comedy shows

    **Timeframe:**
    *   Search for events scheduled to happen in future within the **next 30 days** from today's date. If the next 30 days goes to the next month, i.e if someone searches for middle in janurary and 30 day after today happens to be a date in next month then you must look up events for next month also. If the current date is later half of a month, then focus more on events happening next month.

    **Mandatory Response Format:**
    *   Your entire response MUST be a single JSON array `[]`.
    *   Each element in the array must be a JSON object, representing one event.
    *   Do not include any text, explanation, or conversation before or after the JSON array.


    **JSON Object Structure:**
    Each JSON object must contain the following keys, precisely as named:

    {{
    "event_name": "The official name of the event. It should be of at max 10 words. It should clearly denote what type of event it is",
    "start_date": "The start date and time of the event in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). If the time is unavailable, use YYYY-MM-DD.",
    }}

    **Crucial Instructions:**
    1.  **Strictly Valid JSON Only:** Your entire response must be a single, syntactically perfect JSON array. Do NOT include comments within the JSON. It should be readily parsable into json.
    2.  **No Extraneous Text:** Do not write "Here is the JSON you requested," or any other conversational text. The response must begin with `[` and end with `]`.
    3.  **No Trailing Commas:** Ensure that there are no trailing commas after the last element in an object or the last object in the array.
    4.  **Consider as many sources as possible**: Few of the potential sources are already mentioned. Look beyond and further to find interesting events happening in the future for tourists to enjoy. 
    5.  **Find at max 5-6 events**: We don't need too many events, at max 10 high quality events are enough.
    6.  **No business or Academic Conferences**: The events we are trying to find which a traveler can attend purely for leisure purposes. Hence no technical or academic meet and workshops should be entertained.
    

    **Example for [DESTINATION]: Paris, France**
    [
    {{
        "event_name": "Paris International Art Fair",
        "start_date": "2025-10-15T10:00:00"
    }}
    ]
    """
    return prompt.format(DESTINATION=DESTINATION, CURRENT_DATE=current_date_str)


def get_conversation(DESTINATION):
    """Creates the conversation payload for the Perplexity API."""
    system_prompt = get_system_prompt(DESTINATION)
    user_prompt = f"Give me interesting events that are happening in {DESTINATION}"
    conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return conversation

def filter_past_events(events):
    """
    Programmatically filters out any events from a list that have already passed.
    This provides a final guarantee of correctness.
    """
    if not events:
        return []
    
    # Get the current date in UTC, ignoring time for a simple date comparison
    today = datetime.now(timezone.utc).date()
    future_events = []

    for event in events:
        start_date_str = event.get("start_date")
        if not start_date_str:
            continue # Skip events without a start date

        try:
            # The start_date could be a full ISO string or just YYYY-MM-DD
            # We only care about the date part for filtering.
            event_date = datetime.fromisoformat(start_date_str).date()
            if event_date >= today:
                future_events.append(event)
        except (ValueError, TypeError):
            # If the date format is invalid, we'll keep it for now and let the UI handle it,
            # but you could also choose to discard it here.
            print(f"Warning: Could not parse date '{start_date_str}' for event '{event.get('event_name')}'.")
            future_events.append(event) # Or `continue` to be stricter

    return future_events

def append_response_to_file(data_to_append, filepath):
    lock_path = filepath + ".lock"
    while os.path.exists(lock_path): time.sleep(0.1)
    with open(lock_path, 'w') as f: pass
    try:
        all_data = []
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                try: all_data = json.load(f)
                except json.JSONDecodeError: all_data = []
        destination_to_add = data_to_append.get("destination")
        all_data = [item for item in all_data if item.get("destination") != destination_to_add]
        all_data.append(data_to_append)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved data for '{destination_to_add}' to '{filepath}'")
    finally:
        if os.path.exists(lock_path): os.remove(lock_path)



def fetch_events_for_destination(destination, output_filepath=None, cron_job=False):
    """
    Main function with a switch for two different processing paths:
    - cron_job=True: Slow, high-quality, two-step enrichment.
    - cron_job=False: Fast, single-call enrichment with parallel validation.
    """
    start_time = time.time()
    print(f"Starting event data collection for: {destination} (Mode: {'Cron Job' if cron_job else 'Live Request'})")
    
    # Step 1: Get initial event list. For live requests, this prompt MUST ask for the booking link.
    conversation = get_conversation(destination) # Ensure your get_conversation/prompt logic handles this difference
    raw_response = llm_response_gemini_with_google_search_grounding(conversation, structured=False)
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_response.strip())

    # Parse JSON string into Python list
    events = json.loads(cleaned)
    print(f"number of evetns before fitlering: {len(events)}")
    events = filter_past_events(events = events)
    print(f"number of events after filtering: {len(events)}")

    print(f"type of raw_response: {type(events)}")

   

    # print(f"\n\n{raw_response}\n\n")
    
    
    end_time = time.time()
    response_time = end_time - start_time
    if output_filepath:
        final_output = {
            "destination": destination,
            "response": events,
            "response_time": response_time
        }
        # append_response_to_file(final_output, output_filepath)
    print(f"time took for {destination}: {response_time}")

    return events



if __name__ == "__main__":
    destinations = ["Jaipur", "Udaipur", "Goa", "Varanasi", "Kerala", "Rishikesh and Haridwar",  "Ladakh", "Mumbai",  "Hampi",  "Darjeeling", 
                "Spiti Valley",  "Kochi", "Andaman Islands", "Dehradun", "Pondicherry", "Maldives", "Thailand", "Tokyo", "Mauritus", "Switzerland", 
                "Hong Kong and Macau", "Abu Dhabi", "Rome", "London", "Istanbul", "Turkey", "Barcelona", "Bali", "Mykonos", "Kandy, Sri Lanka"]


    OUTPUT_FILENAME = "poc_events_results/events_response_gemini_genai.json"

    for destination in destinations:
        fetch_events_for_destination(destination=destination, output_filepath=OUTPUT_FILENAME, cron_job=True)
        
