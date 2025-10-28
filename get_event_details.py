import time
from google import genai
from google.genai import types 
from perplexity import Perplexity
from perplexity.types import SearchCreateParams
import re
import os
import requests
import urllib
import json
from json_repair import repair_json
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image
from PIL import Image, UnidentifiedImageError
import io
from utils import llm_response_gemini_genai
from serpapi import GoogleSearch

# load_dotenv()

@st.cache_resource
def setup_gcp_credentials():
    """Sets up Google Cloud credentials from Streamlit Secrets."""
    try:
        # Check if running in Streamlit Cloud
        if "GCP_CREDENTIALS" in st.secrets:
            # Create a temporary file with the credentials
            
            creds_dict = dict(st.secrets["GCP_CREDENTIALS"])
            with open("gcp_key.json", "w") as f:
                json.dump(creds_dict, f)
            # Point the library to this temporary file
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"
        # If not on Streamlit Cloud, it will fall back to local gcloud auth
    except Exception as e:
        st.error(f"Failed to set up GCP credentials: {e}")

# Call the setup function at the start
setup_gcp_credentials()

GOOGLE_CLOUD_PROJECT="traverse-project-421916"
GOOGLE_GENAI_USE_VERTEXAI=True
GOOGLE_CLOUD_LOCATION="global"

genai_client = genai.Client(
    http_options=types.HttpOptions(api_version="v1"), 
    vertexai=True, 
    project=GOOGLE_CLOUD_PROJECT, 
    location=GOOGLE_CLOUD_LOCATION
)

# You can reuse this function from your previous script

def convert_to_json(inp, response_format: dict):

    conversation = [
        {"role": "system", "content": f"Convert the input text to json in the exact same format as below:\n\n{response_format}"},
        {"role": "user", "content": f"INPUT TEXT:\n{inp}"}
    ]

    response = llm_response_gemini_genai(conversation=conversation)
    # print(f"Response from LLM: {json.dumps(response, indent=4, default=str)}")
    return response


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
 
def get_page_content(url):
    """Fetches the HTML content of a URL if it's accessible."""
    if not url or not url.startswith('http'):
        return None
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=3, allow_redirects=True)
        if 200 <= response.status_code < 300:
            print(f"Successfully fetched content from {url}")
            return response.text
        else:
            print(f"Failed to fetch {url}, status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Request failed for {url}. Error: {e}")
        return None

# --- NEW HELPER: LLM-based Content Validation ---
def get_content_validation_prompt(html_content, event_name, destination):
    """Generates a prompt to ask an LLM to validate the page content."""
    # Truncate content to avoid excessive token usage
    truncated_html = html_content
    return f"""You are a content validation expert. Your task is to determine if the provided HTML belongs to a webpage that is a valid booking, registration, or official information page for a specific event.

    **Event Details:**
    - **Event Name:** "{event_name}"
    - **Location:** "{destination}"

    **HTML Content to Analyze:**
    ---
    {truncated_html}
    ---

    **Instructions:**
    Analyze the HTML. Does this page clearly represent the event mentioned above? Look for the event name, location, dates, and terms like "Book Tickets," "Register," or pricing information. A news article or a general listings page is NOT valid.

    **Response:**
    Respond with a single word: **YES** or **NO**.
    """

def is_booking_page_valid_llm(html_content, event_name, destination):
    """Uses an LLM to confirm if the HTML content is a valid booking page."""
    print("Validating page content with LLM...")
    prompt = get_content_validation_prompt(html_content, event_name, destination)
    conversation = [
        {"role": "system", "content": "You are a web page analyst that responds with only YES or NO."},
        {"role": "user", "content": prompt}
    ]
    response = llm_response_gemini_with_google_search_grounding(conversation, structured=False)
    
    # Check if the response is a string and is "YES"
    if isinstance(response, str) and response.strip().upper() == "YES":
        print("✅ Content validation PASSED.")
        return True
    else:
        print(f"❌ Content validation FAILED. LLM response: {response}")
        return False

def extract_image_from_html(html_content):
    """Extracts a high-quality image URL from HTML, prioritizing og:image."""
    soup = BeautifulSoup(html_content, 'html.parser')
    # Best case: Find the Open Graph image tag
    og_image = soup.find('meta', property='og:image')
    if og_image and og_image.get('content'):
        print(f"Found og:image: {og_image['content']}")
        return og_image['content']
    # Fallback: Find the first large-ish image (simple heuristic)
    for img in soup.find_all('img'):
        if img.get('src') and img.get('src').startswith('http'):
            return img['src']
    return None


def query_unsplash(query):
    UNSPLASH_API_KEY = st.secrets["UNSPLASH_API_KEY"]
    url = "https://api.unsplash.com/search/photos"
    
    start_time = time.time()
    # Parameters for the request
    try:
        params = {
            "client_id": UNSPLASH_API_KEY,
            "query": query,
            "per_page": 1,
            "order_by": "relevant"
        }

        # Send GET request to Unsplash API
        response = requests.get(url, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            if data['results']:
                # Extract relevant information from the first (top) result
                top_image = data['results'][0]
                image_url = top_image['urls']['regular']
                photographer = top_image['user']['name']
                
                # print(f"Top recommended image URL for '{query}': {image_url}")
                # print(f"Photographer: {photographer}")
                return image_url
            else:
                print(f"No results found for '{query}'")
                return ''
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return ''
    
    except Exception as e:
        print(f"Error in query_unsplash: {str(e)}")
        return ''
    
    finally:
        print(f"Total time taken by query_unsplash: {time.time() - start_time} seconds")



def save_data_to_file(data, filepath):
    """Saves the final data to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved all enriched data to '{filepath}'")
    except Exception as e:
        print(f"Failed to save data to '{filepath}': {e}")





def get_event_details_prompt(destination, event_name, start_date):
    """
    Generates a highly specific prompt to find detailed information for a single event,
    including a fallback Unsplash query.
    """
    prompt = f"""You are highly knowledgeable about happening events around the world. You are a meticulous data verification and enrichment agent. Your task is to find specific, accurate details for a known event and return them in a strict JSON format.

    **Objective:** Find the following details for the event provided using the most official sources.

    **Event to Research:**
    - **Event Name:** "{event_name}"
    - **Start Date:** "{start_date}"
    - **Location:** "{destination}"

    **Information to Find:**
    1.  **end_date**: The event's end date in YYYY-MM-DD format.
    2.  **venue_name**: The official name of the venue.
    3.  **full_address**: The complete physical address of the venue.
    4.  **description**: A brief, one or two-sentence beautiful summary eliciting interest in the event.
    5.  **unsplash_query**: A concise, effective search query for Unsplash.com to find a high-quality photo representing the event's theme. For example, for "Oktoberfest Goa," a good query is "beer festival beach celebration." For "Diwali in Jaipur," a query could be "Jaipur city lights festival." This is a fallback and should always be generated.

    **Crucial Instructions:**
    1.  **Strict JSON Object Only:** Your response MUST be a single JSON object `{{}}`.
    2.  **No Hallucination:** If you cannot find a piece of information, you MUST use `null`.
    
    **Mandatory JSON Response Format:**
    ```json
    {{
      "end_date": "The event's end date in YYYY-MM-DD format.",
      "venue_name": "The official name of the event venue.",
      "full_address": "The full, complete street address of the venue.",
      "description": "A concise, engaging summary of the event for a tourist.",
      "unsplash_query": "A creative and effective search term for Unsplash."
    }}
    ```
    """
    return prompt


def llm_response_pplx(conversation, error_response=None, max_retries=3, retry_delay=1):
    """Sends a request to Perplexity AI with an automatic retry mechanism."""
    pplx_url = "https://api.perplexity.ai/chat/completions"
    pplx_api_key = st.secrets["PERPLEXITY_API_KEY"]

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
        


def extract_json(text: str):
    """
    Extracts and parses JSON from a string.
    Handles:
      - Raw JSON strings
      - ```json ... ``` fenced JSON blocks
      - Text with extra characters before/after JSON
    Returns a Python dict or list.
    """
    #Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    #Try to extract JSON from ```json ... ``` or ``` ... ``` blocks
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```', text)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    
    return {"response": "Invalid JSON"}


    
def get_clean_page_text(url):
    """
    Fetches the content of a URL and returns clean, readable text.
    Strips out all HTML tags, scripts, and styles.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, allow_redirects=True, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Remove all script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            # Get text, strip whitespace, and join lines with a space
            text = ' '.join(soup.stripped_strings)
            # Truncate to a reasonable length to not overwhelm the LLM context window
            return text
        else:
            print(f"Failed to fetch {url}, status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def find_candidate_links_using_perplexity_search(event_name, destination):
    client = Perplexity()
    query = f"{event_name} at {destination} 2025 booking link"

    try:
        search_params = SearchCreateParams(
            query=query,
            max_results=5,
            # return_images=True,
            # return_snippets=True
        )

        search = client.search.create(**search_params)

        candidate_links = []
        for result in search.results:
            candidate_links.append(result.url)
            

        return candidate_links


    except Exception as e:
        print(f"Error fetching results for {event_name} at {destination}: {e}")
        return []
    

def find_images_with_serpapi(query: str, limit: int = 3) -> list[str]:
    """
    Performs a Google Image search using SerpApi and returns a list of image URLs.
    """
    print(f"Querying SerpApi for images with: '{query}'")
    serpapi_key = os.getenv("SERP_API_KEY")
    if not serpapi_key:
        print("Warning: SERPAPI_API_KEY not found. Skipping image search.")
        return []

    params = {
        "q": query,
        "engine": "google_images",
        "ijn": "0",  # Page number
        "api_key": serpapi_key,
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "images_results" in results:
            # Extract the URL of the full-sized original image
            image_urls = [result["original"] for result in results["images_results"]]
            return image_urls[:limit]
        else:
            return []
    except Exception as e:
        print(f"Error calling SerpApi: {e}")
        return []



def get_event_visuals(event_data: dict, source_url: str, destination: str, unsplash_query: str):
    """
    Given a validated source URL, finds a verified event image.
    If it fails, it falls back to querying Unsplash.

    Args:
        event_data (dict): The event object containing 'event_name', etc.
        source_url (str): The validated booking/info link for the event.
        destination (str): The location of the event.
        unsplash_query (str): The pre-generated query for Unsplash fallback.

    Returns:
        A dictionary containing either 'image_url' or 'unsplash_images'.
    """
    print(f"Starting visual asset search for '{event_data['event_name']}'")
    
    # --- Primary Path: Find and Verify Image from Source URL ---
    # html_content = requests.get(source_url, headers={'User-Agent': 'Mozilla/5.0'}).text
    # if html_content:
    event_name = event_data.get("event_name")
    query = f"{event_name} in {destination}"
    candidate_urls = find_images_with_serpapi(query = query)
    print(f"\n\ncandidate urls are {candidate_urls}")
    print(f"Found {len(candidate_urls)} candidate images on the page.")
    
    for img_url in candidate_urls:
        image_data = download_image_data(img_url)
        if image_data:
            if is_image_relevant_llm(image_data, event_data['event_name'], destination, description = event_data["description"]):
                # Success! We found a verified image.
                return {"image_url": img_url, "unsplash_images": None}

    # --- Fallback Path: Query Unsplash ---
    print(f"Could not find a verified image for '{event_data['event_name']}'. Falling back to Unsplash.")
    
    if not unsplash_query:
        # Generate a simple fallback query if one wasn't provided
        unsplash_query = f"{event_data['event_name']} {destination}"
        print(f"Generated a basic Unsplash query: '{unsplash_query}'")

    unsplash_results = query_unsplash(unsplash_query)
    return {"image_url": None, "unsplash_images": unsplash_results}


# --- Helper 1: Download Image Data ---
def download_image_data(url: str):
    """Attempts to download image data from a URL and checks its validity."""
    if not url: return None
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10, stream=True)
        if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
            image_data = response.content
            try:
                Image.open(io.BytesIO(image_data))
                print(f"Successfully downloaded and avalidated image from {url}")
                return {"data": image_data, "mime_type": response.headers['Content-type']}
            except UnidentifiedImageError:
                print(f"DOWNLOAD FAILED: Content from {url} is not a valid image, despite headers.")
                return None
    except (requests.RequestException, IOError) as e:
        print(f"Error downloading or verifying image {url}: {e}")
        return None

# --- Helper 2: Extract Candidate Image URLs ---
def extract_candidate_image_urls(html_content: str, base_url: str, limit: int = 10) -> list[str]:
    """Extracts a list of potential image URLs from HTML, prioritizing og:image."""
    soup = BeautifulSoup(html_content, 'html.parser')
    candidate_urls = set()
    
    og_image = soup.find('meta', property='og:image')
    if og_image and og_image.get('content'):
        candidate_urls.add(og_image['content'])

    for img in soup.find_all('img'):
        src = img.get('src')
        if src and not src.startswith('data:image'):
            absolute_url = urllib.parse.urljoin(base_url, src)
            candidate_urls.add(absolute_url)
    
    return list(candidate_urls)[:limit]



import base64

def is_image_relevant_llm(image_data: dict, event_name: str, destination: str, description: str) -> bool:
    """Sends image data to Gemini to verify its relevance to the event."""
    print(f"Verifying image relevance for '{event_name}' with Gemini...")
    
    prompt = f"""You are a visual content moderator for a travel website. Your task is to determine if the provided image is a high-quality, relevant, and appropriate promotional image for the event described below.

**Event Context:**
- **Event Name:** "{event_name}"
- **Location:** "{destination}"
- **Description:** "{description}"

**Evaluation Criteria:**
1. **Relevance:** Does the image clearly relate to the event's theme (e.g., music, art, festival)?
2. **Quality:** Is it a clear, high-resolution photo? It must NOT be a small logo, a website banner with lots of text, or a blurry picture.
3. **Appropriateness:** Is it a promotional photo, a poster, or a picture from a past event? It must not be an advertisement for something else.

**Response Format:**
Respond with a single word: **YES** or **NO**.
"""
    
    try:
        # Convert bytes to base64 if needed
        if isinstance(image_data["data"], bytes):
            image_b64 = base64.b64encode(image_data["data"]).decode('utf-8')
        else:
            image_b64 = image_data["data"]
        
        # Create the image part correctly
        image_part = types.Part.from_bytes(
            data=image_b64,
            mime_type=image_data["mime_type"]
        )
        
        
        
        response = genai_client.models.generate_content(
            model = "gemini-2.5-flash", 
            contents = [prompt, image_part])
        
        verdict = response.text.strip().upper()
        print(f"Gemini response: {verdict}")
        
        if "YES" in verdict:
            print("✅ Gemini VERIFIED the image is relevant.")
            return True
        else:
            print(f"❌ Gemini REJECTED the image. Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Gemini image verification failed: {e}")
        return False



def enrich_event_with_llm_validation(event, destination):
    """
    The new worker function.
    1. Gets candidate links from a search API.
    2. Fetches the page content for each.
    3. Validates the *content* with an LLM until one passes.
    """
    event_name = event.get("event_name")
    if not event_name:
        event['booking_link'] = None
        return event

    candidate_links = find_candidate_links_using_perplexity_search(event_name, destination)
    print(f"candidate links: {candidate_links}")
    if not candidate_links:
        event['booking_link'] = None
        return event

    final_booking_link = None
    visual_assets = {}
    for link in candidate_links:
        # Step 3a: Fetch the clean text from the page
        page_text = get_clean_page_text(link)
        
        # Step 3b: If we got text, send it to the LLM for validation
        if page_text:
            if is_booking_page_valid_llm(html_content=page_text, event_name=event_name, destination=destination):
                final_booking_link = link

                visual_assets = get_event_visuals(
                event_data=event,
                source_url=final_booking_link,
                destination=destination,
                unsplash_query=event.get("unsplash_query")
                )
                break # Success! Stop searching.

    if final_booking_link is None:
        print("No valid URL found, falling back to Unsplash directly.")
        unsplash_query = event.get("unsplash_query", f"{event['event_name']} {destination}")
        visual_assets["unsplash_images"] = query_unsplash(unsplash_query)
    
    event['booking_link'] = final_booking_link
    event["image_url"] = visual_assets.get("image_url")
    event["unsplash_images"] = visual_assets.get("unsplash_images")

    return event



def enrich_single_event(destination, event_data):
    """
    Enriches an event by getting all details in a single LLM call, then locally
    validating the source URL and deciding whether to use a real image or the
    fallback Unsplash query.
    """
    event_name = event_data.get("event_name")
    start_date = event_data.get("start_date")

    if not event_name or not start_date:
        print(f"Skipping event due to missing name or start date: {event_data}")
        return event_data

    print(f"--- Enriching details for: '{event_name}' in {destination} ---")

    # Step 1: Single LLM call to get all candidate details, including unsplash_query
    system_prompt = get_event_details_prompt(destination, event_name, start_date)
    user_prompt = f"Find detailed information for '{event_name}' starting on {start_date} in {destination}."
    conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    pplx_response = llm_response_pplx(conversation)
    event_details = extract_json(pplx_response)
    print(f"candidate_details: {json.dumps(event_details, indent=4)}")
    

    if not isinstance(event_details, dict):
        print(f"Failed to get base details for '{event_name}'. Aborting enrichment.")
        return event_data 

    # Step 2: Validate the source_url and try to extract a real image

    event_data.update(event_details)
    
    event_data = enrich_event_with_llm_validation(event = event_data, destination=destination)
    

    return event_data

if __name__ == "__main__":
    INPUT_FILENAME = "poc_events_results/events_gemini.json"
    OUTPUT_FILENAME = "poc_events_results/events_fully_enriched_using_pplx_with_images.json"

    # Step 1: Load the initial events data
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            destinations_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not read or parse the input file '{INPUT_FILENAME}'. Error: {e}")
        exit()

    all_enriched_data = []

    # Step 2: Iterate through each destination and its events
    for destination_entry in destinations_data[25:]:
        destination_name = destination_entry.get("destination")
        events = destination_entry.get("response", [])
        
        if not destination_name or not events:
            continue

        print(f"--- Starting enrichment process for {len(events)} events in {destination_name} ---")
        
        enriched_events_list = []
        for event in events:
            enriched_event = enrich_single_event(destination_name, event)
            if enriched_event:
                enriched_events_list.append(enriched_event)
            # Optional: Add a small delay to avoid hitting rate limits if you have many events
            time.sleep(1) 

        # Create a new structure for the output file
        destination_entry["response"] = enriched_events_list
        all_enriched_data.append(destination_entry)

    # Step 3: Save the newly enriched data to a different file
    save_data_to_file(all_enriched_data, OUTPUT_FILENAME)

    print("--- All destinations have been processed and enriched. ---")


    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#



    