from google import genai
from google.genai import types
import json
from json_repair import repair_json
import time
import streamlit as st
import os


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


def llm_response_gemini_genai(conversation, output_key=None, mandatory_keys=[], error_response=None, model_args={}, max_retries=3, retry_delay=1, structured=True, stream=False):
    """
    Send a request to Gemini API with automatic retry mechanism.
    
    Args:
        conversation: List of conversation messages
        output_key: Optional key to extract from the response
        error_response: Default error response to return
        model_args: Additional model parameters
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (will be exponentially increased)
        structured: Boolean indicating if the response should be structured (JSON) or unstructured (text)
    
    Returns:
        Parsed JSON response or error_response
    """

    retries = 0
    response = None
    
    while retries <= max_retries:
        try:
            config = types.GenerateContentConfig(
                system_instruction=conversation[0].get("content"),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                temperature=model_args.get("temperature") or 0.2,
                max_output_tokens=model_args.get("max_tokens") or 8191,
                response_mime_type="application/json" if structured else "text/plain",
            )
            if "thinking_budget" in model_args:
                config.thinking_config = types.ThinkingConfig(thinking_budget=model_args.get("thinking_budget"))
            
            contents = "\n".join([conv.get("content") for conv in conversation[1:]])
            response = genai_client.models.generate_content(
                model=model_args.get("model") or "gemini-2.0-flash-001",
                config = config,
                contents=contents
            )

            # Extract the answer from the response
            content = response.text
            print(f"Response from Gemini LLM:\n{content}")
            
            try:
                content = repair_json(content)
                answer = json.loads(content)
            except json.JSONDecodeError:
                print(f"JSON decode error: {content}")
                raise ValueError("JSON decode error")
            
            if mandatory_keys:
                for key in mandatory_keys:
                    if key not in answer.keys():
                        print(f"key {key} not found in response")
                        raise ValueError(f"key {key} not found in response")
            
            if output_key is None:
                return answer
            
            if output_key in answer:
                return answer.get(output_key)
            else:
                raise ValueError(f"key {output_key} not found in response")
            
        except Exception as e:
            
            retries += 1
            if retries > max_retries:
                print(f"Max retries ({max_retries}) exceeded. Last error: {str(e)}")
                return error_response
            
            # Calculate exponential backoff delay
            backoff_delay = retry_delay * (2 ** (retries - 1))
            print(f"Attempt {retries}/{max_retries} failed: {str(e)}. Retrying in {backoff_delay} seconds...")
            
            time.sleep(backoff_delay)
