# import streamlit as st

# # Import the main functions from your other files
# from get_preliminary_details import fetch_events_for_destination
# from get_event_details import enrich_single_event

# # --- Streamlit Page Configuration ---
# st.set_page_config(
#     page_title="Event Finder AI",
#     page_icon="✈️",
#     layout="wide"
# )

# st.title("✈️ AI Event Finder")
# st.markdown("Enter a destination to find upcoming events. Click on an event to get the full details, including a verified booking link and a promotional image.")

# # --- Session State Initialization ---
# # This is crucial for a multi-step app to remember user's search and results.
# if 'destination' not in st.session_state:
#     st.session_state.destination = ""
# if 'events' not in st.session_state:
#     st.session_state.events = []
# if 'enrich_history' not in st.session_state:
#     st.session_state.enrich_history = {} # Tracks which events have been enriched

# # --- Search Bar and Buttons ---
# default_destinations = ["Paris", "Tokyo", "Goa", "Jaipur", "London"]
# cols = st.columns([2] + [1]*len(default_destinations)) # Give more space to the text input

# with cols[0]:
#     user_input = st.text_input("Enter Destination", value=st.session_state.destination, label_visibility="collapsed", placeholder="Enter Destination")

# # Logic to trigger a search
# def perform_search(destination_name):
#     st.session_state.destination = destination_name
#     st.session_state.enrich_history = {} # Reset enrichment on new search
#     with st.spinner(f"Searching for events in {destination_name}..."):
#         st.session_state.events = fetch_events_for_destination(destination_name)

# if cols[0].button("Search"):
#     perform_search(user_input)

# for i, dest in enumerate(default_destinations):
#     if cols[i+1].button(dest):
#         perform_search(dest)

# # --- Results Display Area ---
# if st.session_state.destination:
#     st.header(f"Upcoming Events in {st.session_state.destination}")
    
#     if not st.session_state.events:
#         st.warning("No events found for this destination. Try another one!")
#     else:
#         for i, event in enumerate(st.session_state.events):
            
#             # Use the event's name and date as a unique key for the expander
#             event_key = f"{event['event_name']}_{event['start_date']}"
            
#             with st.expander(f"**{event['event_name']}** - *Starting {event['start_date']}*"):
                
#                 # Check if this event has been enriched before
#                 if not st.session_state.enrich_history.get(event_key):
#                     if st.button("Find Out More", key=event_key):
#                         with st.spinner(f"Getting full details for {event['event_name']}... This may take a moment."):
#                             # Enrich the event and update the session state
#                             enriched_data = enrich_single_event(event_data = event, destination=st.session_state.destination)
#                             st.session_state.events[i] = enriched_data
#                             st.session_state.enrich_history[event_key] = True
#                         st.rerun() # Rerun to display the new details immediately
                
#                 # Display enriched details if they exist
#                 if st.session_state.enrich_history.get(event_key):
                    
#                     # --- FIX 1 & 2: Use columns for layout and fix image width ---
#                     # Create a 1:2 ratio layout (image column is 1/3, text is 2/3)
#                     col1, col2 = st.columns([2, 3])

#                     with col1:
#                         # The image will now fill the container of this smaller column
#                         if event.get('image_url'):
#                             st.image(event['image_url'], caption="Verified Event Image", use_container_width=True)
#                         elif event.get('unsplash_images'):
#                             st.image(event['unsplash_images'], caption="Image from Unsplash", use_container_width=True)
#                         else:
#                             st.write("No image available.")

#                     with col2:
#                         # Display Text Details in the second, wider column
#                         st.markdown(f"**Description:** {event.get('description', 'Not available.')}")
#                         st.info(f"""
#                             **Venue:** {event.get('venue_name', 'N/A')}  
#                             **Address:** {event.get('full_address', 'N/A')}  
#                             **End Date:** {event.get('end_date', 'N/A')}
#                         """)
                        
#                         # Display Link Button, making it fill the column width
#                         if event.get('booking_link'):
#                             st.link_button("➡️ Go to Booking/Info Page", event['booking_link'], use_container_width=True)
#                         else:
#                             st.warning("A verified booking link could not be found for this event.")


# In app.py

import streamlit as st

# Import the main functions from your other files
from get_preliminary_details import fetch_events_for_destination
from get_event_details import enrich_single_event

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Event Finder AI",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ AI Event Finder")
st.markdown("Enter a destination to find upcoming events. Click on an event to get the full details, including a verified booking link and a promotional image.")

# --- Session State Initialization ---
if 'destination' not in st.session_state:
    st.session_state.destination = ""
if 'events' not in st.session_state:
    st.session_state.events = []
if 'enrich_history' not in st.session_state:
    st.session_state.enrich_history = {}

# --- Search Bar and Buttons ---
default_destinations = ["Concerts in Paris", "Folk and Regional Music shows in Jaipur", "Operas happening in London"]
cols = st.columns([2] + [1]*len(default_destinations))

with cols[0]:
    user_input = st.text_input("Enter Destination", value=st.session_state.get('last_query', ''), label_visibility="collapsed", placeholder="e.g., 'Music festivals in Dubai'")

# --- MODIFIED perform_search FUNCTION ---
def perform_search(user_query: str):
    """Handles the new dictionary response from the backend."""
    st.session_state.last_query = user_query # Remember the last typed query
    st.session_state.enrich_history = {} # Reset enrichment on new search
    
    with st.spinner(f"Searching for events based on: '{user_query}'..."):
        # The backend function now returns a dictionary
        search_result = fetch_events_for_destination(query=user_query)
        
        # Correctly unpack the dictionary into session state
        st.session_state.destination = search_result.get("destination")
        st.session_state.events = search_result.get("events", [])
        
        # Provide feedback if no destination was found
        if not st.session_state.destination:
            st.warning(f"Could not identify a destination in your query: '{user_query}'. Please be more specific.")

if cols[0].button("Search"):
    if user_input: perform_search(user_input)

for i, dest in enumerate(default_destinations):
    if cols[i+1].button(dest):
        perform_search(dest)

# --- Results Display Area (This part now works correctly) ---
if st.session_state.destination:
    # The header will now correctly display the extracted destination (e.g., "Dubai")
    st.header(f"Upcoming Events in {st.session_state.destination}")
    
    if not st.session_state.events:
        st.warning("No events found. Try a different search!")
    else:
        for i, event in enumerate(st.session_state.events):
            event_key = f"{event['event_name']}_{event['start_date']}"
            
            with st.expander(f"**{event['event_name']}** - *Starting {event['start_date']}*"):
                
                if not st.session_state.enrich_history.get(event_key):
                    if st.button("Find Out More", key=event_key):
                        with st.spinner(f"Getting full details for {event['event_name']}..."):
                            # This call is now guaranteed to have the correct destination
                            enriched_data = enrich_single_event(
                                destination=st.session_state.destination,
                                event_data=event 
                            )
                            st.session_state.events[i] = enriched_data
                            st.session_state.enrich_history[event_key] = True
                        st.rerun()
                
                if st.session_state.enrich_history.get(event_key):
                    col1, col2 = st.columns(2)
                    with col1:
                        if event.get('image_url'):
                            st.image(event['image_url'], caption="Verified Event Image", use_container_width=True)
                        elif event.get('unsplash_image_url'):
                            st.image(event['unsplash_image_url'], caption="Image from Unsplash", use_container_width=True)
                        else:
                            st.write("No image available.")
                    with col2:
                        st.markdown(f"**Description:** {event.get('description', 'N/A')}")
                        st.info(f"""
                            **Venue:** {event.get('venue_name', 'N/A')}  
                            **Address:** {event.get('full_address', 'N/A')}  
                            **End Date:** {event.get('end_date', 'N/A')}
                        """)
                        if event.get('booking_link'):
                            st.link_button("➡️ Go to Booking/Info Page", event['booking_link'], use_container_width=True)
                        else:
                            st.warning("A verified booking link could not be found.")