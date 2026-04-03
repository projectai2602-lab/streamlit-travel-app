import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ AI Travel Planner")
st.markdown("Plan your trip + Ask anything interactively 💬")
st.markdown("---")

# ---------------- API KEY ----------------
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("⚠️ API key not configured.")
    st.stop()

# ---------------- MODEL ----------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    api_key=api_key
)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "plan_generated" not in st.session_state:
    st.session_state.plan_generated = False

# ---------------- PROMPTS ----------------
prompt_country = ChatPromptTemplate.from_template(
    "Give a short summarized description of the country {country}. "
    "Mention culture, geography, and tourism highlights."
)

prompt_cities = ChatPromptTemplate.from_template(
    "Based on the following description:\n{text}\n\n"
    "List top cities in {country} with 2-3 attractions each."
)

prompt_tour = ChatPromptTemplate.from_template(
    "Create a {days}-day itinerary for {city}.\n\n"
    "Include:\n- Day-wise plan\n- Attractions\n- Travel tips"
)

chat_prompt = ChatPromptTemplate.from_template(
    "You are a smart travel assistant.\n\n"
    "Context:\nCountry: {country}\nCity: {city}\n\n"
    "Conversation:\n{history}\n\n"
    "User: {question}\n"
    "Assistant:"
)

# ---------------- CHAINS ----------------
planner_chain = (
    {
        "country": RunnablePassthrough(),
        "city": RunnablePassthrough(),
        "days": RunnablePassthrough()
    }
    | RunnablePassthrough.assign(
        text=(prompt_country | model | StrOutputParser())
    )
    | RunnablePassthrough.assign(
        cities=(prompt_cities | model | StrOutputParser())
    )
    | RunnablePassthrough.assign(
        tour_plan=(prompt_tour | model | StrOutputParser())
    )
)

chat_chain = (
    {
        "country": RunnablePassthrough(),
        "city": RunnablePassthrough(),
        "question": RunnablePassthrough(),
        "history": RunnablePassthrough()
    }
    | chat_prompt
    | model
    | StrOutputParser()
)

# ---------------- INPUT ----------------
country = st.text_input("🌍 Enter Country")
city = st.text_input("🏙 Enter City")
days = st.number_input("📅 Days", 1, 15, 3)

# ---------------- LOCATION FUNCTION ----------------
def get_location(city_name):
    try:
        geo = Nominatim(user_agent="travel_app", timeout=10)
        return geo.geocode(city_name)
    except (GeocoderTimedOut, GeocoderServiceError):
        return None

# ---------------- GENERATE PLAN ----------------
if st.button("Generate Travel Plan 🚀"):

    if not country or not city:
        st.warning("⚠️ Fill all fields")
        st.stop()

    with st.spinner("Planning... ✨"):
        try:
            result = planner_chain.invoke({
                "country": country,
                "city": city,
                "days": days
            })

            st.session_state.plan_generated = True

            # OUTPUT
            st.subheader("📘 Country")
            st.write(result["text"])

            st.subheader("🏙 Cities")
            st.write(result["cities"])

            st.subheader("🗺 Itinerary")
            st.write(result["tour_plan"])

            # MAP
            st.subheader("📍 Map")
            loc = get_location(city)

            if loc:
                df = pd.DataFrame({
                    "lat": [loc.latitude],
                    "lon": [loc.longitude]
                })
                st.map(df)
            else:
                st.warning("Map not available")

        except Exception as e:
            st.error("Error generating plan")
            st.exception(e)

# ---------------- CHAT SECTION ----------------
st.markdown("---")
st.subheader("💬 Travel Assistant Chat")

# Show messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Ask anything about your trip...")

if user_input:

    # Save user msg
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # Prepare history
    history = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):

            try:
                response = chat_chain.invoke({
                    "country": country if country else "Unknown",
                    "city": city if city else "Unknown",
                    "question": user_input,
                    "history": history
                })

                st.write(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:
                st.error("Chat error")
                st.exception(e)

# ---------------- CLEAR CHAT ----------------
if st.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.success("Chat cleared!")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + LangChain + Gemini")