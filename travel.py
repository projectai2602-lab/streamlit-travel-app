import streamlit as st
import os
import pandas as pd
from geopy.geocoders import Nominatim

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Travel Planner",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Travel Planner")
st.markdown("Plan your trip with AI 🌍")
st.markdown("---")

# ---------------- API KEY (SECURE) ----------------
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("API key not configured. Please contact admin.")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]

# ---------------- MODEL ----------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    api_key=api_key
)

# ---------------- PROMPTS ----------------
prompt_country = ChatPromptTemplate.from_template(
    "Give a short summarized description of the country {country}. "
    "Mention its culture, geography, and why tourists visit it."
)

prompt_cities = ChatPromptTemplate.from_template(
    "Based on the following description:\n{text}\n\n"
    "List the main cities in {country} that tourists should visit. "
    "Mention 2 or 3 famous attractions for each city."
)

prompt_tour = ChatPromptTemplate.from_template(
    "Create a travel itinerary for visiting {city} in {days} days.\n\n"
    "Include:\n"
    "- Day-by-day travel plan\n"
    "- Popular tourist places\n"
    "- Best order to visit these places\n"
    "- Short travel tips"
)

# ---------------- CHAIN ----------------
chain = (
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

# ---------------- USER INPUT ----------------
country = st.text_input("🌍 Enter Country")
city = st.text_input("🏙 Enter City to Visit")
days = st.number_input("📅 Number of Days", min_value=1, max_value=15, value=3)

# ---------------- BUTTON ----------------
if st.button("Generate Travel Plan 🚀"):

    if country and city:

        with st.spinner("Planning your trip... ✨"):

            try:
                result = chain.invoke({
                    "country": country,
                    "city": city,
                    "days": days
                })

                # ---------------- OUTPUT ----------------
                st.subheader("📘 Country Summary")
                st.write(result["text"])

                st.subheader("🏙 Major Cities & Attractions")
                st.write(result["cities"])

                st.subheader("🗺 Travel Itinerary")
                st.write(result["tour_plan"])

                # ---------------- MAP ----------------
                st.subheader("📍 Location Map")

                geolocator = Nominatim(user_agent="travel_app")
                location = geolocator.geocode(city)

                if location:
                    map_data = pd.DataFrame({
                        "lat": [location.latitude],
                        "lon": [location.longitude]
                    })
                    st.map(map_data)
                else:
                    st.warning("Could not find location on map")

            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.warning("Please fill all fields!")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + LangChain")