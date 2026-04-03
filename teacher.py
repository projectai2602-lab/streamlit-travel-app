import streamlit as st
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 AI Learning Assistant")

# ---------------- API KEY ----------------
# Safe handling (prevents crash if missing)
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("API key not found. Please add it in Streamlit Cloud Secrets.")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]

# ---------------- MODEL ----------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    api_key=api_key
)

# ---------------- PROMPTS ----------------
prompt1 = ChatPromptTemplate.from_template(
    "Write a short explanation about the topic {topic} for a {level} level student in a {style} style."
)

prompt2 = ChatPromptTemplate.from_template(
    "Summarize the following text in a short and simple way:\n{text}"
)

prompt3 = ChatPromptTemplate.from_template(
    "Create {number} quiz questions from the following summary.\n{summary}\n\nFormat:\nQuestion\nAnswer"
)

# ---------------- UI ----------------
topic = st.text_input("Enter topic")
level = st.selectbox("Select level", ["Beginner", "Intermediate", "Advanced"])
style = st.selectbox("Explanation style", ["Simple", "Detailed", "Story"])
number = st.number_input("Number of questions", min_value=1, max_value=10, value=3)

# ---------------- BUTTON ----------------
if st.button("Generate 🚀"):

    if topic:
        with st.spinner("Generating..."):

            try:
                # Step 1
                response1 = (prompt1 | model).invoke({
                    "topic": topic,
                    "level": level,
                    "style": style
                })

                # Step 2
                response2 = (prompt2 | model).invoke({
                    "text": response1.content
                })

                # Step 3
                response3 = (prompt3 | model).invoke({
                    "summary": response2.content,
                    "number": number
                })

                # ---------------- OUTPUT ----------------
                st.subheader("📖 Explanation")
                st.write(response1.content)

                st.subheader("📝 Summary")
                st.write(response2.content)

                st.subheader("❓ Quiz")
                st.write(response3.content)

            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.warning("Please enter a topic")