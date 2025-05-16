import streamlit as st
from rag_engine import query_groq

st.set_page_config(page_title="PEC AnswerHub", layout="wide")
st.title("📘 Pakistan Engineering Council (PEC) AnswerHub")
st.markdown("Ask questions about PEC registration for contractors and engineering graduates.")

query = st.text_input("💬 Enter your question:", placeholder="e.g., What are the PEC registration requirements for contractors?")
if query and st.button("Submit"):
    with st.spinner("Searching PEC documents..."):
        try:
            answer = query_groq(query.strip())
            st.success(answer)
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# Feedback form
st.sidebar.header("Feedback")
with st.sidebar.form("feedback_form"):
    feedback = st.text_area("Share your feedback:")
    submitted = st.form_submit_button("Submit")
    if submitted and feedback:
        with open("feedback.txt", "a") as f:
            f.write(f"{feedback}\n")
        st.success("Thank you for your feedback!")
        