import streamlit as st
from streamlit_navigation_bar import st_navbar
from files import *
from ngram_cosine import *

st.set_page_config(initial_sidebar_state="collapsed")

pages = ["DOC/TXT", "Raw Text"]
styles = {   
    "nav": {
        "background-color": "rgb(58,70,100)",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(255,255,255)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    }
}

page = st_navbar(pages, styles=styles)
st.write(page)

if page == "DOC/TXT":
    st.title("Plagiarism Detection System")
    st.write("Upload an original document and a suspicious document to check for plagiarism.")

    original_file = st.file_uploader("Upload Original Document", type=["txt", "pdf"])
    suspicious_file = st.file_uploader("Upload Suspicious Document", type=["txt", "pdf"])

    n = 3

    if st.button("Compute Similarity"):

        if original_file is not None and suspicious_file is not None:
            if original_file.type == "application/pdf":
                original_text = extract_text_from_pdf(original_file)
            else:
                original_text = original_file.read().decode("utf-8")

            if suspicious_file.type == "application/pdf":
                suspicious_text = extract_text_from_pdf(suspicious_file)
            else:
                suspicious_text = suspicious_file.read().decode("utf-8")

            test_accuracy, similarity = train_test_model(original_text, suspicious_text, n)

            st.write(f"Test Accuracy: {test_accuracy:.2f}")
            st.write(f"Similarity: {similarity:.2f}")
            if similarity > 0.25:
                st.write("The documents are likely plagiarized.")
            else:
                st.write("The documents are not plagiarized.")

elif page == "Raw Text":
    st.title("N-Gram Cosine Similarity")

    st.write("Enter two texts to compute their cosine similarity based on n-gram vectors.")

    text1 = st.text_area("Text 1", "")
    text2 = st.text_area("Text 2", "")
    n = 2  # Set n-gram size to 3

    def calculate_ngram_cosine_similarity(text1, text2, n):
        text1 = preprocess_text(text1)
        text2 = preprocess_text(text2)

        ngrams_text1 = Counter(get_ngrams(text1, n))
        ngrams_text2 = Counter(get_ngrams(text2, n))

        dot_product = sum(ngrams_text1[key] * ngrams_text2.get(key, 0) for key in ngrams_text1)
        magnitude1 = math.sqrt(sum(ngrams_text1[key] ** 2 for key in ngrams_text1))
        magnitude2 = math.sqrt(sum(ngrams_text2[key] ** 2 for key in ngrams_text2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        return round(dot_product / (magnitude1 * magnitude2), 2)

    if st.button("Compute Similarity"):
        if text1 and text2:
            similarity_score = calculate_ngram_cosine_similarity(text1, text2, n)
            st.write(f"Cosine Similarity (based on {n}-gram): {similarity_score:.4f}")
            if similarity_score > 0.25:
                st.write("The documents are likely plagiarized.")

            else:
                st.write("The documents are not plagiarized.")

        else:
            st.write("Please enter text in both fields.")