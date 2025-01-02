import streamlit as st
import time
import uuid
import os
from datetime import datetime

from main import process


def save_uploaded_file(uploaded_file, folder = "uploaded_pdfs"):
    """
    Save uploaded file with a unique ID and return the filename
    """
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}_{uploaded_file.name}"
    file_path = os.path.join(folder, filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    return file_path

def process_pdfs_with_llm(data_source_pdf_path, model) -> str:
    """
    Process the PDF files with an LLM.
    Replace this with your actual LLM implementation.
    """
    try:
        response = process(data_source_pdf_path, model)
    except Exception as e:
        response = "Something went wrong.. :/\n"
        response += str(e)
    
    # Return mock response - Replace with actual LLM response
    return response

def main():
    st.title("Data Sources - Suggest Services")

    # Custom CSS to hide the Streamlit toolbar
    st.markdown("""
        <style>
            .stAppToolbar {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # File uploaders
    data_source_pdf = st.file_uploader(
        "Upload Data Source PDF",
        type = ['pdf'],
        accept_multiple_files = False,
        key = "data_source_pdf"
    )

    # Model Selector
    model_options = ["gemini-exp-1206", "gemini-1.5-pro", "gemini-2.0-flash-exp"]
    selected_model = st.selectbox("Select Model", options = model_options, index = 0)
    
    # GO button
    if data_source_pdf is not None:
        if st.button("GO"):
            with st.spinner("Processing..."):
                # Save files
                data_source_pdf_path = save_uploaded_file(data_source_pdf)
                
                st.info(f"Files saved as:\n- {os.path.basename(data_source_pdf_path)}")
                
                # Process files
                result = process_pdfs_with_llm(data_source_pdf_path, selected_model)

                # Check the type of result
                if isinstance(result, str):
                    st.error("Processing failed or unexpected response.")
                    st.error(result)  # Display the string directly
                elif isinstance(result, tuple) and len(result) == 3:
                    analysis_recommendation_response, time_taken, total_cost = result
                    st.success("Processing complete!")
                    st.write(f"Time taken: {time_taken:.6f} seconds")

                    with st.expander("Analysis - Recommendation Response", expanded=True):
                        st.text(analysis_recommendation_response)

                    with st.expander("Cost", expanded=False):
                        st.write(f"Total cost: ${total_cost:.6f}")
                else:
                    st.error("Unexpected response format received.")

if __name__ == "__main__":
    main()