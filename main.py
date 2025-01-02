from dotenv import load_dotenv
import os
import time
import google.generativeai as genai

from prompts import SYSTEM_PROMPT

# Load variables from the .env file
load_dotenv()

# Access the variables using os.getenv
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key = api_key)

def calculate_cost(input_tokens, output_tokens, model):
    # Define pricing
    pricing_pro = {
        "input": {"under_128k": 1.25 / 1_000_000, "over_128k": 2.50 / 1_000_000},
        "output": {"under_128k": 5.00 / 1_000_000, "over_128k": 10.00 / 1_000_000}
    }

    pricing_flash = {
        "input": {"under_128k": 0.075 / 1_000_000, "over_128k": 0.3 / 1_000_000},
        "output": {"under_128k": 0.15 / 1_000_000, "over_128k": 0.6 / 1_000_000}
    }

    pricing = pricing_flash if "flash" in model else pricing_pro

    def get_cost(input_tokens, output_tokens):
        if input_tokens <= 128_000:
            input_cost = input_tokens * pricing["input"]["under_128k"]
            output_cost = output_tokens * pricing["output"]["under_128k"]
        else:
            input_cost = input_tokens * pricing["input"]["over_128k"]
            output_cost = output_tokens * pricing["output"]["over_128k"]
        return input_cost + output_cost

    return get_cost(input_tokens, output_tokens)


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    # print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Waits for the given files to be active.

    Some files uploaded to the Gemini API need to be processed before they can be
    used as prompt inputs. The status can be seen by querying the file's "state"
    field.

    This implementation uses a simple blocking polling loop. Production code
    should probably employ a more sophisticated approach.
    """
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")

def process(data_source_pdf_path, model_name = "gemini-1.5-pro"):
    total_cost = 0
    start_time = time.time()

    # Create the model
    generation_config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40 if "flash" in model_name else 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    try:

        model = genai.GenerativeModel(
            model_name = model_name,
            generation_config = generation_config,
            system_instruction = SYSTEM_PROMPT,
        )

        # You may need to update the file paths
        files = [
            upload_to_gemini(data_source_pdf_path, mime_type = "application/pdf"),
        ]

        # Some files have a processing delay. Wait for them to be ready.
        wait_for_files_active(files)

        chat_session = model.start_chat(
            history = [
                {
                    "role": "user",
                    "parts": [
                        files[0],
                    ],
                },
            ]
        )

        print("Analysing data source document....")
        analysis_recommendation_response = chat_session.send_message("Analyse and recommend")
        total_cost += calculate_cost(analysis_recommendation_response.usage_metadata.prompt_token_count, analysis_recommendation_response.usage_metadata.candidates_token_count, model_name)
        analysis_recommendation_response = analysis_recommendation_response.text
        print("Analysed data source document.")

        # delete the files from local system
        os.remove(data_source_pdf_path)

        end_time = time.time()
        time_taken = end_time - start_time
        print(total_cost)

        return analysis_recommendation_response, time_taken, total_cost
    except Exception as e:
        return str(e)
