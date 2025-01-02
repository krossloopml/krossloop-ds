from dotenv import load_dotenv
import os
import time
import google.generativeai as genai
import base64, json
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from google.oauth2 import service_account

from prompts import SYSTEM_PROMPT

# Load variables from the .env file
load_dotenv()

# Access the variables using os.getenv
# api_key = os.getenv("GEMINI_API_KEY")
vertexai_credentials = json.loads(os.getenv("VERTEX_AI_JSON_CREDENTIALS"))

credentials = service_account.Credentials.from_service_account_info(vertexai_credentials)

# Initialize Vertex AI
vertexai.init(project = "adept-shade-444819-a4", location = "us-central1", credentials = credentials)

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

def create_base64(file_path):
    """
    Converts a file to a Base64-encoded string.

    :param file_path: Path to the file to be encoded
    :return: Base64-encoded string
    """
    try:
        with open(file_path, "rb") as file:
            # Read the file's binary content
            file_content = file.read()
            # Encode the binary content to Base64
            base64_encoded = base64.b64encode(file_content).decode('utf-8')
            return base64_encoded
    except Exception as e:
        print(f"Error encoding file to Base64: {e}")
        return None

def process(data_source_pdf_path, model_name = "gemini-1.5-pro"):
    total_cost = 0
    start_time = time.time()

    try:
        model = GenerativeModel(model_name)

        # Convert the file to Base64
        base64_string = create_base64(data_source_pdf_path)
        if not base64_string:
            print("Failed to create Base64 string. Exiting.")
            return

        # Prepare the document
        document1 = Part.from_data(
            mime_type = "application/pdf",
            data = base64.b64decode(base64_string)
        )

        # Generation configuration
        # generation_config = {
        #     "max_output_tokens": 8192,
        #     "temperature": 0.5,
        #     "top_p": 0.95,
        #     "top_k": 40 if "flash" in model_name else 64,
        #     "seed": 50,
        # }
        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.0,
            "top_p": 0.0,
            "top_k": 1,
            "seed": 50,
        }

        # Safety settings
        safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
        ]

        print("Analysing data source document....")
        analysis_recommendation_response = model.generate_content(
            [document1, SYSTEM_PROMPT],
            generation_config = generation_config,
            safety_settings = safety_settings,
        )
        total_cost += calculate_cost(analysis_recommendation_response.usage_metadata.prompt_token_count, analysis_recommendation_response.usage_metadata.candidates_token_count, model_name)
        analysis_recommendation_response = analysis_recommendation_response.candidates[0].content.parts[0].text
        print("Analysed data source document.")

        # delete the files from local system
        os.remove(data_source_pdf_path)

        end_time = time.time()
        time_taken = end_time - start_time
        print(total_cost)

        return analysis_recommendation_response, time_taken, total_cost
    except Exception as e:
        return str(e)
