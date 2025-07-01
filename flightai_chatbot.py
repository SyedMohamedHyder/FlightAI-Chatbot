# Imports

import os
import json
import base64
import logging
import gradio as gr
from PIL import Image
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
from IPython.display import Audio, display

# Initialization
logging.basicConfig(level=logging.INFO)
load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    logging.info(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    logging.error("OpenAI API Key not set")

MODEL = "gpt-4o-mini"
openai = OpenAI()

system_message = (
    "You are a helpful assistant for an airline called FlightAI. "
    "Always respond in a short, courteous sentence. "
    "Provide accurate information only. "
    "If you don’t know something, say so clearly. "
    "Before booking a ticket, strictly follow this order: "
    "1) Check if the destination is available, "
    "2) Then check the ticket price, "
    "3) Collect all neccessary details like name, destination and date of journey, "
    "4) Only then proceed with the booking. "
    "Always use the appropriate tools or APIs for each step before confirming a booking."
)

# Dummy funcs that mimic the ticket booking behaviour
# Replace these will real funcs (that call APIs or make DB transactions) to actually book a ticket

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}


def check_destination_availability(destination: str) -> dict:
    """
    Check if the given destination is available in our ticketing system.

    Args:
        destination (str): The name of the city.

    Returns:
        dict: {"available": bool}
    """
    logging.info(f"Checking availability for destination: {destination}")

    available = destination.lower() in ticket_prices
    return {"available": available}


def fetch_ticket_price(destination_city: str) -> dict:
    """
    Retrieve the ticket price for a given city.

    Args:
        destination_city (str): The name of the destination city.

    Returns:
        dict: {"price": str} or {"price": "Unknown"} if not found
    """
    logging.info(f"Retrieving price for destination: {destination_city}")

    city = destination_city.lower()
    price = ticket_prices.get(city, "Unknown")

    return {"price": price}


def book_ticket(name: str, destination_city: str, journey_date: str) -> dict:
    """
    Book a ticket to a destination city for a given user and date.

    Args:
        name (str): Name of the passenger.
        destination_city (str): Destination city.
        journey_date (str): Date of journey in YYYY-MM-DD format.

    Returns:
        dict: Booking confirmation with name, city, price, and date, or error.
    """
    logging.info(f"Booking ticket for {name} to {destination_city} on {journey_date}")

    city = destination_city.lower()

    if city not in ticket_prices:
        logging.error(f"City '{destination_city}' not found in ticket list.")
        return {"error": "Destination not found."}

    price_info = fetch_ticket_price(destination_city)

    return {
        "name": name,
        "destination_city": destination_city.title(),
        "journey_date": journey_date,
        "price": price_info["price"],
    }


destination_availability_tool = {
    "name": "check_destination_availability",
    "description": "Check if tickets are available for the given destination city before proceeding with any booking or pricing inquiry.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination": {
                "type": "string",
                "description": "The name of the destination city to check for availability.",
            }
        },
        "required": ["destination"],
        "additionalProperties": False,
    },
}

ticket_price_tool = {
    "name": "fetch_ticket_price",
    "description": (
        "Get the price of a return ticket to the specified destination city. "
        "Use this after confirming that the destination is available, especially when the customer asks for the ticket price."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city for which the customer wants the ticket price.",
            }
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}

ticket_booking_tool = {
    "name": "book_ticket",
    "description": (
        "Book a ticket for the customer to the specified destination city on the given journey date. "
        "Use only after availability and price have been checked."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Full name of the person booking the ticket.",
            },
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to.",
            },
            "journey_date": {
                "type": "string",
                "format": "date",
                "description": "The journey date in YYYY-MM-DD format.",
            },
        },
        "required": ["name", "destination_city", "journey_date"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": destination_availability_tool},
    {"type": "function", "function": ticket_price_tool},
    {"type": "function", "function": ticket_booking_tool},
]


def handle_tool_call(message):
    """
    Handles a single OpenAI tool call message and returns both the result
    and a formatted tool response dictionary.

    Args:
        message (object): An OpenAI message containing a tool call.

    Returns:
        tuple: (result_dict, response_dict)
    """
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    result = None

    logging.info(f"Tool call received: {function_name} with arguments: {arguments}")

    if function_name == "check_destination_availability":
        result = check_destination_availability(**arguments)

    elif function_name == "fetch_ticket_price":
        city = arguments.get("destination_city")
        price_info = fetch_ticket_price(city)
        result = {"destination_city": city, "price": price_info["price"]}

    elif function_name == "book_ticket":
        result = book_ticket(**arguments)

    else:
        logging.warning("Unrecognized tool function: %s", function_name)
        result = {"error": f"Unknown function '{function_name}'"}

    response = {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result),
    }

    return result, response


def artist(
    city: str, style: str = "vibrant pop-art", size: str = "1024x1024"
) -> Image.Image:
    """
    Generates a city-themed vacation image using DALL·E.

    Args:
        city (str): Name of the city to visualize.
        style (str): Artistic style for the image prompt.
        size (str): Image resolution (e.g., "1024x1024").

    Returns:
        Image.Image: A PIL Image object representing the generated image.

    Raises:
        ValueError: If city name is empty.
        RuntimeError: If image generation fails.
    """
    if not city.strip():
        raise ValueError("City name cannot be empty.")

    prompt = (
        f"An image representing a vacation in {city}, "
        f"showing iconic tourist attractions, cultural elements, and everything unique about {city}, "
        f"rendered in a {style} style."
    )

    logging.info("Generating image for city: %s with style: %s", city, style)

    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            n=1,
            response_format="b64_json",
        )

        image_base64 = response.data[0].b64_json
        image_data = base64.b64decode(image_base64)
        logging.info("Image generation successful for %s", city)

        return Image.open(BytesIO(image_data))

    except Exception as e:
        logging.error("Failed to generate image for city '%s': %s", city, str(e))
        raise RuntimeError(f"Image generation failed for city '{city}'") from e


def talker(
    message: str, output_filename: str = "output_audio.mp3", autoplay: bool = True
) -> None:
    """
    Converts a text message into speech using OpenAI TTS and plays the audio.

    Args:
        message (str): The text to convert to speech.
        output_filename (str): The filename to save the generated audio.
        autoplay (bool): Whether to autoplay the audio in the notebook.

    Raises:
        ValueError: If the message is empty.
        RuntimeError: If the audio generation fails.
    """
    if not message.strip():
        raise ValueError("Message cannot be empty.")

    logging.info("Generating speech for message: %s", message)

    try:
        response = openai.audio.speech.create(
            model="tts-1", voice="alloy", input=message
        )

        with open(output_filename, "wb") as f:
            f.write(response.content)

        logging.info("Audio written to: %s", output_filename)

        if autoplay:
            display(Audio(output_filename, autoplay=True))

    except Exception as e:
        logging.error("Failed to generate or play audio: %s", str(e))
        raise RuntimeError("Text-to-speech generation failed.") from e


def translate(message, language):
    """
    Translates the given text into the specified language using OpenAI Chat API.

    Args:
        message (str): The text to be translated.
        language (str): Target language for translation (e.g., 'French', 'Japanese').

    Returns:
        str: Translated text.

    Raises:
        ValueError: If input message or language is empty.
        RuntimeError: If translation fails due to API or other issues.
    """
    if not message.strip():
        raise ValueError("Input message cannot be empty.")
    if not language.strip():
        raise ValueError("Target language cannot be empty.")

    logging.info("Translating to %s: %s", language, message)

    messages = [
        {
            "role": "system",
            "content": f"You are a translation assistant. Translate everything the user says to {language}.",
        },
        {"role": "user", "content": message},
    ]

    try:
        response = openai.chat.completions.create(model=MODEL, messages=messages)
        translated = response.choices[0].message.content.strip()
        logging.info("Translation successful.")
        return translated

    except Exception as e:
        logging.error("Translation failed: %s", str(e))
        raise RuntimeError("Failed to translate message.") from e


def transcribe_audio(audio_path):
    """
    Transcribes an audio file using OpenAI's Whisper model.

    Args:
        audio_path (str): Path to the audio file (e.g., .mp3, .wav).
        model (str): OpenAI model for transcription (default: 'whisper-1').

    Returns:
        str: Transcribed text from the audio file.

    Raises:
        ValueError: If the path is invalid or the file does not exist.
        RuntimeError: If the transcription fails.
    """
    if not audio_path or not os.path.exists(audio_path):
        raise ValueError("Invalid or missing audio file path.")

    logging.info("Transcribing audio file: %s using model: whisper-1", audio_path)

    try:
        with open(audio_path, "rb") as f:
            response = openai.audio.transcriptions.create(model="whisper-1", file=f)
        transcript = response.text.strip()
        logging.info("Transcription successful.")
        return transcript

    except Exception as e:
        logging.error("Transcription failed: %s", str(e))
        raise RuntimeError("Failed to transcribe audio.") from e


def chat(
    history: list, language: str, translated_history: list, speaking_language: str
) -> tuple:
    """
    Handles a chat interaction including tool calls, image generation, translation, and TTS playback.

    Args:
        history (list): List of previous conversation messages.
        language (str): Target language for translation and TTS.

    Returns:
        tuple: (updated history list, generated image if any, translated response string)
    """
    messages = [{"role": "system", "content": system_message}] + history
    image = None

    try:
        # Initial assistant response
        response = openai.chat.completions.create(
            model=MODEL, messages=messages, tools=tools
        )
        choice = response.choices[0]

        # Handle tool calls if triggered
        if choice.finish_reason == "tool_calls":
            message = choice.message
            result, tool_response = handle_tool_call(message)

            # Append tool-related messages
            messages.append(message)
            messages.append(tool_response)
            logging.info("Tool call result: %s", result)

            # Generate image if a booking was completed
            if (
                message.tool_calls[0].function.name == "book_ticket"
                and "destination_city" in result
            ):
                image = artist(result["destination_city"])

            # Get final assistant response after tool execution
            response = openai.chat.completions.create(model=MODEL, messages=messages)
            choice = response.choices[0]

        reply = choice.message.content.strip()
        history.append({"role": "assistant", "content": reply})

        # Translate and speak the reply
        translated_reply = translate(reply, language)
        translated_history.append({"role": "assistant", "content": translated_reply})

        if speaking_language == "English":
            talker(reply)
        else:
            talker(translated_reply)

        return history, image, translated_history

    except Exception as e:
        logging.error("Chat processing failed: %s", str(e))
        raise RuntimeError("Failed to complete chat interaction.") from e


force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

with gr.Blocks(js=force_dark_mode) as ui:
    with gr.Row():
        gr.Markdown("### FlightAI Chat with Translation")

    with gr.Row():
        lang_dropdown = gr.Dropdown(
            choices=["Spanish", "French", "German", "Japanese", "Hindi"],
            value="Spanish",
            label="Translate To",
        )

        speak_dropdown = gr.Dropdown(
            choices=["English", "Selected Language"],
            value="English",
            label="Speak out in",
        )

    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages", label="Chat History")
        translated_chatbot = gr.Chatbot(
            height=500, type="messages", label="Translated Chat"
        )
        image_output = gr.Image(height=500)

    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
        audio_input = gr.Audio(
            sources="microphone", type="filepath", label="Or speak to the assistant"
        )

    with gr.Row():
        clear = gr.Button("Clear")

    def do_entry(message, history, audio, translated_history, language):
        if audio:
            message = transcribe_audio(audio)

        if message:
            history += [{"role": "user", "content": message}]
            translated_history += [
                {"role": "user", "content": translate(message, language)}
            ]
        return "", history, None, translated_history

    entry.submit(
        do_entry,
        inputs=[entry, chatbot, audio_input, translated_chatbot, lang_dropdown],
        outputs=[entry, chatbot, audio_input, translated_chatbot],
    ).then(
        chat,
        inputs=[chatbot, lang_dropdown, translated_chatbot, speak_dropdown],
        outputs=[chatbot, image_output, translated_chatbot],
    )

    audio_input.change(
        do_entry,
        inputs=[entry, chatbot, audio_input, translated_chatbot, lang_dropdown],
        outputs=[entry, chatbot, audio_input, translated_chatbot],
    ).then(
        chat,
        inputs=[chatbot, lang_dropdown, translated_chatbot, speak_dropdown],
        outputs=[chatbot, image_output, translated_chatbot],
    )

    clear.click(
        lambda: ["", [], None, [], None],
        inputs=None,
        outputs=[entry, chatbot, audio_input, translated_chatbot, image_output],
        queue=False,
    )

ui.launch(inbrowser=True)
