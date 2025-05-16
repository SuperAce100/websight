from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Any, Optional
from pydantic import BaseModel
import logging
import traceback
import json

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client with OpenRouter endpoint
try:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment variables")
        raise ValueError("OPENROUTER_API_KEY not found")
        
    client = OpenAI(
        base_url="https://api.openrouter.ai/api/v1",
        api_key=api_key,
    )
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}\n{traceback.format_exc()}")
    raise

def analyze_image(
    image_base64: str,
    query: str,
    response_format: Optional[BaseModel] = None,
    model: str = "allenai/molmo-7b-d:free"
) -> str | BaseModel:
    """
    Analyze an image using the Molmo model through OpenRouter.
    
    Args:
        image_base64 (str): Base64 encoded image
        query (str): Query about the image
        response_format (BaseModel, optional): Pydantic model for structured response
        model (str): Model to use for analysis
    """
    try:
        logger.info(f"Starting image analysis with model: {model}")
        logger.debug(f"Query: {query}")
        logger.debug(f"Image base64 length: {len(image_base64)}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Analyze this browser screenshot and {query}. Please provide a structured list of potential actions or locations we can interact with."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        logger.debug(f"Request kwargs (excluding messages): {json.dumps({k: v for k, v in kwargs.items() if k != 'messages'})}")

        if response_format is not None:
            schema = response_format.model_json_schema()
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": schema
                }
            }
            logger.debug(f"Using response format schema: {json.dumps(schema)}")

            try:
                logger.info("Making API request with structured response format")
                response = client.chat.completions.create(**kwargs)
                logger.debug(f"Raw API response: {response.choices[0].message.content}")
                
                parsed_response = response_format.model_validate_json(response.choices[0].message.content)
                logger.info("Successfully parsed structured response")
                return parsed_response
                
            except Exception as e:
                error_msg = f"Failed to parse structured response: {str(e)}"
                logger.error(f"{error_msg}\nRaw response: {response.choices[0].message.content}\n{traceback.format_exc()}")
                raise ValueError(error_msg)

        logger.info("Making API request with text response format")
        response = client.chat.completions.create(**kwargs)
        logger.debug(f"Raw API response: {response.choices[0].message.content}")
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = f"Error during image analysis: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise 