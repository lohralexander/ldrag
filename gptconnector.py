import logging
import time

from ldrag.config import Config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def gpt_request(user_message, system_message=None, previous_conversation=None, retrieved_information=None,
                model="gpt-4o-2024-11-20",
                sleep_time=0, seed=42, temperature=0):
    return gpt_request_with_history(user_message, system_message, previous_conversation, retrieved_information, model,
                                    sleep_time, seed, temperature)[0]


def gpt_request_with_history(user_message, system_message=None, previous_conversation=None, retrieved_information=None,
                             model="gpt-4o-2024-11-20",
                             sleep_time=0, seed=42, temperature=0):
    client = Config.chatgpt_client
    logger.debug(f"GPT Request started. Used model: {model}")
    logger.debug(f"User message: {user_message}")
    message = []
    if previous_conversation is not None:
        message = previous_conversation
        logger.debug(f"Previous conversation: {previous_conversation}")

    if system_message is not None:
        message.append({"role": "system", "content": system_message})
        logger.debug(f"System message: {system_message}")

    if retrieved_information is not None:
        logger.debug(f"Retrieved information: {retrieved_information}")
        message.append({"role": "user",
                        "content": f"{user_message}. Here is ontology retrieved information based on the topic: {retrieved_information}"})
    else:
        message.append({"role": "user", "content": user_message})
    try:
        response = client.chat.completions.create(
            temperature=temperature,
            messages=message,
            model=model,
            seed=seed
        )
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None, previous_conversation
    if response and response.choices:
        response_message = response.choices[0].message.content
        logger.debug(f"GPT Respond: {response_message}")
        message.append({"role": "assistant", "content": response_message})

    else:
        response_message = "Error: No valid response from API."
        logger.error(response_message)

    time.sleep(sleep_time)
    return response_message, message
