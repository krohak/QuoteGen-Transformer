"""AWS Lambda function serving text_recognizer predictions."""
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

MODEL_NAME = "krohak/QuoteGen"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)


def handler(event, _context):
    """Provide main prediction API."""
    print("INFO loading text")
    event = _from_string(event)
    event = _from_string(event.get("body", event))
    text = event.get("text")
    if text is None or not isinstance(text, str):
        return {
            "statusCode": 400,
            "message": "text not found in event",
        }
    print("INFO starting inference")
    pred = generator(text)
    print("INFO inference complete")
    print("METRIC pred_length {}".format(len(pred)))
    print("INFO pred {}".format(pred))
    return {"pred": str(pred)}


def _from_string(event):
    if isinstance(event, str):
        return json.loads(event)
    else:
        return event
