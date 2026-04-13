import json
import requests

API_URL = "https://api-inference.huggingface.co/models/t5-small"
headers = {
    "Authorization": "Bearer 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"
}

def handler(request):
    try:
        body = request.get_json()
        dialogue = body.get("dialogue", "")

        payload = {
            "inputs": "summarize: " + dialogue,
            "parameters": {"max_length": 150, "min_length": 30}
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()

        summary = result[0]["summary_text"]

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"summary": summary})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
