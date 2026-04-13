from http.server import BaseHTTPRequestHandler
import json
import requests

API_URL = "https://api-inference.huggingface.co/models/t5-small"
headers = {
    "Authorization": "Bearerhf_blqeMAbuszfbnAfsDNPiDeHHSUgFEMeoLU"
}

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        dialogue = data.get("dialogue", "")

        payload = {
            "inputs": "summarize: " + dialogue,
            "parameters": {"max_length": 150, "min_length": 30}
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()

        summary = result[0]["summary_text"]

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        self.wfile.write(json.dumps({"summary": summary}).encode())
