import requests
import json


def test_vllm_server():
    url = "http://localhost:8000/v1/chat/completions"

    payload = {
        "model": "qwen3_moe",  # /home/ml/weight/Qwen3-8B-W8A8
        "messages": [
            {"role": "user", "content": "你好，请用中文回答：什么是机器学习？"}
        ],
        "max_tokens": 27,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        print("=== 响应内容 ===")
        print(result['choices'][0]['message']['content'])
        print("\n=== 完整响应 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")


if __name__ == "__main__":
    test_vllm_server()