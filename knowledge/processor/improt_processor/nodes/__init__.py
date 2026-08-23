import requests

url = "https://chatapi.wangjunxi.com.cn/v1/chat/completions"
headers = {
    "Authorization": "Bearer sk-d325a002-7b77-4caf-b282-800cec38b2dd",
    "Content-Type": "application/json"
}
data = {
    "model": "Bouquet",
    "messages": [{"role":"user","content":"你能回我我吃"}]
}
res = requests.post(url, headers=headers, json=data)
print(res.json())