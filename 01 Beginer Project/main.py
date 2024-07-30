import openai

# Đặt API Key của bạn
openai.api_key = 'YOUR_API_KEY'

# Tạo một yêu cầu đến API
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Hi, how can I help you today?",
    max_tokens=150
)

# In ra phản hồi từ API
print(response.choices[0].text.strip())
