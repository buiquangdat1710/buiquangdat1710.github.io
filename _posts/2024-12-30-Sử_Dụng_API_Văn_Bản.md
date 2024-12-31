---
title: "Sử Dụng API Văn Bản"
date: 2024-12-30 00:00:00  + 0800
categories: [AI Production]
tags: [api]
---
---


`API` (Application Programming Interface) là cầu nối cho phép các ứng dụng giao tiếp và sử dụng dịch vụ từ bên thứ ba. Với `API` văn bản, bạn có thể khai thác sức mạnh của các nền tảng như `OpenAI`, `Claude`, hay `Gemini` để xử lý ngôn ngữ tự nhiên, tạo nội dung, và phân tích dữ liệu. Blog này sẽ hướng dẫn bạn cách bắt đầu sử dụng các `API` này, từ việc thiết lập đến tích hợp chúng vào ứng dụng của mình.

## 1. Mô hình Client-Server.




- Mô hình `Client-Server` là kiến trúc cơ bản giúp các hệ thống giao tiếp với nhau qua mạng. Trong mô hình này:

    - `Client:` Là phía yêu cầu, thường là ứng dụng của bạn (ví dụ: một trang web hoặc ứng dụng di động). `Client` gửi yêu cầu (request) để lấy dữ liệu hoặc sử dụng dịch vụ từ `Server`.
    - `Server:` Là phía cung cấp, xử lý yêu cầu từ `Client` và trả về kết quả (response).
  
![anh](./image/203.png)

- Khi tích hợp `API` vào ứng dụng, `API` đóng vai trò trung gian giữa `Client` và `Server`. Cụ thể:

1. `Client` (ứng dụng của bạn) gửi yêu cầu tới `API`, ví dụ: yêu cầu tạo văn bản từ `OpenAI GPT`.
2. `API` nhận yêu cầu, xử lý bằng hệ thống của bên thứ ba (như `OpenAI`, `Claude` hoặc `Gemini`).
3. Server `API` trả về kết quả, thường dưới dạng dữ liệu `JSON`, cho `Client` để hiển thị hoặc sử dụng.

- Ví dụ đơn giản, bạn muốn tạo một đoạn văn bản bằng `OpenAI API`:

    - `Client` của bạn (ứng dụng) gửi yêu cầu gồm nội dung như: "Viết một đoạn văn giới thiệu về công nghệ AI."
    - `Server` của `OpenAI` nhận yêu cầu, xử lý, và trả về đoạn văn đã tạo.
    Client nhận kết quả và hiển thị cho người dùng.

## 2. Cách sử dụng API bên OpenAI.

### API liên quan tới văn bản

- Đầu tiên, bạn cần phải có một thẻ tín dụng hoặc thẻ visa để có thể sử dụng `API` bên `OpenAI`.  Sau khi nạp tiền vào bạn có thể vào phần `Billing` như ảnh dưới, như bạn đang thấy, tài khoản của tôi đang có 11.17 USD:

![anh](./image/204.png)

- Trước khi tạo `API`, chúng ta hãy thử chơi đùa một chút bên [OpenAI PlayGround](https://platform.openai.com/playground/chat?models=gpt-4o)

![anh](./image/205.png)


- Lưu ý rằng, bạn phải nạp tiền thì mới dùng được dịch vụ `PlayGround` này. Nhìn qua thì cái này không khác gì giao diện `Chat GPT` bạn vẫn hay sử dụng hàng ngày, tuy nhiên phần `PlayGround` này bạn sẽ được quyền chỉnh sửa các model khác nhau và cũng như các thông số dành riêng cho mô hình ngôn ngữ lớn. Chúng ta cùng điểm qua một vài thông số nào:
  - `Temperature`: Khi bạn chỉnh thông số này thấp, có nghĩa là bạn muốn mô hình chọn các token có xác suất cao, dùng trong các tác vụ cần chính xác cao như Hỏi đáp (QA) hay suy luận (reasoning). Còn khi bạn chỉnh thông số này cao, có nghĩa là bạn muốn mô hình lựa chọn token ngẫu nhiên hơn, tăng tính sáng tạo, dùng trong các tác vụ sáng tạo như viết văn, sáng tác thơ.
  - `Top P`: `Top P` quan tâm đến một nhóm nhỏ top các tokens có tổng xác suất ít nhất bằng p. `Top p` nhỏ có nghĩa là nhóm tokens được quan tâm nhỏ, tính chắc chắn cao. `Top p` lớn có nghĩa là nhóm tokens được quan tâm lớn, tính bay bổng cao.
  - `Frequency Penalty`: Phạt trên tỉ lệ token tiếp theo với số lần token này đã xuất hiện trên prompt + response. Giá trị này cao, token đã xuất hiện ít có khả năng lặp lại.
  - `Presence Penalty`: Kiểm soát việc lặp lại của những từ phía trước. Tập trung vào việc đa dạng hóa nội dung. Ví dụ từ trước dùng là "chăm chỉ", mô hình sẽ có xu hướng chọn từ sau là "cần cù". Khi giá trị này cao, kết quả sinh ra sẽ tập trung vào việc trên.

- Nếu bạn chưa hiểu thông số trên thì bản chất các mô hình ngôn ngữ lớn là các mô hình xác suất. Tức là nó sẽ chọn token có khả năng xuất hiện nhất để nó trả lời. Hãy nhìn vào ảnh dưới đây:

![anh](./image/206.png)

- Ví dụ như mô hình đang trả lời đến câu: "The boy went to the [?]". Chỗ [?] mô hình sẽ phải suy nghĩa là mình nên chọn token nào có xác suất lớn nhất. Như bạn thấy trên hình, mô hình có 5 phương án là các token: Cafe, Hospital, Playground, Park, School. Trên thực tế thì mô hình có cả triệu token để cân nhắn nên chọn cái nào. 
- Quay trở lại với các thông số. Thường thì người ta sẽ chỉnh thông số  `Top P` trước, lưu ý rằng thông số này chạy từ 0 đến 1, nếu tôi chỉnh `Top P = 0.5`  thì mô hình chỉ còn lại một phương án đó là Playground vì nó có xác suất là 0.4. Nếu như tôi chỉnh `Top P = 0.7` thì mô hình sẽ có hai phương án đó chính là Playground và School. Như vậy thông số `Top P` sẽ cho bạn biết kích thước không gian chọn token của mô hình.
- Sau khi chỉnh xong thông số `Top P` thì người ta sẽ để ý đến thông số `Temperature`. Để cho dễ nhớ, bạn hãy tưởng tượng như này. Nếu `Temperature` (nhiệt độ) càng cao thì càng nóng, mà càng nóng thì sẽ khiến các token "mềm" đi (nóng chảy). Các token mềm đi tức là các token sẽ có xác suất gần bằng nhau hơn. Ví dụ như `Top P = 0.7` thì như ta nói ở trên, mô hình sẽ chỉ có hai phương án là `Playground = 0.4` và `School = 0.3`. Nếu ta để `Temperature` cao thì xác suất của hai token này sẽ tiến gần với nhau, `Playground = 0.35` và `School = 0.35`.
- Oke chơi vậy là đủ rồi ! Giờ hãy quay lại với `API`. Giờ hãy vào phần `API Keys` và chọn `Create new secret key` rồi copy lại giá trị `API`:

![anh](./image/207.png)

> 📝 **Note:** API là giá trị vô cùng nhạy cảm, hãy chắc chắn rằng người khác không biết API của bạn. Vì bạn sẽ không thể kiểm soát được nếu giá trị ấy bị lộ ra ngoài. Thử tượng tưởng 1000 người sử dụng API của bạn, bạn sẽ hết tiền nhanh chóng. Thậm chí không thể đẩy API lên Github, Github sẽ cảnh cáo tài khoản của bạn.


- Hãy lưu trữ giá trị này ở nơi an toàn. Bạn có thể  `export` giá trị này như một biến môi trường bằng lệnh sau ở terminal. Nếu bạn dùng Window thì gõ lệnh sau:

```bash
setx OPENAI_API_KEY "your_api_key_here"
```

- Nếu bạn dùng macOS/Linux:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

- Một cách khác để lưu biến môi trường ảo đó là hãy tạo ra một file có tên là `.env` và ghi vào file ý (bạn phải sử dùng thư viện `dotenv` để load biến môi trường, có thể xem ví ở phần sau nữa):

```python
OPENAI_API_KEY="your_api_key_here"
```

- Để sử dụng `OpenAI API` trong python, hãy đảm bảo bạn đã tạo môi trường ảo và tải thư viện `openai`:

```bash
pip install openai
```

- Giờ hãy lập trình chương trình đầu tiên của bạn sử dụng `OpenAI API`:

```python
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message.content)
```

> Code calls to itself,  
> 
>Layers unfold endlessly—  
>
>Elegance in loops.

- Lưu ý là nếu như bạn tạo file `.env` và ghi giá trị `OPENAI_API_KEY` trong file đó thay vì sử dụng lệnh trên terminal thì bạn phải code như sau:

```python
from openai import OpenAI
from dotenv import load_dotenv
import os

# Nạp biến từ file .env
load_dotenv()

# Lấy giá trị API key từ biến môi trường
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key = OPENAI_API_KEY)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message.content)
```

- Khi gọi `API` về văn bản, tham số  đầu tiên bạn cần quan tâm chính là `model`. Mô hình bạn chọn có thể ảnh hưởng lớn đến đầu ra và tác động đến chi phí của mỗi `request`:
  - Mô hình lớn như `gpt-4o` cung cấp mức độ thông minh rất cao và hiệu suất mạnh mẽ, nhưng có chi phí cao hơn cho mỗi token.
  - Mô hình nhỏ như `gpt-40-mini` có trí thông minh không đạt mức của mô hình lớn hơn, nhưng nó nhanh hơn và ít tốn kém hơn cho mỗi token.
  - Mô hình lý luận như `o1` chậm hơn trong việc đưa ra kết quả và sử dụng nhiều token hơn để "suy nghĩ", nhưng có khả năng lý luận nâng cao, lập trình, và lập kế hoạch nhiều bước.

- Tham số tiếp theo bạn cần quan tâm đó là `messages`, tham số này là một `list` các từ điển. Bạn có thể tưởng tượng một từ điển chính là một lời tin nhắn. Trong từ điển này có hai giá trị khóa là `role` và `content`. 
- Cho tới phiên bản `openai` khi tôi đang viết blog này thì tham số `role` sẽ có ba giá trị là: `user`, `developer`, `assistant`. Giá trị `user` tức là bên phía người dùng gửi tin nhắn, tin nhắn sẽ được lưu vào trong biến `content`. Hãy xem ví dụ ở dưới đây:

```python
from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "Hello! How are you ?"
        }
    ]
)
print(completion.choices[0].message.content)
``` 

> Hello! I’m just a computer program, so I don’t have feelings, but I’m here to help you. How can I assist you today?


- Tham số `role` khi có giá trị là `assistant` tức là tin nhắn phía bên chatbot. Tại sao lại có giá trị này ? Chẳng phải mình chỉ cần gửi tin nhắn bên người dùng thôi sao ? Hãy xem ví dụ dưới đây, khi chỉ có mọt tin nhắn bên người dùng:

```python
from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "Say my name 3 times"
        }
    ]
)
print(completion.choices[0].message.content)
``` 
> I'm here to help, but I need to respect your privacy, so I don't have access to your name. If you'd like, feel free to share it, and I can incorporate it into our interaction.

- Giờ hãy thêm hai tin nhắn giữa người dùng và chatbot vào phía trước câu đề nghị: "Say my name 3 times" như sau:

```python
from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        { "role": "user",
            "content": "My name is Dat"
        },
        {
            "role": "assistant",
            "content": "Hello Dat"
        },
        {
            "role": "user",
            "content": "Say my name 3 times"
        }
    ]
)
print(completion.choices[0].message.content)
```
> Dat, Dat, Dat

- Như bạn thấy, giá trị `assistant` sẽ là tin nhắn bên phía chatbot, mục đích chính là giúp chatbot có được các thông tin từ cuộc hội thoại trong quá khứ.
- Tham số  `role` còn có thể nhận giá trị `developer`. Tưởng tượng đơn giản là bạn muốn chatbot trả lời như thế nào, theo cách nào, giống như việc bạn là lập trình viên, code ra con chatbot vậy. Nếu bạn muốn chatbot chỉ trả lời bằng tiếng hàn quốc, bạn có thể code như sau:

```python
from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        { "role": "developer",
            "content": "Only answer in Korean"
        },
        {
            "role": "user",
            "content": "Tell me a joke"
        }
    ]
)
print(completion.choices[0].message.content)
```
> 왜 스파게티는 컴퓨터를 좋아할까요?  
> 
> 왜냐하면 그들은 이메일을 넣을 수 있어서요!

- Giá trị này rất có ích khi bạn biết các kĩ thuật về  `prompt engineering`. Bạn có thể xem blog `prompt engineering` trên trang này.
- Bạn có thể xem thêm các ví dụ về prompt trong các tác vụ khác nhau [tại đây](https://platform.openai.com/docs/examples). Một ví dụ khá là thú vị là giả sử bạn muốn chatbot trả lơi chỉ được dùng emoji, bạn có thể code như sau:

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "system",
      "content": "You will be provided with text, and your task is to translate it into emojis. Do not use any regular text. Do your best with emojis only."
    },
    {
      "role": "user",
      "content": "Artificial intelligence is a technology with great promise."
    }
  ],
  temperature=0.8,
  max_tokens=256,
  top_p=1
)
print(response.choices[0].message.content)
```

> 🤖✨📚🔮

- Để ý rằng bạn cũng có thể chỉnh được các thông số như `temperature`, `max_tokens`, `top_p` như code trên. Ngoài ra, để code được gọn hơn cũng như bảo trì dễ hơn thì người ta hay tạo ra một `class` riêng cho phần `OpenAI` như sau:

```python
from openai import OpenAI
class OpenAIClient():
    def __init__(self):
        self.client = OpenAI()

    def chat(self, messages):
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages)
        return completion.choices[0].message.content
openAIclient = OpenAIClient()
openAIclient.chat([
    {
        "role": "user",
        "content": "Hello ! Who are you ?"
    },

])

import json
def server():
    print("Start chatting with the OpenAI API (type 'quit' to stop):")
    context = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Add user input to the context
        context.append({"role": "user", "content": user_input})

        # Summarize context

        # Get response from OpenAI
        openai_response = openAIclient.chat(context)

        # Add OpenAI response to the context
        context.append({"role": "assistant", "content": openai_response})

        print(f"AI: {json.dumps(openai_response)}")
server()
```

### API liên quan tới văn bản (kèm hình ảnh)

- Có rất nhiều model `OpenAI` có khả năng nhận diện hình ảnh, nghĩa là bạn có thể gửi request bao gồm cả hình ảnh. Hãy nhìn hình ảnh dưới đây:

![anh](./image/208.png)

- Giờ chúng ta hãy thử hỏi model về hình ảnh này bằng đoạn code như sau:

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)
```
> The image depicts a scenic outdoor landscape featuring a wooden pathway or boardwalk that winds through a lush green field with tall grass. In the background, there are trees and bushes, along with a clear blue sky interspersed with clouds. The setting suggests a serene natural environment, likely in a park or nature reserve.

- Nếu bạn có một bức ảnh đuọc lưu cục bộ (local), bạn có thể truyền vào model ảnh ý dưới dạng mã hóa 64, dưới đây là ví dụ:

```python
import base64
from openai import OpenAI

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "path_to_your_image.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)
```

- Bạn cũng có thể gửi request gồm nhiều ảnh tới model như sau:

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are in these images? Is there any difference between them?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)
print(response.choices[0].message.content)
```
### API liên quan tới sinh ảnh

- Các model liên quan tới xử lý ảnh bên `OPENAI` không phải là các model tốt nhất nhưng chúng ta cũng có thể xem qua. `API` liên quan tới ảnh của `OpenAI` cung cấp cho chúng ta 3 tác vụ chính:
1. Sinh ảnh dựa vào `prompt` (Mô hình `DALL-E 3` và `DALL-E 2`) 
2. Tạo ra một ảnh mới được edit bằng cách thay thế một vài vùng trong ảnh dựa vào `prompt` (Mô hình `DALL-E 2`)
3. Tạo ra các phiên bản khác nhau của ảnh gốc (`DALL-E 2`)

- Oke, giờ chúng ta hãy thử tác vụ sinh ảnh. Mặc định, hình ảnh sẽ được sinh ở mức tiêu chuẩn (bình thường), nhưng khi bạn sử dụng model `DALL-E 3`, bạn có thể đặt `quality: "hd"` cho chất lượng ảnh tốt hơn. Tất nhiên là ảnh càng chất lượng thì càng phải chờ đợi thời gian lâu để sinh:

```python
from openai import OpenAI
client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="a white siamese cat",
    size="1024x1024",
    quality="standard",
    n=1, # số lượng ảnh 
)

print(response.data[0].url)
```

![anh](./image/209.png)

- Giờ hãy thử sinh ảnh nhưng đặt `quality: "hd"`:

```python
from openai import OpenAI
client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="a white siamese cat",
    size="1024x1024",
    quality=hd",
    n=1, # số lượng ảnh 
)

print(response.data[0].url)
```

![anh](./image/210.png)


- Tác vụ tiếp theo chúng ta có thể làm là edit ảnh. Trước tiên chúng ta cần phải có hai ảnh, ảnh thứ nhất là ảnh gốc, ảnh thứ hai là ảnh gốc nhưng đã bị khoanh vùng (mask). Sau đó bạn sẽ nhập `prompt` yêu cầu model sinh ra vùng ảnh ở phần khoanh vùng theo như bạn chỉ định (bạn có thể tải hai ảnh ở code dưới [tại đây](https://imgur.com/a/dall-e-edits-infill-example-images-oGbgYlm)):

```python
from openai import OpenAI
client = OpenAI()

response = client.images.edit(
    model="dall-e-2",
    image=open("sunlit_lounge.png", "rb"),
    mask=open("mask.png", "rb"),
    prompt="A sunlit indoor lounge area with a pool containing a flamingo",
    n=1,
    size="1024x1024",
)

print(response.data[0].url)
```

![anh](./image/211.png)

- Tác vụ thứ ba đó chính là sinh ra các phiên bản khác nhau của ảnh gốc. Hãy cùng xem qua đoạn code dưới đây:

```python
from openai import OpenAI
client = OpenAI()

response = client.images.create_variation(
    model="dall-e-2",
    image=open("corgi_and_cat_paw.png", "rb"),
    n=1,
    size="1024x1024"
)

print(response.data[0].url)
```
![anh](./image/212.png)


- Còn rất nhiều thứ hay ho nữa như sinh âm thanh, text to speech, Embediding,... bạn có thể xem trên [OpenAI](https://platform.openai.com/docs/guides/text-to-speech). Vì blog này tập trung vào gọi API liên quan đến văn bản nên tôi sẽ dừng ở đây.

## 3. Cách sử dụng API bên TogetherAI

- `Together AI` giống như một "sân chơi chung" dành cho trí tuệ nhân tạo, nơi mọi người có thể sử dụng, chia sẻ và phát triển các công nghệ AI một cách dễ dàng và không bị phụ thuộc vào các công ty lớn. Nó cung cấp các công cụ mạnh mẽ và miễn phí để giúp ai cũng có thể tạo ra hoặc ứng dụng AI, giống như việc mở một kho tài nguyên cho mọi người cùng học và sáng tạo.
- Gọi `API` bên `Together AI` khá giống bên `OpenAI`. Đầu tiên bạn cần phải tải thư viện `together`:

```bash
pip install together 
```

- Tiếp theo chúng ta làm y hệt như bên OpenAI để tạo ra biến môi trường ảo, nhưng lần này, chúng ta sẽ đặt tên biến là `TOGETHER_API_KEY`. Như đã nói, bạn có hai cách, cách thứ nhất là gõ lệnh trên terminal, giống hệt như bên `OPENAI`. Cách thứ hai là tạo ra file `.env` và gán  `TOGETHER_API_KEY = <your api>`. Đoạn code dưới đây là cách gọi `API` trong `TogetherAI`:

```python
from together import Together

client = Together()

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ],
    max_tokens=512,
    temperature=0.7,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1,
    stop=["<|eot_id|>","<|eom_id|>"],
    stream=True
)
for token in response:
    if hasattr(token, 'choices'):
        print(token.choices[0].delta.content, end='', flush=True)
```

> Functions call within
> 
> Echoes of repeating code

## 4. Cách sử dụng API bên Gemini

- `Gemini` là một bộ công cụ AI tiên tiến được phát triển bởi `Google`, chuyên cung cấp các mô hình học sâu phục vụ nhiều ứng dụng khác nhau, bao gồm xử lý ngôn ngữ tự nhiên, nhận diện hình ảnh, và phân tích dữ liệu. Với khả năng xử lý thông tin mạnh mẽ và chính xác, `Gemini` hỗ trợ các nhà phát triển tích hợp các tính năng AI vào ứng dụng của mình thông qua `API` dễ sử dụng, giúp tăng cường hiệu suất và trải nghiệm người dùng. `Gemini` được thiết kế để tối ưu hóa khả năng học máy và cải thiện khả năng tự động hóa trong các ngành công nghiệp khác nhau.
- Cách setup `API` tôi sẽ không nói nữa, tương tự như các phần ở trên, sử dụng biến `GEMINI_API_KEY`. Đầu tiên, bạn sẽ cần phải cài đặt thư viện `google-generativeai`, hãy đảm bảo môi trường ảo của bạn có phiên bản `python >= 3.9`, nếu không thì sẽ xảy ra lỗi khi chạy code:

```bash
pip install google-generativeai
```

- Hãy xem đoạn code dưới đây:

```python
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "Xin chào\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Chào bạn! Rất vui được trò chuyện với bạn. Bạn có khỏe không? Hôm nay bạn muốn nói về điều gì?\n",
      ],
    },
  ]
)

response = chat_session.send_message("Who are you ?")

print(response.text)
```

> I am a large language model, trained by Google.

## 5. Tổng kết.

- Còn rất nhiều các mô hình ngôn ngữ khác ngoài kia như `Claude`, `DeepSeek`,... cũng như rất nhiều mô hình mở trên `Hugging Face` mà bạn có thể gọi `API`. Hy vọng blog này có ích với bạn trên con đường trở thành AI engineer.