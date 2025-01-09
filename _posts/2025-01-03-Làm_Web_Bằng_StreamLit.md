---
title: "Làm Web Bằng StreamLit"
date: 2025-01-03 00:00:00  + 0800
categories: [AI Engineer]
tags: [streamlit]
---
---


`Streamlit` là một thư viện Python giúp tạo ứng dụng web tương tác dễ dàng, thường được dùng để trình bày dữ liệu, mô hình học máy hoặc công cụ phân tích mà không cần kiến thức lập trình web.


## 1. Cài đặt
- Đầu tiên, hãy tạo môi trường ảo trong python. Nếu bạn chưa biết cách tạo môi trường ảo, hãy xem lại các blog cũ. Tiếp theo, ta chạy lệnh sau để tải `streamlit`:

```bash
pip install streamlit
```

- Tiếp theo, bạn có thể chạy lệnh dưới để kiểm tra xem `streamlit` đã được tải chưa, đồng thời cũng xem đươc bản demo mà `streamlit` đã làm sẵn cho mình:

```bash
streamlit hello
```

- Để chạy một file chứa `streamlit`, chúng ta dùng lệnh dưới đây:

```bash
streamlit run your_file_name.py
```

- Nhưng trước khi chạy thì chúng ta cần code một thứ gì đó đã, hãy đến với phần tiếp theo.

## 2. Các hàm cơ bản trong Streamlit

- Trước tiên, để khai báo thư viện `streamlit` thì người ta thường code như sau:

```python
import streamlit as st
```

- Có một số cách để hiển thị dữ liệu (bảng, mảng, data frames) trong ứng dụng `Streamlit`. Dưới đây, bạn sẽ được giới thiệu về  `magic` và `st.write()`, có thể được sử dụng để viết bất cứ thứ gì từ văn bản đến bảng.

### Magic

- Bạn có thể code mà không cần dùng bất cứ hàm nào trong `streamlit`, và cái đó gọi là `magic commands`, có nghĩa là bạn sẽ không phải sử dụng lệnh `st.write()` (lệnh này là gì, tôi sẽ nói sau). Hãy xem đoạn code dưới đây:

```python
import streamlit as st
import pandas as pd
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df
```
- Bất cứ khi nào `Streamlit` nhìn thấy một biến hoặc một giá trị bằng chữ trên dòng riêng của nó, nó sẽ tự động ghi giá trị đó vào ứng dụng của bạn bằng cách sử dụng `st.write()`.

### Write

- Cùng với `magic commands` thì `st.write()` là hàm thông dụng nhất trong `StreamLit`. Bạn có thể đẩy gần như mọi thứ vào hàm `st.write()` như văn bản, các kiểu dữ liệu,... Ví dụ hãy xem đoạn code sau:

```python
import streamlit as st
import pandas as pd

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))
```

- Một điều hay ho nữa là bạn có thể trang trí bảng dữ liệu. Hãy xem đoạn code đơn giản dưới đây:

```python
import streamlit as st
import numpy as np

dataframe = np.random.randn(10, 20)
st.dataframe(dataframe)
```

- Giờ hãy thử sử dụng phương thức `Styler` trong `Pandas` để tô màu một số trong bảng:

```python
import streamlit as st
import numpy as np
import pandas as pd

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))
```
- Nếu như bạn muốn một bảng dữ liệu tĩnh, bạn có thể sử dụng `st.table()`:
  
```python
import streamlit as st
import numpy as np
import pandas as pd

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))
st.table(dataframe)
```

### Vẽ biểu đồ và bản đồ

- Bạn có thể dễ dàng vẽ biểu đồ bằng cách sử dụng `st.line_chart()`. Dưới đây là một ví dụ đơn giản:

```python
import streamlit as st
import numpy as np
import pandas as pd

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)
```

- Hàm `st.map()` trong Streamlit được sử dụng để hiển thị dữ liệu địa lý trên bản đồ. Nó nhận một DataFrame chứa các cột `latitude` (vĩ độ) và `longitude` (kinh độ) và vẽ các điểm đó trên bản đồ:

```python
import streamlit as st
import pandas as pd

# Dữ liệu mẫu với cột latitude và longitude
data = pd.DataFrame({
    'latitude': [10.762622, 21.028511, 16.054407],
    'longitude': [106.660172, 105.854444, 108.202167]
})

# Hiển thị bản đồ
st.map(data)

```

### Widgets

- Khi bạn đã đưa dữ liệu hoặc mô hình vào trạng thái muốn khám phá, bạn có thể thêm vào các tiện ích như `st.slider()`, `st.button()` hoặc `st.selectbox()`.
- Hàm `st.slider()` trong `Streamlit` tạo một thanh trượt (slider) để người dùng chọn giá trị trong một phạm vi nhất định. Đây là một công cụ tương tác hữu ích trong các ứng dụng web để nhận đầu vào từ người dùng. Ví dụ:

```python
import streamlit as st

# Thanh trượt số
number = st.slider('Chọn một số', min_value=0, max_value=100, value=50)

# Thanh trượt giá trị thập phân
decimal = st.slider('Chọn một số thập phân', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# Thanh trượt ngày
import datetime
date = st.slider(
    'Chọn một ngày',
    min_value=datetime.date(2022, 1, 1),
    max_value=datetime.date(2025, 1, 1),
    value=datetime.date(2023, 1, 1)
)

st.write("Số đã chọn:", number)
st.write("Số thập phân đã chọn:", decimal)
st.write("Ngày đã chọn:", date)

```

- Hàm `st.button()` trong `Streamlit` tạo một nút bấm trên giao diện. Khi người dùng nhấn nút, hàm sẽ trả về giá trị `True`, giúp bạn kích hoạt một hành động cụ thể:

```python
import streamlit as st

# Hàm callback để thực hiện hành động khi nút được nhấn
def chao_nguoi_dung(ten):
    st.write(f"Xin chào, {ten}!")

# Hiển thị nút bấm với tất cả các tham số
if st.button(
    label="Nhấn vào đây",
    key="nut_chao",
    help="Nhấn để gửi lời chào",
    on_click=chao_nguoi_dung,
    args=("Streamlit User",)  # Tham số truyền vào hàm callback
):
    st.write("Nút đã được nhấn!")
else:
    st.write("Nút chưa được nhấn. Hãy thử nhấn!")

```
- Giải thích tham số:
    - `label`: Tên nút hiển thị là "Nhấn vào đây".
    - `key`: Khóa duy nhất của nút là "nut_chao".
    - `help`: Khi di chuột qua nút, xuất hiện gợi ý "Nhấn để gửi lời chào".
    - `on_click`: Gọi hàm chao_nguoi_dung khi nút được nhấn.
    - `args`: Truyền tham số "Streamlit User" vào hàm chao_nguoi_dung.


- Hàm `st.selectbox()` trong `Streamlit` hiển thị một hộp chọn (dropdown) cho phép người dùng chọn một giá trị từ danh sách. Đây là cách tương tác đơn giản để nhận đầu vào từ người dùng:

```python
import streamlit as st

# Danh sách các lựa chọn
options = ["Python", "Java", "C++", "JavaScript"]

# Hộp chọn
choice = st.selectbox("Chọn ngôn ngữ lập trình yêu thích:", options)

# Hiển thị kết quả lựa chọn
st.write("Bạn đã chọn:", choice)

```

- 
Hàm `st.checkbox()` trong `Streamlit` tạo một hộp kiểm (checkbox) cho phép người dùng bật hoặc tắt một trạng thái (True/False). Đây là một công cụ hữu ích để tạo các lựa chọn có hoặc không trong ứng dụng:

```python
import streamlit as st

# Hộp kiểm
agree = st.checkbox("Tôi đồng ý với các điều khoản")

# Hiển thị trạng thái
if agree:
    st.write("Cảm ơn bạn đã đồng ý!")
else:
    st.write("Hãy xem lại các điều khoản trước khi tiếp tục.")

```

### Layout

- `Streamlit` cho phép bạn tổ chức các `widgets` bên trái giao diện với hàm `st.sidebar`. Mỗi phần tử được truyền vào hàm `st.sidebar` sẽ được hiện thị bên trái, điều này giúp cho người dùng tập trung vào nội dung ở màn hình chính mà vẫn có thể tương tác với sidebar. Ví dụ, nếu bạn muốn có `selectbox` và `slider`, bạn có thể sử dụng hàm `st.sidebar.slider` và hàm `st.sidebar.selectbox`:

```python
import streamlit as st

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)
```

- Ngoài `sidebar`, `Streamlit` còn cung cấp một số cách khác để kiểm soát bố cục ứng dụng của bạn. `st.columns` cho phép bạn đặt các widgets cạnh nhau và `st.expander` cho phép bạn tiết kiệm không gian bằng cách ẩn nội dung lớn:

```python
import streamlit as st

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")
```

### Hiển thị thanh progress

- Bạn có thể sử dụng `st.progress()` để hiện thị thanh `progress` trong thời gian thực. Dưới đây là code ví dụ:

```python
import streamlit as st
import time

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'
```


### Pages

- Khi các ứng dụng trở nên lớn, việc tổ chức chúng thành nhiều trang sẽ rất hữu ích. Điều này giúp ứng dụng dễ quản lý hơn đối với nhà phát triển và dễ sử dụng hơn đối với người dùng. `Streamlit` cung cấp một cách dễ dàng để tạo các ứng dụng nhiều trang. Tính năng này được thiết kế để việc xây dựng một ứng dụng nhiều trang dễ dàng như xây dựng một ứng dụng một trang duy nhất!

- Chỉ cần thêm nhiều trang vào ứng dụng hiện có như sau:
  1. Trong thư mục chứa tập lệnh chính của bạn, tạo một thư mục mới có tên là `pages`.
  2. Giả sử tập lệnh chính của bạn được đặt tên là `main_page.py`.
  3. Thêm các tệp `.py` mới vào thư mục `pages` để thêm nhiều trang vào ứng dụng của bạn.
- Chạy lệnh `streamlit run main_page.py` như thông thường. Tập lệnh `main_page.py` sẽ tương ứng với trang chính của ứng dụng của bạn. Bạn sẽ thấy các tập lệnh khác từ thư mục pages xuất hiện trong trình chọn trang ở thanh bên (sidebar). Ví dụ:

```python
# main_page.py
import streamlit as st

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")
```

```python
# pages/page_2.py
import streamlit as st

st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# Page 2 ❄️")
```

```python
# pages/page_3.py
import streamlit as st

st.markdown("# Page 3 🎉")
st.sidebar.markdown("# Page 3 🎉")
```


##  3. Làm một App LLM chat đơn giản bằng Streamlit

- Trong hướng dẫn này, chúng ta sẽ bắt đầu bằng cách tìm hiểu các thành phần chat của `Streamlit`, `st.chat_message` và `st.chat_input`. Sau đó, chúng ta sẽ tiến hành xây dựng ba ứng dụng khác nhau, mỗi ứng dụng thể hiện mức độ phức tạp và chức năng tăng dần:
  - Đầu tiên, chúng ta sẽ xây dựng một bot phản hồi y hệt câu hỏi của bạn, giúp bạn làm quen với các thành phần chat và cách chúng hoạt động. Chúng ta cũng sẽ giới thiệu `session state` và cách nó có thể được sử dụng để lưu trữ lịch sử trò chuyện. 
  - Tiếp theo, bạn sẽ học cách xây dựng một giao diện chatbot đơn giản với tính năng `streaming`.
  -Cuối cùng, chúng ta sẽ Xây dựng một ứng dụng giống `ChatGPT` sử dụng `session state` để ghi nhớ ngữ cảnh hội thoại, tất cả trong chưa đầy 50 dòng code.

- Đầu tiên, hãy nói qua về  `st.chat_message`, hàm này  cho phép bạn tạo một hộp tin nhắn trong ứng dụng. Hộp này có thể chứa nhiều thành phần khác nhau của Streamlit như biểu đồ, bảng, hoặc văn bản. Để thêm nội dung vào hộp, bạn dùng cú pháp with.
- Tham số đầu tiên của `st.chat_messag`e là tên của người gửi tin nhắn, có thể là `user` (người dùng) hoặc `assistant` (trợ lý) để sử dụng kiểu dáng và `avatar` sẵn có. Bạn cũng có thể đặt một tên tùy chỉnh, và `avatar` sẽ hiện chữ cái đầu của tên. Dưới đây là một ví dụ đơn giản:

```python
import streamlit as st

with st.chat_message("user"):
    st.write("Hello 👋")
```


- `st.chat_input` cho phép bạn hiển thị một widget nhập liệu trò chuyện để người dùng có thể gõ tin nhắn. Giá trị trả về là đầu vào của người dùng, hoặc None nếu người dùng chưa gửi tin nhắn. Bạn cũng có thể truyền vào một gợi ý mặc định để hiển thị trong widget nhập liệu. Dưới đây là ví dụ về cách sử dụng `st.chat_input` để hiển thị một widget nhập liệu trò chuyện và hiển thị đầu vào của người dùng:

```python
import streamlit as st

prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")
```

### Làm chatbot phản hồi lại y hệt câu của bạn

- Trong phần này, chúng ta sẽ xây dựng một bot phản hồi hoặc lặp lại đầu vào của bạn. Cụ thể hơn, bot sẽ trả lời đầu vào của bạn bằng chính thông điệp đó. Chúng ta sẽ sử dụng `st.chat_message` để hiển thị đầu vào của người dùng và `st.chat_input` để chấp nhận đầu vào từ người dùng. Chúng ta cũng sẽ sử dụng `session state` để lưu trữ lịch sử trò chuyện, để có thể hiển thị nó trong các container tin nhắn. Dưới đây là code mẫu:

```python
import streamlit as st

st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
```

- Trong đoạn mã trên, chúng ta đã thêm tiêu đề cho ứng dụng và một vòng lặp for để lặp qua lịch sử trò chuyện và hiển thị mỗi tin nhắn trong container tin nhắn trò chuyện (với vai trò tác giả và nội dung tin nhắn). Chúng ta cũng đã thêm một kiểm tra để xem khóa messages có tồn tại trong `st.session_state` hay không. Nếu không, chúng ta khởi tạo nó thành một danh sách rỗng. Điều này là vì chúng ta sẽ thêm tin nhắn vào danh sách sau này, và không muốn ghi đè danh sách mỗi lần ứng dụng được chạy lại.

- Bây giờ, chúng ta hãy chấp nhận đầu vào từ người dùng bằng st.chat_input, hiển thị tin nhắn của người dùng trong container tin nhắn trò chuyện và thêm nó vào lịch sử trò chuyện:

```python
# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
```

- Chúng ta đã sử dụng toán tử `:=` để gán đầu vào của người dùng cho biến prompt và kiểm tra xem nó có phải là None hay không trong cùng một dòng. Nếu người dùng đã gửi một tin nhắn, chúng ta sẽ hiển thị tin nhắn đó trong container tin nhắn trò chuyện và thêm nó vào lịch sử trò chuyện.

- Bây giờ, tất cả những gì còn lại là thêm phản hồi của chatbot trong khối `if`. Chúng ta sẽ sử dụng logic giống như trước để hiển thị phản hồi của bot (chỉ là lời nhắc của người dùng) trong container tin nhắn trò chuyện và thêm nó vào lịch sử.

```python
response = f"Echo: {prompt}"
# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.markdown(response)
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})
```
- Dưới đây là full đoạn code:

```python
import streamlit as st

st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
```

- Mặc dù ví dụ trên rất đơn giản nhưng đây là điểm khởi đầu tốt để xây dựng các ứng dụng chatbot phức tạp hơn. Hãy chú ý cách bot phản hồi ngay lập tức với thông tin đầu vào của bạn. Trong phần tiếp theo, ctôi sẽ thêm độ trễ để mô phỏng quá trình "suy nghĩ" của bot trước khi phản hồi.

### Làm chatbot đơn giản phản hổi theo cách streaming

- Trong phần này, chúng ta sẽ xây dựng một giao diện chatbot đơn giản, phản hồi lại đầu vào của người dùng bằng một tin nhắn ngẫu nhiên từ danh sách các phản hồi được xác định trước. Trong phần tiếp theo, chúng ta sẽ chuyển ví dụ thử nghiệm đơn giản này thành một trải nghiệm giống như `ChatGPT` bằng cách sử dụng `OpenAI`.
- Cũng giống như trước đây, chúng ta vẫn cần các thành phần tương tự để xây dựng chatbot của mình. Hai hộp chứa tin nhắn trò chuyện để hiển thị tin nhắn từ người dùng và từ bot. Một tiện ích nhập tin nhắn để người dùng có thể gõ tin nhắn. Và một cách để lưu trữ lịch sử trò chuyện để chúng ta có thể hiển thị trong các hộp chứa tin nhắn trò chuyện.
- Hãy sao chép mã từ phần trước và thêm một vài thay đổi nhỏ vào nó:

```python
import streamlit as st
import random
import time

st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
```

- Điểm khác biệt duy nhất cho đến nay là chúng ta đã thay đổi tiêu đề của ứng dụng và thêm các thư viện `random` và `time`. Chúng ta sẽ sử dụng `random` để chọn ngẫu nhiên một phản hồi từ danh sách các phản hồi và `time` để thêm độ trễ nhằm mô phỏng việc chatbot đang "suy nghĩ" trước khi phản hồi.
- Việc còn lại là thêm các phản hồi của chatbot vào trong khối `if`. Chúng ta sẽ sử dụng một danh sách các phản hồi và chọn ngẫu nhiên một phản hồi để hiển thị. Đồng thời, chúng ta sẽ thêm một độ trễ để mô phỏng việc chatbot đang "suy nghĩ" trước khi phản hồi (hoặc hiển thị dần phản hồi). Hãy tạo một hàm trợ giúp cho việc này và chèn nó vào đầu ứng dụng của chúng ta.

```python
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
```

- Quay lại việc viết phản hồi trong giao diện trò chuyện, chúng ta sẽ sử dụng `st.write_stream` để hiển thị phản hồi theo kiểu hiệu ứng máy đánh chữ.

```python
with st.chat_message("assistant"):
    response = st.write_stream(response_generator())
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})
```

- Ở phần trên, chúng ta đã thêm một placeholder để hiển thị phản hồi của chatbot. Chúng ta cũng thêm một vòng lặp for để duyệt qua từng từ trong phản hồi và hiển thị nó từng từ một. Để mô phỏng việc chatbot đang "suy nghĩ" trước khi trả lời, chúng ta thêm một độ trễ 0.05 giây giữa mỗi từ. Cuối cùng, chúng ta thêm phản hồi của chatbot vào lịch sử trò chuyện. Như bạn có thể đoán, đây là một cách triển khai sơ khai của việc phát trực tuyến nội dung. Trong phần tiếp theo, chúng ta sẽ xem cách triển khai phát trực tuyến với OpenAI.

- Dưới đây là đoạn code full:

```python
import streamlit as st
import random
import time


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
```


### Làm chatbot giống như ChatGPT.
- Bây giờ, khi bạn đã hiểu được các yếu tố cơ bản của giao diện trò chuyện trong `Streamlit`, hãy thực hiện một vài thay đổi để xây dựng ứng dụng giống như `ChatGPT` của riêng chúng ta. Bạn sẽ cần cài đặt thư viện `OpenAI` dành cho Python và lấy một khóa `API` để thực hiện theo hướng dẫn.
- Đầu tiên hãy tải các thư viện cần thiết:
  
```bash
pip install openai streamlit
```

- Tiếp theo, hãy thêm khóa `API OpenAI` của bạn vào `Streamlit secrets`. Chúng ta thực hiện việc này bằng cách tạo tệp `.streamlit/secrets.toml` trong thư mục dự án của bạn và thêm các dòng sau vào tệp đó:

```python
# .streamlit/secrets.toml
OPENAI_API_KEY = "YOUR_API_KEY"
```

- Bây giờ, hãy viết ứng dụng. Chúng ta sẽ sử dụng cùng code như trước, nhưng thay danh sách các phản hồi bằng một lệnh gọi đến `API OpenAI`. Đồng thời, chúng ta cũng sẽ thực hiện một vài thay đổi nữa để làm cho ứng dụng giống `ChatGPT` hơn:
  

```python
import streamlit as st
from openai import OpenAI

st.title("ChatGPT-like clone")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
```

- Tất cả những gì đã thay đổi là chúng ta đã thêm một mô hình mặc định vào `st.session_state` và thiết lập khóa `API OpenAI` từ `Streamlit secrets`. Đây là lúc mọi thứ trở nên thú vị: chúng ta có thể thay thế luồng dữ liệu giả lập (emulated stream) bằng các phản hồi thực từ mô hình OpenAI:

```python
   # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

- Ở trên, chúng ta đã thay thế danh sách các phản hồi bằng một lệnh gọi đến `OpenAI().chat.completions.create`. Chúng ta đã thiết lập `stream=True` để truyền trực tiếp các phản hồi đến giao diện người dùng. Trong lệnh gọi `API`, chúng ta truyền tên mô hình đã được cố định trong `session state` và lịch sử trò chuyện dưới dạng danh sách các tin nhắn. Chúng ta cũng truyền vai trò (`role`) và nội dung (`content`) của mỗi tin nhắn trong lịch sử trò chuyện. Cuối cùng, `OpenAI` trả về một luồng các phản hồi (được chia thành các đoạn nhỏ chứa các token), mà chúng ta sẽ duyệt qua và hiển thị từng đoạn. Dưới đây là full code:

```pythofrom openai import OpenAI
import streamlit as st

st.title("ChatGPT-like clone")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

## 4. Tổng kết

- `Streamlit` là một framework khá hữu ích nếu bạn muốn deploy một sản phẩm nhanh. Tất nhiên là nó sẽ không thể ứng biến cao nếu như bạn code bằng HTML, CSS, JS. `StreamLit` chỉ có tác dụng trình bày nhanh sản phẩm chứ không có tác dụng thương mại hóa.

