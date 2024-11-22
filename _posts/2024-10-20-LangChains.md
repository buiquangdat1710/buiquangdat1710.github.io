---
title: "Làm Quen Với LangChain"
date: 2024-10-20 00:00:00  + 0800
categories: [AI Production]
tags: [langchain]
---
---

<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [['$','$'], ['\\(','\\)']],
            processEscapes: true
        }
    });
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML-full"></script>

LangChain là một framework mạnh mẽ được thiết kế để xây dựng các ứng dụng dựa trên ngôn ngữ tự nhiên sử dụng các mô hình ngôn ngữ lớn (LLMs). Với khả năng tích hợp linh hoạt các công cụ như các cơ sở dữ liệu, API, và các mô hình ngôn ngữ hiện đại, LangChain cho phép bạn tạo ra các ứng dụng thông minh, có khả năng tương tác và suy luận phức tạp. Nó hỗ trợ nhiều tính năng từ truy vấn dữ liệu, tự động hóa quy trình đến tối ưu hóa chuỗi hội thoại, giúp người dùng phát triển các giải pháp AI sáng tạo và linh hoạt trong nhiều lĩnh vực.

## 1. Giới Thiệu.

- LangChain hoạt động bằng cách xâu chuỗi các thành phần lại với nhau để tạo ra một quy trình xử lý ngôn ngữ tự nhiên liền mạch. Giờ hãy xem một đoạn code sau:

```python
class Step1:
    def __init__(self, data):
        self.data = data

    def __or__(self, next_step):
        # Phương thức __or__ được sử dụng để xâu chuỗi các bước lại với nhau.
        self.next_step = next_step
        # Khi bạn viết (step1 | step2 | step3), bạn đang thực hiện liên kết step1 với step2, và sau đó step2 với step3.
        return self
    def invoke(self):
        # Step1 sẽ chuyển đổi data thành chữ in hoa
        transformed_data = self.data.upper()
        return self.next_step.invoke(transformed_data)

class Step2:
    # Phương thức __or__ được sử dụng để xâu chuỗi các bước lại với nhau.
    def __or__(self, next_step):
        self.next_step = next_step
        # Khi bạn viết (step1 | step2 | step3), bạn đang thực hiện liên kết step1 với step2, và sau đó step2 với step3.
        return self

    def invoke(self, data):
        # Step2 sẽ đảo ngược chuỗi
        transformed_data = data[::-1]
        return self.next_step.invoke(transformed_data)

class Step3:
    def invoke(self, data):
        # Step3 sẽ thay thế khoảng trắng bằng dấu gạch ngang
        transformed_data = data.replace(" ", "-")
        return transformed_data

# Sử dụng
input_data = "hello world"
step1 = Step1(input_data)
step2 = Step2()
step3 = Step3()

# Chuỗi các bước
chain = (step1 | step2 | step3).invoke()
print(chain)
```

> HELLO-WORLD

- Class Step1:

    - Phương thức __init__: Nhận dữ liệu ban đầu và gán nó cho thuộc tính data.
    - Phương thức __or__: Gán bước tiếp theo (next_step) và trả về chính đối tượng để cho phép xâu chuỗi các bước.
    - Phương thức **invoke**: Chuyển đổi dữ liệu thành chữ in hoa và gọi phương thức invoke của bước tiếp theo với dữ liệu đã chuyển đổi.
- Class Step2:
    - Phương thức __or__: Giống như trong Step1, gán bước tiếp theo và trả về chính đối tượng.
    - Phương thức **invoke**: Đảo ngược chuỗi dữ liệu và gọi phương thức invoke của bước tiếp theo với dữ liệu đã chuyển đổi.
- Class Step3:

  - Phương thức **invoke**: Thay thế khoảng trắng trong chuỗi bằng dấu gạch ngang và trả về chuỗi đã chuyển đổi.
- Sử dụng:

    - Tạo đối tượng Step1 với dữ liệu đầu vào "hello world".
    - Tạo đối tượng Step2 và Step3.
    - Chuỗi các bước bằng cách sử dụng toán tử | và gọi phương thức invoke trên chuỗi các bước.

- LangChain cũng họat động theo dạng chuỗi như thế. Ta chỉ cần lập trình các thành phần và xâu chuỗi lại với nhau, điều này khiến LangChain rất linh hoạt