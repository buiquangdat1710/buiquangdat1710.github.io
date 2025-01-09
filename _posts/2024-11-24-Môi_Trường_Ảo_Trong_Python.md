---
title: "Môi Trường Ảo Trong Python"
date: 2024-11-24 00:00:00  + 0800
categories: [Machine Learning]
tags: [python]
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

Môi trường ảo (virtual environment) trong Python là một cách để cô lập môi trường làm việc của một dự án cụ thể, đảm bảo rằng các gói thư viện và các phiên bản Python được sử dụng trong dự án không xung đột với các dự án khác trên cùng một máy tính. Nói một cách đơn giản, giả sử bạn đang làm 2 dự án A và B:
- Dự án A: Bạn đang làm một dự án học máy sử dụng `numpy` **phiên bản 1.21.0**, phù hợp với một thư viện như `scikit-learn 0.24.0`.
- Dự án B: Bạn cần phân tích dữ liệu với `numpy` **phiên bản 1.25.0**, cần thiết để chạy thư viện `pandas 2.1.0`.

Nếu cả hai dự án sử dụng cùng một môi trường Python hệ thống:

- Khi bạn cài `numpy 1.25.0` để chạy Dự án B, Dự án A sẽ báo lỗi vì `scikit-learn 0.24.0` không hỗ trợ phiên bản này.
- Nếu bạn quay lại cài `numpy 1.21.0` để sửa Dự án A, Dự án B sẽ không chạy được do `pandas 2.1.0` yêu cầu phiên bản mới hơn.
  
Giải pháp với môi trường ảo:

- Tạo một môi trường ảo riêng cho mỗi dự án.
- Trong Dự án A: Cài `numpy 1.21.0` và `scikit-learn 0.24.0`.
- Trong Dự án B: Cài `numpy 1.25.0` và `pandas 2.1.0`.

![anh](./image/python-virtual-envs1.webp)

Thậm chí, mỗi dự án của bạn đôi khi lại cần một phiên bản python khác nhau:
- Dự án A: Là một dự án cũ yêu cầu `Python 3.8`, vì thư viện `tensorflow 2.4.0` chỉ hoạt động ổn định trên phiên bản này.
- Dự án B: Là một dự án mới yêu cầu `Python 3.11`, vì bạn cần sử dụng các tính năng mới của Python và thư viện `pandas 2.1.0` chỉ hỗ trợ tốt trên `Python >= 3.10`.

Dưới đây là một số công cụ để tạo môi trường ảo trong Python:

- venv (tích hợp sẵn trong Python)
- virtualenv
- conda
- pyenv (kết hợp với pyenv-virtualenv)
- Pipenv
- docker (không chỉ cho Python, nhưng hỗ trợ tạo môi trường cô lập)

Nhưng ở bài viết này, tôi sẽ chỉ tập trung nói về conda vì tôi nghĩ rằng conda là công cụ hoàn hảo để tạo môi trưởng ảo.

## 1. Giới thiệu về Conda.

- `Conda` là một công cụ quản lý môi trường và quản lý gói (package manager) mạnh mẽ, giúp cài đặt, cập nhật, và quản lý các thư viện, phần mềm trong các môi trường cô lập. Conda có thể được sử dụng cho nhiều ngôn ngữ, không chỉ Python, và đặc biệt hữu ích trong các dự án khoa học dữ liệu, học máy vì nó hỗ trợ cài đặt các gói phần mềm phức tạp. Bạn có thể sử dụng conda thông qua **Anaconda** hoặc **Miniconda**.

### Sự khác nhau giữa Anaconda và Miniconda:

![anh](./image/conda_diagram.png.webp)

- **Anaconda**:
  - Là một bản phân phối Python đầy đủ, bao gồm nhiều thư viện phổ biến cho khoa học dữ liệu (như numpy, pandas, scikit-learn, matplotlib, ...) và công cụ như Jupyter Notebook.
  - Dung lượng lớn (~3GB), vì nó cài sẵn rất nhiều thư viện.
  - Phù hợp với người mới bắt đầu vì nó cung cấp một môi trường Python đã được cấu hình sẵn.
- **Miniconda**:
  - Là một phiên bản nhẹ hơn của **Anaconda**, chỉ cài đặt conda và một số gói cơ bản.
  - Dung lượng nhỏ (~50MB), bạn có thể cài đặt chỉ những thư viện cần thiết, giúp tiết kiệm không gian đĩa.
  - Phù hợp nếu bạn muốn kiểm soát hoàn toàn các gói cài đặt và không cần đến toàn bộ thư viện có sẵn trong Anaconda.

### Nên sử dụng cái nào?
- **Anaconda**: Nếu bạn muốn một công cụ đầy đủ với nhiều thư viện và công cụ hỗ trợ sẵn có mà không cần phải cài thêm nhiều gói.
- **Miniconda**: Nếu bạn muốn tiết kiệm dung lượng và chỉ cài đặt các thư viện mà bạn cần cho dự án của mình.
- Cá nhân tôi thì tôi sử dụng **Miniconda**, đơn giản vì nó nhẹ và tôi cũng không cần nhiều thư viện hỗ trợ sẵn ở bên **Anaconda**. Nhưng dùng cái nào thì hoàn toàn phụ thuộc vào nhu cầu của bạn.
  
## 2. Tạo môi trường ảo bằng Conda.

- Đầu tiên, bạn cần phải cài đặt [Anaconda](https://www.anaconda.com/download) hoặc [Miniconda](https://docs.anaconda.com/miniconda/) trên trang web.
- Kiểm tra xem conda đã được cài đặt chưa bằng lệnh:

```bash
conda --version
```

- Nếu của bạn chưa được cài đặt đúng, khả năng cao bạn sẽ phải thêm một số đường dẫn tuyệt đối đến chỗ bạn để file cài đặt conda trong Environment Variables, cái này thì hỏi ChatGPT cho tiện nhé 😆.

- Tiếp theo, bạn có thể liệt kê danh sách các môi trường ảo bạn đã tạo bằng lệnh:
  
```bash
conda env list
```

- Nếu bạn mới dùng conda lần đầu thì sẽ thấy nó không hiện ra gì vì bạn chưa tạo môi trường ảo nào. Bây giờ, hãy thử tạo môi trường ảo của riêng bạn bằng lệnh sau:

```bash
conda create --name <Tên_bạn_muốn_đặt_cho_môi_trường_ảo> python=<Phiên_bản_python_bạn_muốn_dùng>
```

- Ví dụ tôi muốn đặt tên là py39 và sử dụng phiên bản python là 3.9 thì tôi có thể code như sau (lưu ý là tên môi trường ảo bạn đặt bằng tên gì cũng được nhưng nên đặt tên gợi nhớ về dự án bạn, tôi đặt là py39 vì tôi muốn biết rằng môi trường này sử dụng dụng `python=3.9`)

```bash
conda create --name py39 python=3.9
```

- Sau đó hãy đợi một lúc để chương trình khởi tạo môi trường ảo, bấm Y cho mọi câu hỏi nếu có. Kết thúc quá trình, máy của bạn đã có một môi trưởng ảo, và để khởi tạo môi trường ảo ý, bạn hãy dùng câu lệnh:

```bash
conda activate <Tên_bạn_muốn_đặt_cho_môi_trường_ảo>
```

- Trong trường hợp của tôi thì sẽ là:

```bash
conda activate py39
```

- Lúc này nếu bạn thấy tên môi trưởng ảo hiện ở đầu đường dẫn và trong dấu ngoặc tròn như ở dưới thì xin chúc mừng bạn, bạn đã khởi tạo thành công môi trường ảo và có thể sẵn sàng dùng nó:

```bash
(py39) dat@dat:~/Downloads/buiquangdat1710.github.io$ 
```

- Giờ hãy cài đặt các thư viện bằng lệnh `pip` hoặc `conda`, có thể chỉ định chi tiết phiên bản của thư viện, nếu không chỉ định thì máy tính sẽ mặc định tải phiên bản mới nhất:

```bash
conda install numpy=1.21
``` 

```bash
pip install numpy
```

- Để dùng một môi trường ảo khác, bạn cần phải hủy bỏ môi trưởng ảo hiện tại bằng lệnh:

```bash
conda deactivate
```

- Sau đó thì tạo môi trưởng ảo khác bằng các câu lệnh như trên. Ở đây tôi chỉ liệt kê một số câu lệnh conda cơ bản, bạn có thể tự tìm hiểu thêm nhiều câu lệnh nữa trên mạng.

## 3. Tổng Kết.

Sử dụng môi trường ảo trong Python giúp:
  - Cô lập phụ thuộc: Đảm bảo các thư viện của dự án này không ảnh hưởng đến dự án khác.
  - Tránh xung đột phiên bản: Dễ dàng quản lý các phiên bản thư viện khác nhau cho từng dự án.
  - Quản lý dự án dễ dàng: Giúp bạn duy trì các gói riêng biệt cho từng dự án.
  - Giảm lỗi tương thích: Tạo môi trường kiểm tra độc lập cho mỗi dự án.
  - Tái tạo môi trường: Chia sẻ cấu hình môi trường dễ dàng giữa các máy tính hoặc người dùng.