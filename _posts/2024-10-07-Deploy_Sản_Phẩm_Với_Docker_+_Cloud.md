---
title: "Deploy Sản Phẩm Với Docker + Cloud"
date: 2024-10-07 00:00:00  + 0800
categories: [AI Production]
tags: [docker, cloud]
---
---

Deploy sản phẩm với Docker và Cloud là phương pháp hiện đại giúp triển khai ứng dụng một cách nhanh chóng, hiệu quả và linh hoạt. Docker cho phép đóng gói ứng dụng và các phụ thuộc vào một container, giúp đảm bảo tính nhất quán khi chạy trên bất kỳ môi trường nào. Kết hợp với các dịch vụ cloud như AWS, Azure hay Google Cloud, việc deploy trở nên dễ dàng hơn bao giờ hết, cho phép mở rộng quy mô ứng dụng, tối ưu tài nguyên và đảm bảo tính ổn định. Đây là giải pháp phổ biến cho các sản phẩm công nghệ hiện đại, giúp tăng tốc độ phát triển và triển khai.

## 1. Docker là gì ?

- **Docker** là một nền tảng mã nguồn mở giúp tự động hóa việc triển khai ứng dụng bên trong các container phần mềm. **Container** là một phương pháp đóng gói ứng dụng và các thành phần phụ thuộc của nó (như thư viện, công cụ, mã nguồn) thành một môi trường có thể chạy độc lập với hệ điều hành. Điều này giúp đảm bảo rằng ứng dụng có thể hoạt động nhất quán trên mọi môi trường từ máy phát triển cho đến các máy chủ sản xuất.
- Nói một cách đơn giản, Docker là công cụ giúp đóng gói ứng dụng và các thư viện phụ thuộc vào một container để chạy trên bất kỳ môi trường nào mà không gặp lỗi. Điều này giống như việc bạn bỏ tất cả mọi thứ cần thiết để chạy ứng dụng vào một "hộp", rồi di chuyển "hộp" đó sang bất kỳ nơi đâu mà ứng dụng vẫn chạy tốt.

## 2. Pipeline giữa Docker, DockerHub, Cloud.

![Anh](./image/Docker.png)

- Giải thích một cách đơn giản thì giờ bạn đang có một project và bạn muốn người khác sử dụng mà không cần tải các thư viện liên quan. Ví dự như sản phẩm của bạn được code bằng ngôn ngữ python và phần file requirements.txt của bạn chứa rất nhiều thư viện với các version khác nhau, và bạn không muốn khi người khác chạy sản phẩm của bạn, họ sẽ không phải tải lại các thư viện đó nữa. Một cách giải quyết đó chính là sử dụng Docker.
- Mình sẽ nói một cách đơn giản nhất mà ai cũng có thể hiểu được về Pipeline giữa Docker, DockerHub, Cloud. Sau khi bạn có một sản phẩm rồi thì bạn sẽ push Image Docker lên DockerHub, việc này giống như bạn push code lên GitHub vậy. Tiếp theo bạn sẽ phải thuê một bên thứ ba để deploy sản phầm của bạn, hay còn được gọi với cái tên là **Cloud**. Sau khi thuê xong Cloud thì bạn có thể pull từ DockerHub về Cloud và deploy sản phẩm.
- Mình đã chuẩn bị sẵn một sản phẩm đơn giản là Cat Management API. Bạn có thể clone bằng lệnh sau:

`git clone https://github.com/buiquangdat1710/Cat-Management-API-Demo.git`

- Bạn có thể chạy thử sản phẩm này bằng cách chạy hai lệnh sau, nhớ là chạy hai lệnh song song, tức là chạy trên hai cửa số terminal:

`python serve.py`

`streamlit run client.py`

- Project bạn vừa clone đã chứa sẵn 2 file quan trọng là `.Dockerfile` và `docker-compose.yaml`. Nói qua một chút về project thì nó sẽ có 2 file là `client.py` để hiện giao diện và file `server.py` để hiện phần xử lý backend. Thông thường sau khi code xong một project thì bạn sẽ phải tạo hai file `.Dockerfile` và `docker-compose.yaml` nhưng mình đã tạo sẵn cho bạn và mình sẽ nói chi tiết 2 file này ở phần sau.

- Trong bài viết này thì mình sẽ sử dụng DigitalOcean làm Cloud để deploy sản phẩm, bạn hoàn toàn có thể sử dụng các dịch vụ khác như AWS, Google Cloud, etc. Trước tiên hãy tạo một project mới trên DigitalOcean:

![Anh](./image/CreateProject.png)

- Tiếp theo hãy bấm vào Create Droplets, hay chính là tạo ra một cloud servers:

![Anh](./image/Droplets.png)

- Sau khi hoàn thành hết tất cả các bước thì chúng ta có thể truy cập vào IP như sau, lưu ý là phần ipv4. Bạn có thể mua domain riêng ở ngoài và nối vào ip này:

![Anh](./image/IP.png)

- Tuy nhiên, giờ chúng ta vẫn chưa thiết lập gì nên sẽ không truy cập được vào trang web này. Giờ chúng ta có thể truy cập vào cửa sổ console để thiết lập như một cửa sổ console thông thường.
- Giờ hãy xem qua về file `.Dockerfile`:

```python
# Sử dụng Python làm image cơ bản
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép mã nguồn và các yêu cầu
COPY . /app

# Cài đặt các thư viện cần thiết
RUN pip install -r server-requirements.txt
RUN pip install -r client-requirements.txt

# Mở cổng cho Flask và Streamlit
EXPOSE 5002
EXPOSE 8501

# Chạy cả hai ứng dụng cùng lúc
CMD ["sh", "-c", "python server.py & streamlit run client.py --server.port=8501 --server.address=0.0.0.0"]

```

- Đây là một tệp Dockerfile dùng để tạo một Docker image cho ứng dụng Python, trong đó chạy cả ứng dụng Flask (server.py) và Streamlit (client.py). File này thiết lập môi trường cần thiết để container hoạt động, bao gồm việc cài đặt các thư viện và cấu hình chạy đồng thời hai ứng dụng. Dưới đây là giải thích chi tiết cho từng dòng:

  - `FROM python:3.9-slim`: Dòng này xác định image cơ bản để xây dựng Docker image của bạn. Ở đây, sử dụng image python:3.9-slim, là một phiên bản tối giản của Python 3.9, chỉ bao gồm các thành phần cần thiết để chạy Python mà không có các công cụ không cần thiết.
  - `WORKDIR /app`: Thiết lập thư mục làm việc mặc định trong container là /app. Tất cả các lệnh sau đó sẽ được thực hiện trong thư mục này. Nếu thư mục không tồn tại, Docker sẽ tự động tạo ra.
  - `COPY . /app`: Sao chép toàn bộ nội dung của thư mục hiện tại trên máy chủ (thư mục chứa tệp Dockerfile) vào thư mục /app trong container. Điều này bao gồm cả mã nguồn và các tệp cấu hình.
  - `RUN pip install -r server-requirements.txt`: Cài đặt các thư viện Python cần thiết để chạy ứng dụng Flask từ tệp server-requirements.txt. Tệp này thường chứa danh sách các thư viện với phiên bản cụ thể mà ứng dụng Flask phụ thuộc.
  - `RUN pip install -r client-requirements.txt`: Cài đặt các thư viện Python cần thiết để chạy ứng dụng Streamlit từ tệp client-requirements.txt. Tệp này cũng chứa danh sách các thư viện cần thiết cho ứng dụng Streamlit.
  - `EXPOSE 5002`: Mở cổng 5002 để container có thể lắng nghe các yêu cầu đến cho ứng dụng Flask.
  - `EXPOSE 8501`: Mở cổng 8501 để container có thể lắng nghe các yêu cầu đến cho ứng dụng Streamlit.
  - `CMD ["sh", "-c", "python server.py & streamlit run client.py --server.port=8501 --server.address=0.0.0.0"]`: Lệnh CMD dùng để chỉ định câu lệnh mặc định khi container được khởi động. Ở đây, câu lệnh này sử dụng sh -c để chạy cả hai ứng dụng cùng lúc:
    - python server.py & để chạy ứng dụng Flask ở chế độ nền.
    - streamlit run client.py --server.port=8501 --server.address=0.0.0.0 để chạy ứng dụng Streamlit, lắng nghe tại cổng 8501 và chấp nhận kết nối từ tất cả các địa chỉ IP (0.0.0.0).

- Tiếp theo bạn cần đăng nhập vào DockerHub và tạo một repo mới. Như trong hình mình đã tạo một repo có tên là test:
  
![Anh](./image/DockerHub.png)

- Bạn sẽ cần phải dùng lệnh sau ở trên terminal của máy mình (không phải trên cửa sổ console của DigitalOcean) dể xây dựng một Docker image từ thư mục hiện tại:

`docker build ./ -f .Dockerfile -t buiquangdat1710/test:flask-resful-api-v1.1`

- `docker build ./`: Dùng để xây dựng (build) một Docker image từ thư mục hiện tại (./). Thư mục này chứa các tệp cần thiết để xây dựng image, bao gồm tệp Dockerfile và các tệp nguồn của dự án.
- `-f .Dockerfile`: Chỉ định tệp Dockerfile cụ thể để sử dụng cho quá trình build. Trong trường hợp này, tệp .Dockerfile là tệp cấu hình Docker dùng để tạo image.
- `-t buiquangdat1710/test:flask-resful-api-v1.1`: Tùy chọn -t dùng để gán thẻ (tag) cho Docker image. Tên của image ở đây là buiquangdat1710/test và tag (phiên bản) của image là flask-resful-api-v1.1. Điều này giúp phân biệt image này với các phiên bản hoặc image khác cùng tên. Lưu ý rằng là bạn có thể đặt tên tài khoản và tên tag khác tài khoản mình nên chỗ này phải sửa lại nhé.

- Tiếp theo, ta cần đẩy Docker Image này lên DockerHub bằng lệnh sau:

`docker push buiquangdat1710/test:flask-resful-api-v1.1`

- Tiếp theo, ở cửa sổ console của DigitalOcean, ta sẽ pull về bằng lệnh sau:

`docker pull buiquangdat1710/test:flask-resful-api-v1.1`

- Lưu ý rằng bạn phải cài đặt docker cũng như tạo môi trường ảo cần thiết trên cửa sổ console của DigitalOcean. Tiếp theo bạn hãy chạy lệnh này:

`docker run -d -p 5002:5002 -p 8501:8501 --name test buiquangdat1710/test:flask-resful-api-v1.1`

- `docker run`: Đây là lệnh dùng để khởi chạy một container mới từ một Docker image.
- `-d`: Tùy chọn -d (detached mode) cho phép container chạy ở chế độ nền. Khi sử dụng tùy chọn này, container sẽ khởi chạy và tiếp tục chạy ở chế độ nền thay vì chiếm quyền điều khiển của terminal.
- `-p 5002:5002 -p 8501:8501`: Tùy chọn -p dùng để ánh xạ các cổng giữa máy chủ (host) và container.
- `5002:5002`: Ánh xạ cổng 5002 của máy chủ với cổng 5002 của container. Điều này có nghĩa là các yêu cầu gửi tới cổng 5002 trên máy chủ sẽ được chuyển tiếp tới cổng 5002 của container.
- `8501:8501`: Ánh xạ tương tự, nhưng cho cổng 8501. Việc ánh xạ cổng này giúp các ứng dụng hoặc dịch vụ đang chạy bên trong container có thể được truy cập từ bên ngoài.
- `--name test`: Tùy chọn --name dùng để đặt tên cho container. Trong trường hợp này, container sẽ có tên là test.Tên này có thể được sử dụng để quản lý container dễ dàng hơn, ví dụ như dừng (docker stop test), xóa (docker rm test), hoặc kiểm tra log (docker logs test).
- `buiquangdat1710/test:flask-resful-api-v1.1`: Đây là tên của Docker image mà container sẽ được khởi tạo từ đó.

- Giờ bạn có thể truy cập vào IP `http://104.248.159.85:5002/` để xem backend, truy cập vào IP `http://104.248.159.85:8501/` để xem phần giao diện.

## 3. Tài liệu tham khảo.
- [1] [Docker - ProtonX](https://protonx.io/courses/66e7b29476c94100195c25cd/topics/66f0dd6e76c941001963e0fb)

