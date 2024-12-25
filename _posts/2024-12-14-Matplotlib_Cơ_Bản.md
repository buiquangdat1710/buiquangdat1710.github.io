---
title: "Matplotlib Cơ Bản"
date: 2024-12-14 00:00:00  + 0800
categories: [Giáo Trình Dạy AI ProPTIT]
tags: [matplotlib]
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

Matplotlib là một thư viện mạnh mẽ và phổ biến trong Python được sử dụng để tạo các biểu đồ và hình ảnh minh họa dữ liệu. Được phát triển bởi John D. Hunter, Matplotlib cung cấp giao diện linh hoạt cho phép người dùng dễ dàng vẽ các biểu đồ như biểu đồ đường, cột, tròn, scatter và nhiều loại khác. 

## 1. Cách cài đặt Matplotlib


- Bạn có thể gõ lệnh sau trong cmd sau khi tạo môi trường ảo để cài đặt thư viện `Matplotlib`:

```
pip install matplotlib
```

- Dùng lệnh dưới và chạy trong một cell nếu bạn đang dùng Google Colab, Kaggle Notebook,...:

```
!pip install matplotlib
```

- Vì `Matplotlib` là một thư viện trong python nên bạn có thể dùng `import` để khai báo:

```python
import matplotlib
print(matplotlib.__version__)
```
> 3.8.0

- Người ta thường dùng từ `plt` để viết tắt cho `matplotlib`. Dưới đây là một ví dụ vẽ đường thẳng đơn giản từ hai điểm:

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.array([0,6])
y = np.array([0,250])
plt.plot(x, y)
plt.show()
```

![anh](./image/124.png)

## 2. Các hàm cơ bản trong Matplotlib

- Ham `plot` được sử dụng để vẽ các điểm (markers) trong biểu đồ. Mặc định hàm này sẽ nối các điểm bằng một đường thẳng. Hàm này nhận các tham số để xác định các điểm trong biểu đồ, tham số thứ nhất là một mảng chứ tọa độ trục x, tham số thứ hai là một mảng chứa tọa độ trên trục y. 
- Ví dụ, để vẽ một đường thẳng từ điểm (1,3) đến (8,10), ta cần truyền hai mảng `[1,8]` và `[3,10]` vào hàm `plot`:

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.array([1,8])
y = np.array([3,10])
plt.plot(x,y)
plt.show()
```

![anh](./image/125.png)

- Nếu bạn chỉ muốn vẽ điểm, bạn có thể truyền thêm tham số `'o'` vào hàm `plot`:

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.array([1,8])
y = np.array([3,10])
plt.plot(x, y, 'o')
plt.show()
```

![anh](./image/126.png)

- Bạn có thể vẽ đồ thị nhiều điểm, chỉ cần đảm bảo là hai trục có cùng số lượng phần tử là được:

```python
import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])

plt.plot(xpoints, ypoints)
plt.show()
```

![anh](./image/127.png)

- Nếu bạn không truyền trục x vào hàm `plot` thì hàm này sẽ mặc định trục x là 0,1,2,3,... phụ thuộc vào số lượng của mảng y:

```python
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10, 5, 7])

plt.plot(ypoints)
plt.show()
```

![anh](./image/128.png)


### Markers 

- Bạn có thể dùng tham số `marker` để biểu thị cho các điểm. Ví dụ bạn muốn các điểm có hình tròn thì bạn có thể code như sau:

```python
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10, 5, 7])

plt.plot(ypoints, marker = 'o')
plt.show()
```
![anh](./image/129.png)

- Hoặc là nếu bạn muốn các điểm có hình dạng như ngôi sao:

```python
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10, 5, 7])

plt.plot(ypoints, marker = '*')
plt.show()
```
![anh](./image/130.png)

- Tham khảo bảnh các marker có thể dùng dưới đây:

| Marker   | Mô tả                    |
|----------|--------------------------|
| `'.'`    | Điểm                    |
| `','`    | Pixel                   |
| `'o'`    | Hình tròn               |
| `'v'`    | Tam giác hướng xuống    |
| `'^'`    | Tam giác hướng lên      |
| `'<'`    | Tam giác hướng trái     |
| `'>'`    | Tam giác hướng phải     |
| `'1'`    | Tam giác nhỏ hướng xuống|
| `'2'`    | Tam giác nhỏ hướng lên  |
| `'3'`    | Tam giác nhỏ hướng trái |
| `'4'`    | Tam giác nhỏ hướng phải |
| `'s'`    | Hình vuông              |
| `'p'`    | Hình ngũ giác           |
| `'*'`    | Hình sao                |
| `'h'`    | Lục giác (kiểu 1)       |
| `'H'`    | Lục giác (kiểu 2)       |
| `'+'`    | Dấu cộng                |
| `'x'`    | Dấu nhân                |
| `'D'`    | Hình thoi               |
| `'d'`    | Hình thoi mỏng          |
| `'|'`    | Đường thẳng đứng        |
| `'_'`    | Đường thẳng ngang       |
| `''`     | Không có marker         |
| `None`   | Mặc định (không có marker) |
| `' '`    | Không có marker (cách khác) |
| `'$...$'`| Biểu thức LaTeX làm marker (ví dụ: `'$\\alpha$'`) |


- Bạn có thể dùng một tham số gọi là `fmt` để chỉ thị 3 thứ: marker, line, color. Tham số `fmt` sẽ có dạng `marker|line|color`, xem ví dụ dưới để hiểu rõ hơn:

```python
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, 'o:r')
plt.show()
```

![anh](./image/131.png)

- `'o'` có nghĩa là marker sẽ có dạng hình tròn, `':'` có nghĩa là line sẽ là dạng đường nét đứt, `r` có nghĩa là màu đỏ (red). Dưới đây là bảng line và color:

| Line Style | Mô tả                  |
|------------|------------------------|
| `'-'`      | Đường thẳng liền       |
| `'--'`     | Đường gạch đứt         |
| `'-.'`     | Đường gạch chấm        |
| `':'`      | Đường chấm chấm        |
| `''`       | Không có đường kẻ      |
| `None`     | Không có đường kẻ (tương tự) |


| Ký hiệu | Màu                     |
|---------|-------------------------|
| `'b'`   | Xanh dương (blue)       |
| `'g'`   | Xanh lá cây (green)     |
| `'r'`   | Đỏ (red)                |
| `'c'`   | Xanh ngọc (cyan)        |
| `'m'`   | Tím (magenta)           |
| `'y'`   | Vàng (yellow)           |
| `'k'`   | Đen (black)             |
| `'w'`   | Trắng (white)           |

- Bạn có thể chỉnh kích thước của marker bằng tham số  `markersize` hoặc ngắn ngọn hơn là `ms`:

```python
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, marker = 'o', ms = 20)
plt.show()
```

![anh](./image/132.png)

### Labels

- Bạn có thể dùng hàm `xlabel` và `ylabel` để đặt tên cho trục x và trục y:

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)

plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.show()
```

![anh](./image/133.png)

- Bạn có thể dùng hàm `title` để đặt tên cho đồ thị:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.show()
```

![anh](./image/134.png)

- Bạn còn có thể thay đổi font chữ, loại chữ, màu chữ với tham số  `fontdict` trong hàm `xlabel`, `ylabel`, `title`:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.title("Sports Watch Data", fontdict = font1)
plt.xlabel("Average Pulse", fontdict = font2)
plt.ylabel("Calorie Burnage", fontdict = font2)

plt.plot(x, y)
plt.show()
```

![anh](./image/135.png)

- Bạn có thể căn giữa, căn trái, căn phải `title` bằng tham số  `loc`:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data", loc = 'left')
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)
plt.show()
```

![anh](./image/136.png)


### Subplot

- Với hàm `subplot`, bạn có thể vẽ nhiều đồ thị trong một hình:

```python
import matplotlib.pyplot as plt
import numpy as np
# đồ thị 1
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])
plt.subplot(1, 2, 1) # Hình vẽ chia làm 1 hàng và 2 cột, và đồ thị này sẽ được vẽ ở ô thứ 1
plt.plot(x,y)
# đồ thị 2
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(1, 2, 2) # Hình vẽ chia làm 1 hàng và 2 cột, và đồ thị này sẽ được vẽ ở ô thứ 2
plt.plot(x,y)

plt.show()
```

![anh](./image/137.png)

- Nếu chúng ta muốn vẽ hai hình, một hình trên và một hình dưới thì có thể code như sau:

```python
import matplotlib.pyplot as plt
import numpy as np

#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 1, 1)
plt.plot(x,y)

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 1, 2)
plt.plot(x,y)

plt.show()
```

![anh](./image/138.png)

- Bạn có thể dùng hàm `title` sau mỗi lần dùng `plot` để viết title cho từng hình, đồng thời bạn cũng có thể dùng hàm `suptitle` để làm title cho toàn thể:

```python
import matplotlib.pyplot as plt
import numpy as np

#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(1, 2, 1)
plt.plot(x,y)
plt.title("SALES")

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(1, 2, 2)
plt.plot(x,y)
plt.title("INCOME")

plt.suptitle("MY SHOP")
plt.show()
```

![anh](./image/139.png)

- Chúng ta có thể vẽ nhiều đường trên một đồ thị như sau:
  
```python
import matplotlib.pyplot as plt
import numpy as np

y1 = np.array([3, 8, 1, 10])
y2 = np.array([6, 2, 7, 11])

plt.plot(y1)
plt.plot(y2)

plt.show()
```

![anh](./image/140.png)

### Scatter

- Bạn có thể dùng hàm `scatter` để biểu diễn các điểm:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

plt.scatter(x, y)
plt.show()
```

![anh](./image/141.png)

- Xem thêm nhiều thứ hay ho về `scatter` tại: [w3school](https://www.w3schools.com/python/matplotlib_scatter.asp) (Do tôi cảm thấy mấy cái sau không có nhiều ích lợi lắm nên tôi sẽ không trình bày)

### Bars

- Để tạo biểu đồ cột, bạn có thể sử dụng hàm `bar`:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.bar(x,y)
plt.show()
```

![anh](./image/142.png)

- Để vẽ biểu đồ cột ngang bạn có thể dùng hàm `barh`:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.barh(x, y)
plt.show()
```

![anh](./image/143.png)

### Histogram

- Histogram là dạng biểu đồ thể hiện tần xuất. Nếu bạn hỏi chiều cao của 250 người, bạn có thể có biểu đồ dưới đây:

![anh](./image/144.png)

- Nhìn vào biểu đồ, bạn có thể ước lượng những điều sau:
  - 2 người cao từ 140 đến 145cm
  - 5 người cao từ 145 đến 150cm
  - 15 người cao từ 151 đến 156cm
  - 31 người cao từ 157 đến 162cm
  - 46 người cao từ 163 đến 168cm
  - 53 người cao từ 168 đến 173cm
  - 45 người cao từ 173 đến 178cm
  - 28 người cao từ 179 đến 184cm
  - 21 người cao từ 185 đến 190 cm
  - 4 người cao từ 190 đến 195cm

- Để vẽ histogram, bạn có thể dùng hàm `hist` như sau:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(170, 10, 250)

plt.hist(x)
plt.show() 
```

![anh](./image/145.png)

### Pie Charts

- Để vẽ biểu đồ tròn, bạn có thể dùng hàm `pie`:

```python
import matplotlib.pyplot as plt
import numpy as np

y = np.array([35, 25, 25, 15])

plt.pie(y)
plt.show() 
```

![anh](./image/146.png)

- Vì tôi cũng ít dùng biểu đồ này và cũng lười viết tiếp nên hãy tham khảo thêm tại: [w3school](https://www.w3schools.com/python/matplotlib_pie_charts.asp)

## 3. Tổng kết.

Matplotlib là công cụ mạnh mẽ giúp trực quan hóa dữ liệu trong Python, từ phân tích cơ bản đến ứng dụng trong AI như minh họa quá trình huấn luyện mô hình và đánh giá hiệu suất. Nắm vững Matplotlib là nền tảng để khai thác dữ liệu hiệu quả.