---
title: "Pandas Cơ Bản"
date: 2024-12-12 00:00:00  + 0800
categories: [Machine Learning]
tags: [pandas]
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

Pandas là thư viện Python quan trọng trong học AI, hỗ trợ xử lý và phân tích dữ liệu dễ dàng. Với các cấu trúc như `DataFrame`, Pandas giúp đọc, làm sạch, và biến đổi dữ liệu nhanh chóng, tạo nền tảng vững chắc cho các bước tiền xử lý và mô hình hóa dữ liệu trong AI.

## 1. Cách cài đặt Pandas

- Bạn có thể gõ lệnh sau trong cmd sau khi tạo môi trường ảo để cài đặt thư viện `Pandas`:

```
pip install pandas
```

- Dùng lệnh dưới và chạy trong một cell nếu bạn đang dùng Google Colab, Kaggle Notebook,...:

```
!pip install pandas
```

## 2. Các hàm cơ bản trong Pandas

- Vì `Pandas` là một thư viện nên chúng ta phải dùng lệnh `import`, dưới đây là một ví dụ:

```python
import pandas
mydataset = {
    'cars': ["BMW", "Volvo", "Ford"],
    'passings': [3,7,2]
}
myvar = pandas.DataFrame(mydataset)
print(myvar)
```

|   |   cars   | passings |
|---|----------|----------|
| 0 |   BMW    |    3     |
| 1 |  Volvo   |    7     |
| 2 |   Ford   |    2     |

- Người ta thường dùng `pd` để viết tắt cho `pandas`. Chúng ta có thể kiểm tra phiên bản `pandas` bằng câu lệnh:

```python
import pandas as pd
print(pd.__version__)
```
> 2.2.2

### Series trong Pandas

- Một `Series` trong `Pandas` giống như một cột trong bảng. Nó là một mảng một chiều chứa dữ liệu thuộc bất kỳ loại nào. Ví dụ:

```python
import pandas as pd
a = [1, 7, 2]
myvar = pd.Series(a)
myvar
```

|---|-------|
| 0 |   1   |
| 1 |   7   |
| 2 |   2   |

> dtype: int64

- Bạn có thể truy cập vào chỉ số trong `Series` như sau:

```python
import pandas as pd
a = [1, 7, 2]
myvar = pd.Series(a)
print(myvar[0])
```
> 1

- Hoặc bạn có thể tự tạo nhãn như sau:

```python
import pandas as pd
a = [1,7,2]
myvar = pd.Series(a, index = ["x", "y", "z"])
print(myvar["x"])
```
> 1

- Bạn cũng có thể tạo sẵn từ điển và cho vào `Series`:
  
```python
import pandas as pd
calories = {"day1": 420, "day2": 380, "day3": 390}
myvar = pd.Series(calories)
print(myvar)
```

|---|-------|
| day1 |   420   |
| day2 |   380  |
| day3 |   390   |

> dtype: int64

- Bạn có thể chọn một số item trong từ điển, sử dụng `index`:

```python
import pandas as pd

calories = {"day1": 420, "day2": 380, "day3": 390}

myvar = pd.Series(calories, index = ["day1", "day2"])

print(myvar)
```

|---|-------|
| day1 |   420   |
| day2 |   380  |

> dtype: int64

### DataFrame trong Pandas

- Các bộ dữ liệu trong `Pandas` thường là các bảng đa chiều, được gọi là `DataFrames`. `Series` giống như một cột, còn `DataFrame` là toàn bộ bảng:

```python
import pandas as pd
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}
df = pd.DataFrame(data)
print(df)
```

|   | Calories | Duration |
|---|----------|----------|
| 0 |   420    |    50    |
| 1 |   380    |    40    |
| 2 |   390    |    45    |

- Dùng hàm `loc` để lấy ra hàng được chỉ định, ví dụ dưới lấy ra hàng đầu tiên:

```python
print(df.loc[0])
```

|---|-------|
| calories |   420   |
| duration |  50  |

- Giờ hãy thử lấy ra hai hàng đầu tiên:

```python
df.loc[[0,1]]
```

|   | Calories | Duration |
|---|----------|----------|
| 0 |   420    |    50    |
| 1 |   380    |    40    |

- Bạn có tạo ra `index` của riêng mình như code dưới:

```python
import pandas as pd
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}
df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
print(df) 
```

|   | Calories | Duration |
|---|----------|----------|
| day1 |   420    |    50    |
| day2 |   380    |    40    |
| day3  |   390    |    45    |

```python
print(df.loc["day1"])
```

|   | Calories | Duration |
|---|----------|----------|
| day1 |   420    |    50    |
| day2 |   380    |    40    |

- Bạn có thể load file chứa data bằng `pandas`, ví dụ file của bạn dưới dạng `.csv` thì bạn có thể dùng lệnh sau:

```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df) 
```

| Duration | Pulse | Maxpulse | Calories |
|----------|-------|----------|----------|
|    60    |  110  |   130    |  409.1   |
|    60    |  117  |   145    |  479.0   |
|    60    |  103  |   135    |  340.0   |
|    45    |  109  |   175    |  282.4   |
|    45    |  117  |   148    |  406.0   |
|    ...   |  ...  |   ...    |   ...    |
|    60    |  105  |   140    |  290.8   |
|    60    |  110  |   145    |  300.0   |
|    60    |  115  |   145    |  310.2   |
|    75    |  120  |   150    |  320.4   |
|    75    |  125  |   150    |  330.4   |

> 169 rows × 4 columns

- Bạn có thể tải file `data.csv` [tại đây](https://drive.google.com/file/d/1wiPA_F5pDlrCBz1zziw_yv7DTlLHzFeO/view?usp=drive_link). Như bạn thấy thì khi bảng dữ liệu quá lớn, `DataFrame` chỉ hiện 5 hàng đầu và 5 hàng cuối. Bạn có thể xem `DataFrame` hiển thị được tối đa bao nhiêu hàng bằng lệnh:

```python
import pandas as pd
print(pd.options.display.max_rows) 
```

> 60

- Vậy mặc định là nếu như bảng chỉ có 60 hàng trở xuôngs thì `DataFrame` sẽ hiển thị hết, còn nếu quá số hàng kia thì nó sẽ hiển thị một phần như trên. Bạn có thể thay đổi tham số này như dưới:

```python
import pandas as pd
pd.options.display.max_rows = 9999
df = pd.read_csv('data.csv')
print(df)
```

- Bạn có thể in ra 3 hàng đầu tiên hoặc bao nhiêu hàng cũng được bằng hàm `head`:

```python
print(df.head(3)) # nếu ko truyền số vào: df.head() sẽ lấy 5 hàng đầu tiên
```

| Duration | Pulse | Maxpulse | Calories |
|----------|-------|----------|----------|
|    60    |  110  |   130    |  409.1   |
|    60    |  117  |   145    |  479.0   |
|    60    |  103  |   135    |  340.0   |

- Để lấy các hàng cuối thì dùng hàm `tail`:

```python
print(df.tail(3)) # nếu ko truyền số vào: df.tail() sẽ lấy 5 hàng cuối
```

| Duration | Pulse | Maxpulse | Calories |
|----------|-------|----------|----------|
|    60    |  115  |   145    |  310.2   |
|    75    |  120  |   150    |  320.4   |
|    75    |  125  |   150    |  330.4   |

- Bạn có thể xem thông tin cơ bản của data bằng hàm `info`:

```python
print(df.info())
```

> <class 'pandas.core.frame.DataFrame'>
> RangeIndex: 169 entries, 0 to 168
> Data columns (total 4 columns):
>    Column    Non-Null Count  Dtype  
> ---  ------    --------------  -----  
> 0   Duration  169 non-null    int64  
> 1   Pulse     169 non-null    int64  
> 2   Maxpulse  169 non-null    int64  
> 3   Calories  164 non-null    float64
> dtypes: float64(1), int64(3)
> memory usage: 5.4 KB
> None

- Hãy để ý rằng hàm `info` nói cho chúng ta rất nhiều điều, như bạn thấy, data có 169 hàng và 4 cột, 3 cột đầu có kiểu dữ liệu là số nguyên, cột cuối có kiểu dữ liệu là số thực. Riêng cột `Calories` thì có 5 ô không có giá trị (164 non-null, tức chỉ có 164 ô có giá trị).

[tại đây](https://drive.google.com/file/d/1o-n8B5g-dVUXmKbtnMHdo_B_PbWugLEW/view?usp=drive_link)

### Làm sạch Data trong Pandas

- Làm sạch data có nghĩa là loại bỏ những data xấu trong data của bạn. Data xấu có thể là: ô bị trống, data không đúng format, data sai, data bị lặp,...
- Giờ hãy thử tạo data mới bằng lệnh dưới đây:

```python
import pandas as pd

# Dữ liệu
data = {
    "Duration": [60, 60, 60, 45, 45, 60, 60, 450, 30, 60, 60, 60, 60, 60, 60, 60, 60, 60, 45, 60, 45, 60, 45, 60, 45, 60, 60, 60, 60, 60, 60,60],
    "Date": ['2020/12/01', '2020/12/02', '2020/12/03', '2020/12/04', '2020/12/05', '2020/12/06', '2020/12/07', '2020/12/08', '2020/12/09', '2020/12/10', 
             '2020/12/11', '2020/12/12', '2020/12/12', '2020/12/13', '2020/12/14', '2020/12/15', '2020/12/16', '2020/12/17', '2020/12/18', '2020/12/19', 
             '2020/12/20', '2020/12/21', None, '2020/12/23', '2020/12/24', '2020/12/25', '2020/12/26', '2020/12/27', '2020/12/28', '2020/12/29', '2020/12/30', '2020/12/31'],
    "Pulse": [110, 117, 103, 109, 117, 102, 110, 104, 109, 98, 103, 100, 100, 106, 104, 98, 98, 100, 90, 103, 97, 108, 100, 130, 105, 102, 100, 92, 103, 100, 102,92],
    "Maxpulse": [130, 145, 135, 175, 148, 127, 136, 134, 133, 124, 147, 120, 120, 128, 132, 123, 120, 120, 112, 123, 125, 131, 119, 101, 132, 126, 120, 118, 132, 132, 129,115],
    "Calories": [409.1, 479.0, 340.0, 282.4, 406.0, 300.0, 374.0, 253.3, 195.1, 269.0, 329.3, 250.7, 250.7, 345.3, 379.3, 275.0, 215.2, 300.0, None, 323.0, 243.0, 364.2, 282.0, 300.0, 246.0, 334.5, 250.0, 241.0, None, 280.0, 380.3, 243.0]
}

df = pd.DataFrame(data)

df.head()
```

| Duration | Date       | Pulse | Maxpulse | Calories |
|----------|------------|-------|----------|----------|
| 60       | 2020/12/01 | 110   | 130      | 409.1    |
| 60       | 2020/12/02 | 117   | 145      | 479.0    |
| 60       | 2020/12/03 | 103   | 135      | 340.0    |
| 45       | 2020/12/04 | 109   | 175      | 282.4    |
| 45       | 2020/12/05 | 117   | 148      | 406.0    |

- Nếu bạn hiển thị hết bảng này thì sẽ thấy có một vài ô ghi là `NaN` (Not a Number) hoặc là `None`. Ở hàng 7 thì cột `Duration` là 450, có thể coi đây là data sai. Hàng 11 và 12 thì giống hệt nhau, đây là bị lặp data.

### Cách xử lý với trường hợp ô bị trống

- Ô bị trống có thể cho bạn kết quả sai khi bạn phân tích dữ liệu. Một cách để xử lý là bạn có thể loại bỏ luôn hàng chứa ô trống đấy. Data thực tế thường khá lớn nên cứ yên tâm, bạn có thể vứt bỏ một số hàng. Dùng lệnh `dropna` để thực hiện:

```python
new_df = df.dropna()
print(new_df)
```

- Hãy thử kiểm tra xem, bảng data của bạn bây giờ trông thật sạch. Chú ý một điều là hàm `dropna` sẽ tạo ra một `DataFrame` mới, nếu như bạn muốn thay đổi bảng ban đầu thì hãy dùng tham số `inplace = True`:

```python
df.dropna(inplace = True)
print(df)
```

- Một cách khác để xử lý ô trống đó là chúng ta tự điền một giá trị vào đó. Hàm `fillna()` sẽ giúp bạn làm điều đó:

```python
df.fillna(130, inplace = True)
print(df)
```

- Như bạn thấy, nó thay hết tất cả ô trống thành giá trị 130. Có thể đây không phải là điều ta muốn, ví dụ cột `Date` phai có giá trị format theo thời gian. Chúng ta có thể điền vào ô trống của cột được chỉ định:

```python
df["Calories"].fillna(130, inplace = True) # pandas 3.0 trở lên thì dùng df.fillna({"Calories":130}, inplace = True)
print(df)
```

- Như bạn thấy, cột `Calories` không còn ô trống nữa. Nhưng thường thì người ta sẽ điền các ô trống theo giá trị trung bình, giá trị trung vị hay mode. Tượng trưng cho các hàm `mean`, `median`, `mode`, hoặc dùng các hàm `max`, `min` để chèn vào giá trị lớn nhất, nhỏ nhất:

```python
x = df["Calories"].mean()
df["Calories"].fillna(x, inplace = True)
print(df)
``` 
### Cách xử lý nếu như data bị sai format

- Nếu cột `Date` của bạn có một vài ô bị sai format (kiểu 20201226), bạn có thể xem câu lệnh dưới đây:

```python
df['Date'] = pd.to_datetime(df['Date'])

print(df)
```

- Như bạn thấy. hàng 26 đã được fix như hàng 22 thì lại hiện `NaT`. Hãy dùng `dropna` để fix:

```python
df.dropna(subset=['Date'], inplace = True)
print(df)
```

### Cách xử lý với trường hợp data sai

- Đôi khi bạn có thể nhận biết ô nào có thể bị sai khi nhìn vào các ô còn lại. Và tất nhiên có một số ô bị sai nhưng rất khó nhận ra, bạn phải là chuyên gia phân tích dữ liệu thì mới nhìn thấy. Một cách xử lý là thay thành giá trị bạn nghĩ là đúng, như bạn thấy thì hàng 7 ở cột `Duration` có giá trị là 450, khả năng cao là sai, giờ hãy sửa lại:

```python
df.loc[7, 'Duration'] = 45 # thay thành giá trị 45
```

- Với bảng data lớn. bạn không thể thay thủ công từng ô được, hãy xem câu lệnh dưới đây thay tất cả ô có giá trị lớn hơn 120 thành 120 trong cột `Duration`:

```python
for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.loc[x, "Duration"] = 120
```

- Một cách khác là xóa luôn cột đó:

```python
for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.drop(x, inplace = True)
```

### Cách xử lý với data bị trùng

- Chúng ta có thể kiểm tra bảng data xem có bị trùng không bằng hàm `duplicated`:

```python
print(df.duplicated()) # True là bị trùng, False là ko bị
```

- Hãy dùng hàm `drop_duplicates` để xử lý data bị trùng, nó sẽ loại bỏ các hàng bị trùng:

```python
df.drop_duplicates(inplace = True)
print(df)
```

## 3. Tổng kết

Pandas là công cụ không thể thiếu trong việc xử lý dữ liệu, đặc biệt trong học máy và AI. Với khả năng thao tác dữ liệu bảng nhanh chóng và hiệu quả, Pandas giúp chuẩn bị dữ liệu cho các mô hình AI, nâng cao độ chính xác và hiệu quả trong quá trình phân tích. Thành thạo Pandas là bước đầu quan trọng để học và phát triển các ứng dụng AI.