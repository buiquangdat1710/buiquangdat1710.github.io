---
title: "Python Cơ Bản"
date: 2024-11-24 00:00:00  + 0800
categories: [Giáo Trình Dạy AI ProPTIT]
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

Python thuộc loại ngôn ngữ lập trình bậc cao. Python được công bố lần đầu tiên vào tháng 12 năm 1989. Tác giả của Python là **Guido van Rossum**, người Hà Lan, tại Centrum Wiskunde & Informatica (CWI).

## 1. Chương trình đầu tiên của bạn với Python

Chúng ta sẽ bắt đầu làm quen với lệnh đầu tiên của Python, đó là lệnh `print()`. Ví dụ lệnh sau sẽ in ra dòng chữ Hello World trên màn hình:
```python
print("Hello World")
```
> Hello World

#### Lệnh trợ giúp help()
Muốn xem trợ giúp cú pháp hay ý nghĩa của một lệnh bất kỳ của Python, chúng ta dùng lệnh `help()`. Có 2 cách thực hiện như sau:
- Cách 1: Thực hiện trực tiếp lệnh `help(<tên lệnh cần xem trợ giúp>)`. Ví dụ:

```python
help(print)
```
> Help on built-in function print in module builtins:
>
> print(...)
> 
>    print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)    
> 
>    Prints the values to a stream, or to sys.stdout by default.
> 
>    Optional keyword arguments:
> 
>    file:  a file-like object (stream); defaults to the current sys.stdout.
> 
>    sep:   string inserted between values, default a space.
> 
>    end:   string appended after the last value, default a newline.
> 
>    flush: whether to forcibly flush the stream.

- Cách 2: Gõ lệnh `help()`. Khi đó con trỏ trợ giúp help xuất hiện. Chúng ta có thể gõ lệnh muốn xem trợ giúp ngay sau con trỏ này, ví dụ:

```python
help()
```
> Welcome to Python 3.10's help utility!
>
>If this is your first time using Python, you should definitely check out
>
>the tutorial on the internet at https://docs.python.org/3.10/tutorial/.
>
>Enter the name of any module, keyword, or topic to get help on writing
>
>Python programs and using Python modules.  To quit this help utility and
>
>return to the interpreter, just type "quit".
>
>To get a list of available modules, keywords, symbols, or topics, type
>
>"modules", "keywords", "symbols", or "topics".  Each module also comes
>
>with a one-line summary of what it does; to list the modules whose name
>
>or summary contain a given string such as "spam", type "modules spam".
>
>help> 

Muốn thoát khỏi chế độ con trỏ trợ giúp help> chỉ cần bấm Enter. NHư ta có thể thấy ở trên, cú pháp đầy đủ của lệnh print là:

`print(value1, value2,..., sep = <ký tự ngăn cách>, end = <kí tự kết thức>, file = <đối tượng tệp ra>, flush = true/false)`

Trong đó:
- value1, value2,... là các giá trị mà ta muốn in ra, có thể là số hoặc xâu.
- sep là kí tự ngăn cách giữa các value1, value2,..., mặc định là giá trị này là dấu cách, tức sep = " ".
- end = kí tự kết thúc của câu, mặc định là kí tự xuống dòng, tức end = "\n".
- Các tham biến file và flush chúng ta sẽ tìm hiểu sau.

Ví dụ về in ra ngày tháng đúng format chuẩn:

```python
print(17,10,2004, sep = "/", end = "~~")
```
> 17/10/2004~~

#### Ngăn cách lệnh
Python là chương trình viết mỗi lệnh trên một dòng nhưng bạn có thể gõ nhiều lệnh trên một dòng bằng dấu ngăn cách ";".
```python
print("Hà");print("Nội")
```
> Hà
> 
> Nội

Thông thường chương trình Python sẽ viết mỗi lệnh trên một dòng để dễ debug.

#### Comment trong Python

Comment trong Python được bắt đầu bằng ký tự # nhằm note lại các câu lệnh:

```python
x = 2 # gán 2 cho biến x
print(x) # In ra x
```

#### Nhiều lệnh trên nhiều dòng
Chúng ta có thể viết một lệnh dài trên nhiều dòng. Muốn xuống dòng cần gõ ký hiệu "\\" sau đó có thể nhấn Enter để xuống dòng viết tiếp dòng lệnh đó:
```python
a = "Hà Nội là thủ đô \
của Việt Nam"
b = 1 + 2/3 + 2.004 + \
23.6 - 12/3
print(b)
```
## 2. Kiểu dữ liệu trong Python

#### Lệnh type() và kiểu dữ liệu chính trong Python
Lệnh `type(<data>)` sẽ trả lại kiểu dữ liệu hiện thời của \<data\>. Lấy ví dụ các câu lệnh sau đây:

```python
print(type(2.4))
print(type("Hanoi"))
print(type(3>2))
print(type(123))
```
> <class 'float'>
> 
> <class 'str'>
> 
> <class 'bool'>
> 
> <class 'int'>

Các kiểu dữ liệu cơ bản trong Python bao gồm:
- `int` : số nguyên.
- `float` : số thực.
- `bool` : logic với 2 giá trị **True** và **False**.
- `str` : xâu ký tự.

#### Biến nhớ không cần khai báo và xác định kiểu dữ liệu
Trong Python, các biến nhớ chỉ xác định kiểu dữ liệu tại thời điểm gán giá trị, do đó biến nhớ không cần khai báo kiểu dữ liệu trước. Một biến nhớ có thể nhận nhiều kiểu giá trị khác nhau mà không cần khai báo lại. Xem đoạn chương trình sau, biến nhớ x có thể nhận các giá trị với kiểu dữ liệu lần lươợ là `float`, `int`, `bool`, `str`.
```python
x = 1.23
print(type(x))
x = 2
print(type(x))
x = x < 1
print(type(x))
x = "Hello"
print(type(x))
```
> <class 'float'>
> 
> <class 'int'>
> 
> <class 'bool'>
> 
> <class 'str'>

## 3. Biểu thức trong Python
#### Biểu thức số học
Biểu thức số học trong Python có thể được viết và tính dễ dàng như trong mọi ngôn ngữ lập trình bậc cao khác. Biểu thức có thể chứa giá trị số hoặc biến nhớ, hàm số có giá trị. Xét các ví dụ sau:
```python
15 + 2 # phép cộng
15 - 4 # phép trừ
2 * 5 # phép nhân
2 ** 5 # phép mũ
5 / 2 # phép chia thực
5 // 2 # phép chia nguyên
5 % 2 # phép chia dư
2 ** 3 ** 2 # số mũ tính từ phải sang trái
```
#### Quy tắc đặt tên biến
Biến nhớ trong Python đuợc quy dịnh đặt tên cũng tương tự như các ngôn ngữ lập trình khác. Có thể tóm tắt quy tắc đặt tên biến nhớ trong Python như sau:
- Phân biệt chữ hoa và chữ thường.
- Chỉ có thể chứa ký tự là chữ, chữ số, dấu gạch dưới và không cho phép chứa bất kỳ các ký tự khác.
- Tên biến nhớ phải bắt đầu bằng chữ cái hoặc dấu _ , không được phép chữ số.
Ví dụ các tên biến sau là hợp lệ: ab_1 , _bien_nho , xyz , x_y_z.

#### Lệnh gán
Lệnh gán đon giản nhất của Python có dạng:
`<biến nhớ> = <giá trị>`

Ví dụ:
```python
x = 1
y = 10
z = y + 5
x,y,z
```
> (1, 10, 15)

#### Gán đồng thời
Nếu các biến nhớ có giá trị bằng nhau thì có thể gán bằng một lệnh như sau:
```python
x = y = z = 12
```
Việc gán đồng thời nhiều biến nhớ với các giá trị khác nhau có thể thực hiện như sau:
```python
x,y,z = 1,2,3
```
> ☠️ **Chú ý**: Không được viết như sau, ví dụ: x = 1, y = 2, z = 3

Nếu ta muốn đổi giá trị hai biến nhớ, ta có thể làm như sau:
```python
x = 1;y = 7
x,y = y,x
print(x,y)
```
> 7 1

Từ khóa là các từ mà Python đã dùng cho các mục đích riêng của ngôn ngữ hoặc dành cho các hàm hệ thống. Vì vậy khi đặt tên biến nhớ trong Python cần tránh đặt trùng với các từ khóa này.
Danh sách từ khóa của Python được cập nhật và mở rộng theo các lần phát triển của mình. Bảng dưới là các từ khóa của Python 3.

![anh](./image/42.png)

Chú ý riêng cho biến nhớ "_": Ký tự _ (underscore) sẽ đuợc hiểu là biến nhớ đặc biệt. Biến nhớ này không cần gán mà mặc định luôn được gán giá trị của biểu thức được tính gần đây nhất:
```python
12 + 7
```
> 19

```python
_
```
> 19

## 4. Input dữ liệu
Hàm `input()` dùng để thực hiện việc nhập dữ liệu từ bàn phím, dữ liệu được nhập chính là giá trị trả lại của hàm số này. Hàm `input()` có cú pháp như sau:

`<biến nhớ> = input(<dòng chữ hiện trên màn hình>)`

Khi thực hiện lệnh, <dòng chữ hiện trên màn hình> sẽ xuất hiện, người dùng nhập một xâu ký tự bất kỳ, kết quả của hàm sẽ lưu vào <biến nhớ>
Xét ví dụ đơn giản sau:
```python
Name = input("Nhập tên: ")
```
> Nhập tên: Đạt
> 
> Đạt

Chú ý rằng hàm `input()` luôn trả lại dữ liệu kiểu xâu ký tự, mặc dù trên màn hình dữ liệu thể hiện số như bình thường
## 5. Chuyển đổi trực tiếp dữ liệu
Để có thể nhập được chính xác kiểu dữ liệu muốn nhập, chúng ta sẽ sử dụng các hàm chuyển đổi trực tiếp dữ liệu đơn giản của Python. Đó là các hàm `int()`, `float()` và `str()`.
#### Hàm int()
Hàm `int()` có cú pháp là:
`<giá trị trả về> = int(<giá trị>)`
Hàm này sẽ trả lại giá trị là số nguyên chuyển đổi của tham biến <giá trị>. Xem các ví dụ bên dưới
```python
print(int("23"))
print(int(3.6))
```
> 23
>
> 3

#### Hàm float()

Hàm `float()` có cú pháp là:

`<giá trị trả về> = float(<giá trị>)`

Hàm này sẽ trả lại giá trị là số thực chuyển đổi của tham biến <giá trị>. Ví dụ:

```python
print(float("3.14"))
print(float(5))
```
> 3.14
> 
> 5.0

#### Hàm str()

Hàm `str()` có chức năng chuyển đổi một giá trị hoặc một biểu thức số sang kiểu xâu ký tự. Cú pháp tương tự như 2 hàm `int()` và `float()`:

`<giá trị trả về> = str(<biểu thức>)`

Ví dụ:

```python**
print(str(5))
print(str(3 + 4**5 - 17.8**2))
```
> '5'
> 
> '710.16'

#### Hàm eval()

Hàm số  `eval()` có tác dụng tính toán các biểu thức bên trong một xâu ký tự và chuyển đổi chúng về dạng tương ứng. Cú pháp của hàm `eval()`:

`<giá trị trả về> = eval("<biểu thức>")`

Xét một vài ví dụ sau:

```python
print(eval("12 + 7))
print(eval("24**2 - 15))
```
> 19
>
> 561

Ví dụ chương trình sau yêu cầu người dùng nhập một biểu thức toán học từ bàn phím, sau khi nhập, chương trình sẽ tính toán và đưa ra ngay kết quả:
```python
x = eval(input("Nhập biểu thức toán học: "))
print("Kết quả: ", x)
```
> Nhập biểu thức toán học: 1 + 2 + 3
> 
> Kết quả:  6

Một tính năng đặc biệt là hàm `eval()` có thể nhận biết đồng thời hiều biểu thức trên một hàng, cách nhau bởi dấu phẩy:

```python
print(eval("1+2, 3*4, 5//2"))
```
> (3, 12, 2)

Ví dụ nhập nhiều số tự nhiên trên một dòng từ bàn phím, mỗi số cách nhau bằng dấu phẩy:

```python
m,n,p = eval(input("Nhập 3 số cách nhau bởi dấu phẩy: "))
print(m,n,p)
```
> Nhập 3 số cách nhau bởi dấu phẩy: 1,2,3
>
>1 2 3

#### Hàm split()

Lệnh split() trong Python là một phương thức của kiểu dữ liệu chuỗi (str) được sử dụng để tách một chuỗi thành một danh sách các phần tử con dựa trên một ký tự hoặc chuỗi ký tự được chỉ định. Cú pháp:

`str.split([separator], [maxsplit])`

Các tham số:
- `separator`: là kí tự hoặc chuỗi kí tự dùng để tách chuỗi ban đầu. Nếu không chỉ định, mặc định là dấu cách. Nếu chuỗi có nhiều dấu cách liên tiếp và không chỉ định `separator`, `split()` sẽ tự động bỏ qua các dấu cách dư thừa.
- `maxsplit`: là số nguyên xác định số lần tách tối đa. Nếu không chỉ định, phương thức sẽ tách chuỗi cho đến khi không còn gì để tách.

Ví dụ:
```python
text = "Python là ngôn ngữ lập trình"
words = text.split()
print(words)
```

> ['Python', 'là', 'ngôn', 'ngữ', 'lập', 'trình']

```python
text = "Python,là,ngôn,ngữ,lập,trình"
words = text.split(',')
print(words)
```

> ['Python', 'là', 'ngôn', 'ngữ', 'lập', 'trình']

```python
text = "Python là ngôn ngữ lập trình"
words = text.split(' ', 2)  # Tách tối đa 2 lần
print(words)

```

> ['Python', 'là', 'ngôn ngữ lập trình']

#### Hàm map()
Hàm `map()` trong Python được sử dụng để áp dụng một hàm lên từng phần tử của một đối tượng có thể lặp lại (chẳng hạn danh sách, tuple, v.v.) và trả về một đối tượng map, có thể được chuyển đổi thành danh sách, tuple, hoặc bất kỳ kiểu dữ liệu lặp lại nào khác. Cú pháp:

`map(function, iterable)`

Tham số:
- `function`: là hàm sẽ được áp dụng lên từng phần tử của `iterable`. Có thể là một hàm đã định nghĩa, hàm ẩn danh (lambda), hoặc hàm tích hợp sẵn.
- `iterable`: là đối tượng có thể lặp lại, như danh sách, tuple, chuỗi,...

Ví dụ:

```python
numbers = ["1", "2", "3", "4"]
integers = list(map(int, numbers))  # Áp dụng hàm int lên từng phần tử
print(integers)
```

> [1, 2, 3, 4]

```python
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, numbers))  # Áp dụng hàm lambda
print(squared)
```

> [1, 4, 9, 16]

```python
numbers = ["1", "2", "3"]
a,b,c = map(int, numbers)  # Áp dụng hàm int lên từng phần tử
print(a,b,c)

```

> 1 2 3

Để nhập 3 số nguyên cách nhau bằng dấu cách trong Python, bạn có thể sử dụng hàm input() và kết hợp với phương thức split() để tách chuỗi đầu vào. Sau đó, chuyển đổi từng phần tử thành số nguyên bằng map() hoặc int(). Dưới đây là một ví dụ:

```python
# Nhập 3 số nguyên cách nhau bằng dấu cách
a, b, c = map(int, input("Nhập 3 số nguyên cách nhau bằng dấu cách: ").split())

print(f"Số thứ nhất: {a}")
print(f"Số thứ hai: {b}")
print(f"Số thứ ba: {c}")

```

> Nhập 3 số nguyên cách nhau bằng dấu cách: 1 2 3
> 
> Số thứ nhất: 1
> 
> Số thứ hai: 2
> 
> Số thứ ba: 3

## 6. Câu lệnh if
Cấu trúc lệnh if:
```python
if <điều kiện kiểm tra>:
    <các lệnh của nhóm if>
```
Lệnh trên sẽ thực hiện như sau: Chương trình sẽ kiểm tra <điều kiện> sau từ khóa **if**, nếu điều kiện đúng thì nhóm <các lệnh của nhóm if> sẽ thực hiên. Nếu điều kiênệsai thì lệnh sẽ được bỏ qua.
Cấu trúc lệnh **if**, cú pháp đấy đủ:

```python
if <điều kiện kiểm tra>:
    <các lệnh của nhóm if>
elif:
    <các lệnh của nhóm elif>
else:
    <các lệnh của nhóm else>
```
Dưới đây là chương trình so sánh hai số:

```python
a,b = map(int, input("Nhập hai số: ").split())
if a == b:
    print("Hai số bằng nhau")
elif a > b:
    print("a lớn hơn b")
else:
    print("a nhỏ hơn b")
```

> Nhập hai số: 2 3
> 
> a nhỏ hơn b

## 7. Lệnh lặp for
Xem một chương trình nhỏ dưới đây:
```python
for i in range(1,5):
    print(i)
```
> 1 2 3 4 

Trong Python, chúng ta sử dụng vòng for kết hợp với hàm `range`, hàm `range` sẽ tạo ra một dãy giá trị có thể đếm được trong python. Cú pháp tổng quát của hàm `range`:
```python
range(start, end, step)
```
Trong đó start là điểm bắt đầu, end là điểm kết thức và step chính là bước nhảy. Lưu ý là hàm trên sẽ chỉ bắt đầu từ start và kết thúc ở end - 1. Dưới đây là một số ví dụ:
```python
range(1,6) # 1, 2, 3, 4, 5
range(2,7,2) # 2, 4, 6
range(10,1,-1) # 10, 9, 8, 7, 6, 5, 4, 3, 2
```

Như bạn đã thấy, phần step có thể âm, và start có thể lớn hơn end. Nếu step âm thì hàm `range` sẽ biểu thị dãy số giảm dần. Bạn có thể nghịch hàm `range` này bằng cách thay nhiều số để hiểu cách nó hoạt động. Ví dụ dưới đây sẽ đếm số ước của một số tự nhiên:

```python
n = int(input("Nhập số: "))
count = 0
for i in range(1,n+1):
    if(n%i==0):
        count += 1
print(count)
```
> Nhập số: 10
>
> 4

## 8. Lệnh lặp while
Một lệnh nữa cần đưa ra ở đây là lệnh lặp theo điều kiện `while`. Khác với lệnh `for` là lặp theo số lần lặp cố định, lệnh `while` có số vòng lặp không cố định mà phụ thuộc vào điều kiện dừng của lệnh. Cú pháp đơn giản như sau:
```python
while <điều kiện>:
    <các lệnh lặp>
```
Lệnh này sẽ chạy liên tục \<các lệnh lặp\> trong thân của `while` và chỉ dừng theo \<điều kiện\>. Mỗi lần chạy \<các lệnh lặp\>, chương trình sẽ kiểm tra \<điều kiện\>, nếu đúng thì chạy tiếp, nếu sai thì dừng lại, kết thúc lệnh. Ví dụ dưới là tìm UCLN của 2 số nguyên:

```python
a = int(input())
b = int(input())
while a != b:
    if a > b:
        a -= b
    else:
        b -= a
print(a) # hoặc print(b)
```
> 56
> 
> 24
> 
> 8

## 9. Thư viện Math
Thư viện Math trong Python cung cấp các hàm toán học cơ bản và nâng cao. Để khai báo thư viện Math, chúng ta dùng câu lệnh bên dưới:

```python
import math
```

Dưới đây là một số hàm trong thư viện Math hay dùng:

```python
import math

math.sqrt(x) # trả về căn bậc 2 của x

math.min(a,b) # trả về số nhỏ nhất trong hai số a và b

math.min(a,b,c,...) # trả về số nhỏ nhất trong nhiều số

math.max(a,b) # trả về số lớn nhất trong hai số a và b

math.max(a,b,c,...) # trả về số lớn nhất trong nhiều số

math.abs(x) # trả về giá trị tuyệt đối của x

math.pow(x,y) # trả về x mũ y (x**y)

math.factorial(x) # tính giai thừa của x

math.ceil(x) # làm tròn lên số nguyên gần nhất

math.floor(x) # làm tròn xuống số nguyên gần nhất

math.trunc(x) # cắt bỏ phần thập phân (lấy phần nguyên)

math.hypot(x,y) # tính độ dài của vector có tọa độ (x,y), tương đương với căn(x^2 + y^2)

math.pi # giá trị của số pi

math.e # giá trị của số e

math.inf # giá trị vô cực

math.nan # giá trị không xác định (not a number)

math.gcd(a,b) # ước chung lớn nhất của a và b
```

## 10. Hàm số

Trong Python hỗ trợ dịnh nghĩa cả 2 kiểu hàm số: hàm không có giá trị (hàm `void`) và hàm có trả lại giá trị. Trước tiên chúng ta làm quen vơiớcách tạo các hàm không trả lại giá trị. Các hàm này thường được gọi là hàm `void`. Các hàm `void` còn được gọi là **thủ tục** trong một số ngôn ngữ lập trình khác.

#### Định nghiĩ hàm void (thủ tục)

Định nghĩa hàm void theo mẫu như sau:

```python
def <tên hàm số> (<tham số>):
    <các lệnh mô tả hàm số>
```

Chú ý:
- Tên hàm số cần tuân thủ quy luật đặt tên trong Python.
- Phần thân của hàm số được viết thụt vào theo đúng quy định Python. Định nghĩa hàm số phải kết thúc bằng ký hiệu ":".
- Trong thân của hàm số có thể có hoặc không có lệnh `return`. Lệnh `return` có tác dụng kết thuc ngay hàm số.
- Hàm số có thể không có hoặc có các tham số. Có thể có nhiều tham số, viết cách nhau bởi dấu phẩy. 

Sau đây là một ví dụ đơn giản:

```python
def Welcome():
    print("Xin chào !")
Welcome()
```

> Xin chào !

#### Định nghĩa hàm số có giá trị
Định nghĩa hàm số có giá trị

```python
def <tên hàm số> (<tham số>):
    <các lệnh ...>
    return <giá trị>
    <các lệnh ...>
```

Chú ý:
- Trong phần thân của hàm số bắt buộc phải có lệnh `return <giá trị>`, lệnh này có tác dụng dừng hàm số và trả lại giá trị cho hàm số.

Dưới đây là ví dụ đơn giản:

```python
def luythua(x,n,m):
    return (x**n)**m

luythua(2,2,3)
```
> 64

Truyền biến vào hàm trong Python là truyền theo kiểu tham trị, tức là các dòng lệnh trong hàm số chỉ có tác dụng trong hàm đấy. Ví dụ:

```python
def tang(n):
  n += 1
  print(n)
  return n

n = int(input("Nhập số: "))
tang(n)
print(n)
```

> Nhập số: 5
> 
> 6
> 
> 5

#### Hàm lambda

Hàm `lambda` trong Python là một hàm **ẩn danh** (anonymous function), tức là một hàm không có tên, thường được dùng để định nghĩa nhanh các hàm đơn giản trong một dòng. Hàm `lambda` bắt buộc phải trả về giá trị bằng một biểu thức tính toán. Hàm có thể có một hoặc nhiều tham số. Cú pháp hàm `lambda`:

```python
<tên hàm> = lambda <tham số>: <biểu thức tính>
```

Ví dụ:

```python
square = lambda x: x ** 2
print(square(5)) 
```

> 25

```python
add = lambda a, b: a + b
print(add(3, 7))  # 10
```

> 10

Khi nào nên dùng hàm `lambda`:
- Khi cần định nghĩa hàm nhanh, ngắn gọn.
- Khi không cần tái sử dụng hàm trong nhiều nơi khác.
- Thường dùng trong các hàm xử lý dữ liệu hoặc callback như `map()`, `filter()`,`sorted()`.

Hạn chế:
- Không thể chứa nhiều dòng lệnh.
- Khó đọc hơn so với các hàm định nghĩa thông thường bằng `def` khi code phức tạp.

## 11. Kiểu dữ liệu List
**List** (danh sách) là một trong các kiểu dữ liệu quan trọng nhất của Python. Khác với hầu hết các ngôn ngữ lập trình bậc cao khác, **list** trong Python có rất nhiều tính chất đặc biệt hay và thú vị. Song hành với list, trong Python coòncó 2 kiểu dữ liệu tương tụựkhác là **tuple** và **range**. 
**List** được hiểu là một dãy các giá trị được viết trong dấu ngoặc vuông, việc khởi tạo một list hay danh sách rất đơn giản, ví dụ như sau:

```python
a = [10,4,9,8]
print(type(a))
```
> <class list>

Giống như các ngôn ngữ lập trình khác, có thể truy cập đến từng phần tử của list. Các phần tử của danh sách sẽ được đánh số từ 0. Ví dụ:

```python
print(a[10],a[3])
```
> 10 8

Cách truy cập phần tử của danh sách a với chỉ số k như sau: **a[k]**
Chú ý quan trọng: Các phần tử của một danh sách (list) có thể có các kiểu dữ liệu khác nhau. Ví dụ:

```python
a = [2, "Việt Nam", 3.14]
print(a[0])
```

> 2

#### Các phép toán và phương thức của List
Để tính độ dài một danh sách chúng ta dùng hàm số len() được thiết lập sẵn trong Python:
```python
a = [1,2,3,4,5]
print(len(a))
```
> 5

#### Phép + và * danh sách
Phép + sẽ dùng để ghép nối tự nhiên 2 danh sách:
```python
a = [1,2,3]
b = [4,5,6,7,8,9,10]
c = a + b
print(c)
```

> [1,2,3,4,5,6,7,8,9,10]

Phép * với số tự nhiên chính là phép cộng nhiều lần một danh sách. Ví dụ:

```python
a = [1,2,3]
a*3
```

> [1,2,3,1,2,3,1,2,3]

#### Xóa một phần tử của danh sách
Muốn xóa phần tử thứ k của danh sách chúng ta dùng lệnh **del**:
```python
a = [1,2,3]
del a[1]
print(a)
```
> [1, 3]

#### Các phương thức của List

Chúng ta có thể dùng hàm `dir()` để hiện thị các phương thức hiện có của `List`:

```python
dir(list)
```

> ['__add__',
> 
> '__class__',
> 
> '__class_getitem__',
> 
> '__contains__',
> 
> '__delattr__',
> 
> '__delitem__',
> 
> '__dir__',
> 
> '__doc__',
> 
> '__eq__',
> 
> '__format__',
> 
> '__ge__',
> 
> '__getattribute__',
> 
> '__getitem__',
> 
> '__gt__',
> 
> '__hash__',
> 
> '__iadd__',
> 
> '__imul__',
> 
> '__init__',
> 
> '__init_subclass__',
> 
> '__iter__',
> 
> '__le__',
> 
> '__len__',
> 
> '__lt__',
> 
> '__mul__',
> 
> '__ne__',
> 
> '__new__',
>
> '__reduce__',
> 
> '__reduce_ex__',
> 
> '__repr__',
> 
> '__reversed__',
> 
> '__rmul__',
> 
> '__setattr__',
> 
> '__setitem__',
> 
> '__sizeof__',
> 
> '__str__',
> 
> '__subclasshook__',
> 
> 'append',
> 
> 'clear',
> 
> 'copy',
> 
> 'count',
> 
> 'extend',
> 
> 'index',
> 
> 'insert',
> 
> 'pop',
> 
> 'remove',
> 
> 'reverse',
> 
> 'sort']

Dưới đây là một số hàm hay sử dụng:

```python
a.append(<đối tượng>) # Bổ sung <đối tượng> vào cuối của danh sách như một phần tử 
a.clear() # Xóa dữ liệu, đưa danh sách a thành rỗng
b = a.copy() # Trả về một danh sách khác có giá trị giống a (một bản sao của a).
a.extend(<đối tượng>) # Mở rộng, bổ sung <đối tượng> vào cuối của danh sách a như một mở rộng tự nhiên
k = a.index(<giá trị>) # Trả về chỉ số index đầu tiên trong danh sách của giá trị
a.insert(index, <đối tượng>) # Chèn <đối tượng> vào danh sách ở trước vị trí index
x = a.pop() # Xóa và lấy ra phần tử cuối cùng của danh sách a
a.remove(<giá trị>) # Xóa đi phần tử đầu tiên của a có giá trị = <giá trị>
a.reverse() # Sắp xếp theo thứ tự ngược lại
a.sort() # Sắp xếp danh sách
a.count(<giá trị>) # đếm xem có bao nhiêu phần tử = giá trị trong list
```

Dưới đây là một số đoạn mã minh họa cách sử dụng các phương thức mà bạn đã liệt kê:

```python
# 1. Sử dụng append()
a = [1, 2, 3]
a.append(4)  # Bổ sung 4 vào cuối danh sách
print(a)  # Output: [1, 2, 3, 4]

# 2. Sử dụng clear()
a.clear()  # Xóa dữ liệu trong danh sách a
print(a)  # Output: []

# 3. Sử dụng copy()
a = [1, 2, 3]
b = a.copy()  # Tạo bản sao của a
print(b)  # Output: [1, 2, 3]

# 4. Sử dụng extend()
a.extend([4, 5])  # Mở rộng danh sách bằng cách thêm [4, 5]
print(a)  # Output: [1, 2, 3, 4, 5]

# 5. Sử dụng index()
k = a.index(3)  # Tìm chỉ số của giá trị 3
print(k)  # Output: 2

# 6. Sử dụng insert()
a.insert(2, 99)  # Chèn 99 vào trước vị trí index 2
print(a)  # Output: [1, 2, 99, 3, 4, 5]

# 7. Sử dụng pop()
x = a.pop()  # Lấy và xóa phần tử cuối cùng
print(x)  # Output: 5
print(a)  # Output: [1, 2, 99, 3, 4]

# 8. Sử dụng remove()
a.remove(99)  # Xóa phần tử đầu tiên có giá trị 99
print(a)  # Output: [1, 2, 3, 4]

# 9. Sử dụng reverse()
a.reverse()  # Sắp xếp danh sách theo thứ tự ngược lại
print(a)  # Output: [4, 3, 2, 1]

# 10. Sử dụng sort()
a.sort()  # Sắp xếp danh sách theo thứ tự tăng dần
print(a)  # Output: [1, 2, 3, 4]

numbers = [1, 2, 3, 4, 2, 2, 5]
print(numbers.count(2))  # Output: 3

```

#### Lệnh for trên dữ liệu List
Chúng ta có thể sử dụng vòng for kết hợp với kiểu dữ liệu list. Dưới đây là một ví dụ đơn giản:
```python
a = [1,"Viet Nam",2.5,5]
for item in a:
    print(item, end = " ")

```

> 1 Viet Nam 2.5 5 

Hoặc ta có thể viết một chương trình tương tự như sau:

```python
a = [1,"Viet Nam",2.5,5]
for i in range(len(a)):
    print(a[i], end = " ")
```

> 1 Viet Nam 2.5 5 

#### Chỉ số của List
Chỉ số của List, cũng như một dữ liệu tuần tự bất kỳ sẽ được đánh số từ 0. Chúng ta có thể truy cập trực tiếp dến từng phần tử của danh sách:

```python
a = [1, 5, 7, 9, 3]
print(a[0])
print(a[2:5])
for i in a[2:5]:
  print(i, end = " ")
```

> 1
> 
> [7, 9, 3]
> 
> 7 9 3

Như bạn thấy ví dụ ở trên, chúng ta có thể dùng cú pháp sau để lựa chọn nhiều giá trị:

```python
a[start:end] # lấy ra list con chứa các phần tử từ a[start] đến a[end-1]
```

Ngoài ra chúng ta cũng có thể dùng chỉ số âm như ví dụ dưới, điều này làm nên sự đặc biệt trong Python:

```python
a = [1,5,7,9,3]
print(a[-1]) # lấy ra phần tử cuối cùng
print(a[-3:-1]) # lấy ra phần tử a[-3], a[-2], có thể hiểu là lấy ra phần tử a[2], a[3]
print(a[:3]) # Nếu không có biến start thì mặc định start = 0 
print(a[:]) # in ra hết tất cả phần tử (start = 0, end = len(a))
print(a[2:]) # in ra từ phần tử a[2] đến hết
```

> 3
> 
> [7, 9]
>
> [1, 5, 7]
> 
> [1, 5, 7, 9, 3]
> 
> [7, 9, 3]

#### Hàm range kết hợp với List
Ví dụ sau sẽ tạo ra một danh sách các số chẵn không âm và nhỏ hơn n:

```python
n = int(input())
list(range(0,n,2))
```
> 10
> 
> [0, 2, 4, 6, 8]

Muốn thiết lập danh sách (list) từ kết quả của hàm `range()` chúng ta sử dụng hàm `list()`, chúng ta có thể sử dụng bước nhảy âm như ví dự dưới đây:

```python
list(range(20,0,-3))
```
> [20, 17, 14, 11, 8, 5, 2]

Một điều đặc biệt của List là chúng ta có thể thay đổi giá trị của từng phần tử trong danh sách:

```python
a = ["Việt Nam", "Đức", "Úc"]
a[0] = "Nhật"
a[1] = [1,2,3]
print(a) 
```

> ['Nhật', [1, 2, 3], 'Úc']

## 12. Kiểu dữ liệu Tuple
Kiểu dữ liệu `tuple` trong Python dược gọi là dãy số. Kiểu dữ liệu này rất giống với `list`. Chỉ khác là `list` dùng [], còn `tuple` dùng (). Dưới đây là ví dụ:
```python
a = (1, 2, 3)
b = (12, 3.14, "one", "two", "three")
print(type(a))
print(b)
```

> <class 'tuple'>
> 
> (12, 3.14, 'one', 'two', 'three')

Khác với List, Tuple là kiểu dữ liệu không thay đổi được. Hãy xem ví du bên dưới:

```python
a = (1, 2, 3)
a[0] = 10
a
```

> TypeError                                 Traceback (most recent call last)
>
> a = (1, 2, 3)
> 
> a[0] = 10
>
> a
>
>TypeError: 'tuple' object does not support item assignment

Chúng ta có thể xem các phương thức của `tuple` bằng lệnh `dir(tuple)` và nếu bạn để ý, `tuple` chỉ có hai phương thức là `count()` và `index()`. Hai phương thức này giống hệt hai phương thức `count()` và `index()` trogn `list` nên tôi sẽ không nhắc lại.

#### So sánh List và Tuple

Dưới đây bảng so sánh kiểu dữ liệu `list` và `tuple`:


| **Đặc điểm**               | **`list`**                                | **`tuple`**                              |
|----------------------------|-------------------------------------------|------------------------------------------|
| **Có thể thay đổi**        | Có (mutable): Bạn có thể thêm, sửa, xóa. | Không (immutable): Không thể thay đổi.   |
| **Tốc độ**                 | Chậm hơn do hỗ trợ thay đổi dữ liệu.      | Nhanh hơn nhờ tính bất biến.             |
| **Bộ nhớ**                 | Sử dụng nhiều bộ nhớ hơn.                 | Sử dụng ít bộ nhớ hơn.                   |
| **Ứng dụng**               | Thích hợp cho dữ liệu cần thay đổi thường xuyên. | Dùng cho dữ liệu cố định, không thay đổi.|
| **Phương thức hỗ trợ**     | Cung cấp nhiều phương thức như: `append()`, `remove()`, `sort()`, ... | Hỗ trợ ít phương thức hơn.               |



Khi nào nên dùng `list` ?
  - Khi bạn cần thêm, sửa, hoặc xóa dữ liệu trong quá trình thực thi chương trình.
  - Khi kích thước của tập hợp hoặc nội dung có thể thay đổi theo thời gian.
  - Khi cần sử dụng các phương thức của list như append(), remove(), hoặc sort().

Ví dụ:
```python
# Một danh sách để lưu các công việc có thể thay đổi
tasks = ["Learn Python", "Write code", "Debug"]
tasks.append("Test program")  # Thêm công việc mới
tasks.remove("Write code")    # Xóa một công việc
print(tasks)  # Output: ['Learn Python', 'Debug', 'Test program']
```

Khi nào nên dùng `tuple` ?
  - Khi dữ liệu là bất biến (không cần thay đổi), ví dụ như hằng số hoặc cài đặt cấu hình.
  - Khi cần hiệu suất cao hơn (do tuple nhanh hơn và tiết kiệm bộ nhớ hơn).
  - Khi dữ liệu cần dùng làm key trong từ điển (dict) hoặc trong một tập hợp (set) vì tuple là kiểu dữ liệu hashable, còn list thì không.

Ví dụ:

```python
# Tọa độ điểm trong không gian (bất biến)
point = (3, 4, 5)

# Lưu cấu hình cài đặt
config = ("localhost", 8080, "https")
print(config[0])  # Output: localhost

```

## 13. Kiểu dữ liệu String
Tóm tắt một số đặc tính của kiểu dữ liệu string:
  - Dữ liệu string được khởi tạo bằng cách gán xâu ký tự nằm giữa hai dấu nháy đơn 'one two three' hoặc nháy kép "one two three"
  - Tên kiểu dữ liệu: `str`
  - Kiểu dữ liệu string thuộc dạng dữ liệu tuần tự, do dó có thể thực hiện các hàm như len(), truy cập chỉ số, ví du s[i], hoặc vùng chỉ số, ví dụ s[i:j].
  - Kiểu dữ liệu string không cho phép thay đổi từng ký tự của mình.
  - Có thê thực hiện phép toán + xâu ký tự và phép toán * với một số tự nhiên.
  - Trong Python không có kiểu dữ liệu ký tự, một ký tự cũng thuộc kiểu string.

Xem đoạn chương trình sau để hiểu rõ các tính chất trên:

```python
s = "Hà Nội Việt Nam "
print(s)
print(type(s))
print(len(s))
for i in s:
  print(i, end = ' ')
print()
print(s[0])
print(s*3)
print(s + s)
```

> Hà Nội Việt Nam 
> 
> <class 'str'>
> 
> 16
> 
> H à   N ộ i   V i ệ t   N a m   
> 
> H
> 
> Hà Nội Việt Nam Hà Nội Việt Nam Hà Nội Việt Nam 
> 
> Hà Nội Việt Nam Hà Nội Việt Nam 

Mọi đối tượng dữ liệu trong python đều có thể chuyển đổi sang dạng string bằng hàm `str()`. Ví dụ:

```python
n = 100
x = 3.14
a = [1,2,"One", [3,4]]
s = str(n) + " " + str(x) + " " + str(a)
print(s)
```

> 100 3.14 [1, 2, 'One', [3, 4]]

#### Hàm ord() và chr()
Python có hai hàm hệ thống lõi là `ord()` và `chr()` dùng để biến đổi ký tự sang mã số và ngược lại. Cú pháp như sau:

```python
ord(<ký tự>) = mã hóa của ký tự này trong bảng mã chuẩn
ord(<giá trị>) = ký tự tương ứng với mã hóa <giá trị> trong bảng mã chuẩn
```

Ví dụ:

```python
print(ord("A"))
print(ord("a"))
print(chr(65))
print(ord("Â"))
```

> 65
> 
> 97
> 
> A
> 
> 194

Ví dụ dưới biểu diễn một hàm chuyển xâu ký tự in thường thành in hoa:

```python
def viethoa(s):
    kq = ""
    for ch in s:
        kq = kq + chr(ord(ch) - 32)
    return kq

a = viethoa("baby")
print(a)
```

> BABY

Chú ý: Tính chất trên không còn đúng với các bảng chữ cái tiếng việt nữa. Vì vậy khi xử lý tiếng việt cần rất chú ý điểm này.

#### Các phương thức trong String

Dưới đây là bảng các phương thức của `string`:


| **Nhóm**                  | **Phương thức**                     | **Mô tả**                                                                                 |
|---------------------------|-------------------------------------|------------------------------------------------------------------------------------------|
| **Thao tác chuỗi**        | `capitalize()`                     | Viết hoa chữ cái đầu tiên của chuỗi.                                                     |
|                           | `casefold()`                       | Chuyển chuỗi thành chữ thường (tốt hơn `lower()`).                                       |
|                           | `lower()`                          | Chuyển chuỗi thành chữ thường.                                                          |
|                           | `upper()`                          | Chuyển chuỗi thành chữ hoa.                                                             |
|                           | `title()`                          | Viết hoa chữ cái đầu tiên của mỗi từ.                                                   |
|                           | `swapcase()`                       | Đảo ngược chữ hoa và chữ thường.                                                        |
| **Kiểm tra**              | `isalpha()`                        | Kiểm tra chuỗi chỉ chứa ký tự chữ cái.                                                  |
|                           | `isdigit()`                        | Kiểm tra chuỗi chỉ chứa số.                                                             |
|                           | `isalnum()`                        | Kiểm tra chuỗi chỉ chứa chữ cái và số.                                                  |
|                           | `isspace()`                        | Kiểm tra chuỗi chỉ chứa khoảng trắng.                                                   |
|                           | `isupper()`                        | Kiểm tra chuỗi chỉ chứa chữ hoa.                                                        |
|                           | `islower()`                        | Kiểm tra chuỗi chỉ chứa chữ thường.                                                     |
|                           | `istitle()`                        | Kiểm tra chuỗi có dạng tiêu đề không (mỗi từ bắt đầu bằng chữ hoa).                      |
| **Tìm kiếm**              | `find(sub[, start[, end]])`        | Tìm vị trí xuất hiện đầu tiên của chuỗi con (trả về `-1` nếu không tìm thấy).            |
|                           | `rfind(sub[, start[, end]])`       | Tìm vị trí xuất hiện cuối cùng của chuỗi con.                                           |
|                           | `index(sub[, start[, end]])`       | Tương tự `find()` nhưng ném lỗi nếu không tìm thấy.                                     |
|                           | `rindex(sub[, start[, end]])`      | Tương tự `rfind()` nhưng ném lỗi nếu không tìm thấy.                                    |
|                           | `count(sub[, start[, end]])`       | Đếm số lần xuất hiện của chuỗi con.                                                    |
| **Thay đổi chuỗi**        | `replace(old, new[, count])`       | Thay thế chuỗi con bằng chuỗi khác.                                                    |
|                           | `join(iterable)`                   | Ghép các chuỗi trong một iterable bằng chuỗi hiện tại.                                  |
|                           | `split(sep=None, maxsplit=-1)`     | Chia chuỗi thành danh sách bằng ký tự phân tách.                                        |
|                           | `rsplit(sep=None, maxsplit=-1)`    | Tương tự `split()` nhưng từ phải sang trái.                                            |
|                           | `partition(sep)`                   | Chia chuỗi thành 3 phần tại lần xuất hiện đầu tiên của `sep`.                           |
|                           | `rpartition(sep)`                  | Tương tự `partition()` nhưng tại lần xuất hiện cuối cùng.                               |
|                           | `strip([chars])`                   | Loại bỏ ký tự ở đầu và cuối chuỗi (mặc định là khoảng trắng).                            |
|                           | `lstrip([chars])`                  | Loại bỏ ký tự ở đầu chuỗi.                                                              |
|                           | `rstrip([chars])`                  | Loại bỏ ký tự ở cuối chuỗi.                                                             |
|                           | `expandtabs(tabsize=8)`            | Thay thế ký tự tab bằng khoảng trắng (mặc định mỗi tab là 8 khoảng trắng).              |
| **Căn chỉnh**             | `center(width[, fillchar])`        | Căn giữa chuỗi với độ rộng xác định.                                                   |
|                           | `ljust(width[, fillchar])`         | Căn trái chuỗi với độ rộng xác định.                                                   |
|                           | `rjust(width[, fillchar])`         | Căn phải chuỗi với độ rộng xác định.                                                   |
|                           | `zfill(width)`                     | Thêm số `0` vào đầu chuỗi để đạt độ rộng xác định.                                      |
| **Khác**                  | `startswith(prefix[, start[, end]])`| Kiểm tra chuỗi có bắt đầu bằng chuỗi con không.                                         |
|                           | `endswith(suffix[, start[, end]])` | Kiểm tra chuỗi có kết thúc bằng chuỗi con không.                                        |
|                           | `encode(encoding="utf-8")`         | Mã hóa chuỗi thành bytes.                                                              |
|                           | `decode(encoding="utf-8")`         | Giải mã bytes thành chuỗi.                                                             |
|                           | `format(*args, **kwargs)`          | Định dạng chuỗi.                                                                        |
|                           | `format_map(mapping)`              | Định dạng chuỗi với mapping.                                                           |
|                           | `maketrans(x[, y[, z]])`           | Tạo bảng dịch (cho phương thức `translate()`).                                         |
|                           | `translate(table)`                 | Áp dụng bảng dịch lên chuỗi.                                                           |
|                           | `splitlines([keepends])`           | Chia chuỗi thành danh sách các dòng.                                                   |



Dưới đây là ví dụ minh họa cho tất cả các phương thức trên của chuỗi trong Python:

```python
# Thao tác chuỗi
s = "hello world"

# capitalize()
print(s.capitalize())  # "Hello world"

# casefold()
print(s.casefold())    # "hello world"

# lower()
print(s.lower())       # "hello world"

# upper()
print(s.upper())       # "HELLO WORLD"

# title()
print(s.title())       # "Hello World"

# swapcase()
print(s.swapcase())    # "HELLO WORLD"
# ------------------------------
# Kiểm tra

s = "Hello123"

# isalpha()
print(s.isalpha())  # False

# isdigit()
print(s.isdigit())  # False

# isalnum()
print(s.isalnum())  # True

# isspace()
print("   ".isspace())  # True

# isupper()
print("HELLO".isupper())  # True

# islower()
print("hello".islower())  # True

# istitle()
print("Hello World".istitle())  # True
# ------------------------------
# Tìm Kiếm
s = "hello world hello"

# find()
print(s.find("world"))  # 6

# rfind()
print(s.rfind("hello"))  # 12

# index()
print(s.index("world"))  # 6

# rindex()
print(s.rindex("hello"))  # 12

# count()
print(s.count("hello"))  # 2

#--------------------------------
# Thay đổi chuỗi
s = "hello world"

# replace()
print(s.replace("world", "Python"))  # "hello Python"

# join()
print("-".join(["Python", "is", "fun"]))  # "Python-is-fun"

# split()
print(s.split())  # ['hello', 'world']

# rsplit()
print(s.rsplit())  # ['hello', 'world']

# partition()
print(s.partition("world"))  # ('hello ', 'world', '')

# rpartition()
print(s.rpartition("hello"))  # ('', 'hello', ' world')

# strip()
print("  hello  ".strip())  # "hello"

# lstrip()
print("  hello".lstrip())  # "hello"

# rstrip()
print("hello  ".rstrip())  # "hello"

# expandtabs()
print("hello\tworld".expandtabs(4))  # "hello   world"

#---------------------------------
# Căn Chỉnh

s = "hello"

# center()
print(s.center(10, "-"))  # "--hello---"

# ljust()
print(s.ljust(10, "-"))  # "hello-----"

# rjust()
print(s.rjust(10, "-"))  # "-----hello"

# zfill()
print(s.zfill(10))  # "00000hello"

#---------------------------------
# Khác

s = "hello world"

# startswith()
print(s.startswith("hello"))  # True

# endswith()
print(s.endswith("world"))  # True

# encode()
print(s.encode("utf-8"))  # b'hello world'

# format()
print("Hello, {}".format("Python"))  # "Hello, Python"

# format_map()
data = {"name": "Python"}
print("{name} is great!".format_map(data))  # "Python is great!"

# maketrans() + translate()
table = str.maketrans("abc", "123")
print("abcabc".translate(table))  # "123123"

# splitlines()
print("hello\nworld".splitlines())  # ['hello', 'world']


```

## 14. Kiểu dữ liệu từ điển

Từ điển (dictionary) trong Python là một kiểu dữ liệu lưu trữ các cặp key-value (khóa và giá trị). Các key là duy nhất, còn các value có thể trùng lặp. Từ điển rất linh hoạt, cho phép ánh xạ (mapping) một khóa bất kỳ đến một giá trị.

Cú pháp:
- **Tạo từ điển rỗng:**

```python
d = {}
```

- **Tạo từ điển với các cặp key-value:**

```python
d = {"name": "John", "age": 25, "city": "New York"}

```

- **Thêm hoặc sửa giá trị:**

```python
d["age"] = 30  # Sửa giá trị
d["job"] = "Developer"  # Thêm cặp key-value mới
```

- **Xóa một cặp key-value:**

```python
del d["city"]

```

Dưới đây là bảng các phương thức hay dùng của kiểu dữ liệu từ điển:


| **Phương thức/Hàm**      | **Mô tả**                                                                                      | **Ví dụ**                                                                                     |
|--------------------------|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `len(d)`                 | Trả về số lượng cặp key-value trong từ điển.                                                 | `len({"a": 1, "b": 2})  # 2`                                                                |
| `d[key]`                 | Truy cập giá trị theo khóa `key`.                                                             | `d["name"]  # "John"`                                                                       |
| `d.get(key[, default])`  | Trả về giá trị của `key`, hoặc `default` nếu không tìm thấy.                                  | `d.get("age", 0)  # 25`                                                                     |
| `d.keys()`               | Trả về danh sách tất cả các key trong từ điển.                                                | `list(d.keys())  # ["name", "age", "city"]`                                                |
| `d.values()`             | Trả về danh sách tất cả các giá trị trong từ điển.                                            | `list(d.values())  # ["John", 25, "New York"]`                                             |
| `d.items()`              | Trả về danh sách các cặp `(key, value)` trong từ điển.                                        | `list(d.items())  # [("name", "John"), ("age", 25)]`                                       |
| `d.update(other_dict)`   | Thêm hoặc cập nhật các cặp key-value từ từ điển khác `other_dict`.                             | `d.update({"age": 30, "city": "Los Angeles"})`                                             |
| `d.pop(key[, default])`  | Xóa và trả về giá trị của `key`. Nếu không tìm thấy `key`, trả về `default` (nếu được cung cấp).| `d.pop("age")  # 25`                                                                        |
| `d.popitem()`            | Xóa và trả về một cặp key-value bất kỳ.                                                       | `d.popitem()  # ("name", "John")`                                                          |
| `d.clear()`              | Xóa tất cả các cặp key-value trong từ điển.                                                   | `d.clear()`                                                                                 |
| `d.copy()`               | Tạo một bản sao nông (*shallow copy*) của từ điển.                                             | `new_dict = d.copy()`                                                                       |


Dưới đây là các ví dụ minh họa:

- **Tạo và truy cập từ điển**

```python
d = {"name": "Alice", "age": 28, "city": "London"}

# Truy cập giá trị
print(d["name"])  # Alice
print(d.get("age"))  # 28

# Thêm và sửa giá trị
d["age"] = 30
d["job"] = "Engineer"
print(d)  # {'name': 'Alice', 'age': 30, 'city': 'London', 'job': 'Engineer'}

# Xóa giá trị
del d["city"]
print(d)  # {'name': 'Alice', 'age': 30, 'job': 'Engineer'}

```

- **Duyệt qua từ điển**

```python
d = {"a": 1, "b": 2, "c": 3}

# Duyệt qua key
for key in d.keys():
    print(key, end=" ")  # a b c

# Duyệt qua value
for value in d.values():
    print(value, end=" ")  # 1 2 3

# Duyệt qua cả key và value
for key, value in d.items():
    print(f"{key}: {value}")  # a: 1, b: 2, c: 3

```

- **Kết hợp từ điển**

```python
d1 = {"name": "Alice", "age": 30}
d2 = {"city": "Paris", "job": "Developer"}

# Cập nhật từ điển
d1.update(d2)
print(d1)  # {'name': 'Alice', 'age': 30, 'city': 'Paris', 'job': 'Developer'}

```

- **Sao chép và xóa**

```python
d = {"a": 1, "b": 2, "c": 3}

# Sao chép từ điển
copy_d = d.copy()
print(copy_d)  # {'a': 1, 'b': 2, 'c': 3}

# Xóa tất cả phần tử
d.clear()
print(d)  # {}

```

## 15. Kiểu dữ liệu Set

Set là một kiểu dữ liệu trong Python dùng để lưu trữ một tập hợp các phần tử không trùng lặp. Các phần tử trong set không có thứ tự (unordered) và không có chỉ số (index). Vì vậy, set thường được sử dụng khi bạn cần lưu trữ các giá trị duy nhất mà không quan tâm đến thứ tự của chúng.

Một điểm quan trọng là set **không cho phép các phần tử trùng lặp**.

Cú pháp:
- **Tạo một set rỗng**

```python
s = set()
```

- **Tạo một set với các phần tử**

```python
s = {1, 2, 3, 4}
```

- **Tạo set từ một iterable (danh sách, chuỗi, tuple, v.v.)**

```python
s = set([1, 2, 3, 4])
``` 

Dưới đây là các phương thức thường gặp trong kiểu dữ liệu `set`:

| **Phương thức/Hàm**             | **Mô tả**                                                                                         | **Ví dụ**                                                                                     |
|----------------------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `len(s)`                         | Trả về số lượng phần tử trong set.                                                                | `len({1, 2, 3})  # 3`                                                                        |
| `s.add(x)`                       | Thêm phần tử `x` vào set. Nếu phần tử đã tồn tại thì không thay đổi gì.                          | `s.add(4)`                                                                                   |
| `s.remove(x)`                    | Xóa phần tử `x` khỏi set. Nếu phần tử không tồn tại, sẽ gây lỗi `KeyError`.                      | `s.remove(3)`                                                                                 |
| `s.discard(x)`                   | Xóa phần tử `x` khỏi set. Nếu phần tử không tồn tại, không có gì xảy ra (không gây lỗi).        | `s.discard(3)`                                                                                 |
| `s.pop()`                        | Xóa và trả về một phần tử bất kỳ trong set. (Do set không có thứ tự nên phần tử bị xóa là ngẫu nhiên). | `s.pop()`                                                                                   |
| `s.clear()`                      | Xóa tất cả các phần tử trong set.                                                                  | `s.clear()`                                                                                   |
| `s.copy()`                       | Tạo một bản sao của set.                                                                          | `s2 = s.copy()`                                                                               |
| `s.union(other_set)`             | Trả về một set mới chứa tất cả các phần tử từ cả hai set (liên hợp).                             | `s.union({5, 6})`  # {1, 2, 3, 4, 5, 6}`                                                     |
| `s.intersection(other_set)`      | Trả về một set mới chứa các phần tử chung giữa hai set (giao nhau).                              | `s.intersection({3, 4, 5})`  # {3, 4}`                                                       |
| `s.difference(other_set)`        | Trả về một set mới chứa các phần tử trong set này nhưng không có trong set kia (hiệu).           | `s.difference({3, 4, 5})`  # {1, 2}`                                                         |
| `s.symmetric_difference(other_set)`| Trả về set chứa các phần tử mà chỉ có mặt trong một trong hai set (phần tử đối xứng).           | `s.symmetric_difference({3, 4, 5})`  # {1, 2, 5}`                                           |
| `s.isdisjoint(other_set)`        | Trả về `True` nếu hai set không có phần tử chung, ngược lại trả về `False`.                      | `s.isdisjoint({5, 6})`  # True`                                                            |
| `s.issubset(other_set)`          | Trả về `True` nếu set này là một tập con của set kia.                                              | `s.issubset({1, 2, 3, 4, 5})`  # True`                                                     |
| `s.issuperset(other_set)`        | Trả về `True` nếu set này là một tập siêu tập của set kia.                                         | `s.issuperset({1, 2})`  # True`                                                           |


Ví dụ minh họa:
- **Tạo và sử dụng set:**

```python
# Tạo set từ một danh sách
s = {1, 2, 3, 4}
print(s)  # {1, 2, 3, 4}

# Thêm phần tử
s.add(5)
print(s)  # {1, 2, 3, 4, 5}

# Thêm phần tử đã tồn tại (không thay đổi gì)
s.add(2)
print(s)  # {1, 2, 3, 4, 5}

# Xóa phần tử
s.remove(3)
print(s)  # {1, 2, 4, 5}

# Xóa phần tử không tồn tại (gây lỗi)
# s.remove(6)  # KeyError

# Xóa phần tử không gây lỗi nếu không tồn tại
s.discard(6)  # Không làm gì cả

# Lấy số lượng phần tử
print(len(s))  # 4

```

- **Phép toán trên set**

```python
set_a = {1, 2, 3}
set_b = {3, 4, 5}

# Liên hợp (union)
union_set = set_a.union(set_b)
print(union_set)  # {1, 2, 3, 4, 5}

# Giao nhau (intersection)
intersection_set = set_a.intersection(set_b)
print(intersection_set)  # {3}

# Hiệu (difference)
difference_set = set_a.difference(set_b)
print(difference_set)  # {1, 2}

# Đối xứng (symmetric_difference)
sym_diff_set = set_a.symmetric_difference(set_b)
print(sym_diff_set)  # {1, 2, 4, 5}

```

- **Các phép toán kiểm tra quan hệ giữa các set**

```python
# Kiểm tra một set có phải là tập con của set khác
print(set_a.issubset(set_b))  # False

# Kiểm tra một set có phải là tập siêu tập của set khác
print(set_a.issuperset({1, 2}))  # True

# Kiểm tra hai set có phần tử chung hay không
print(set_a.isdisjoint({4, 5}))  # True

```

## 16. Đọc và Ghi File trong Python

Trong Python, việc đọc và ghi file là một công việc cơ bản và rất quan trọng. Python cung cấp các hàm và phương thức để làm việc với các file trên hệ thống. Dưới đây là các thông tin chi tiết về cách đọc và ghi file trong Python.

#### Các kiểu mở file

Để thao tác với file, đầu tiên bạn cần phải mở file đó bằng hàm open(). Hàm open() trả về một đối tượng file mà bạn có thể sử dụng để đọc hoặc ghi dữ liệu vào file.

Cú pháp cơ bản:

```python
file = open('ten_file.txt', 'mode')
```

Các chế độ mở file (mode) thường gặp:

- `'r'`: Mở file để đọc. Nếu file không tồn tại, sẽ gây lỗi FileNotFoundError.
- `'w'`: Mở file để ghi. Nếu file đã tồn tại, nó sẽ bị ghi đè. Nếu không tồn tại, file mới sẽ được tạo.
- `'a'`: Mở file để ghi thêm (append). Nếu file không tồn tại, nó sẽ được tạo mới.
- `'rb'`: Mở file để đọc dưới dạng nhị phân.
- `'wb'`: Mở file để ghi dưới dạng nhị phân.
- `'r+'`: Mở file để đọc và ghi. Nếu file không tồn tại, gây lỗi.
- `'b'`: Mở file trong chế độ nhị phân.


#### Đọc dữ liệu từ file

Có nhiều cách để đọc dữ liệu từ file, tùy thuộc vào mục đích sử dụng.

- **Đọc toàn bộ nội dung file:** Sử dụng phương thức `read()` để đọc toàn bộ nội dung trong file.

```python
with open('ten_file.txt', 'r') as file:
    content = file.read()
    print(content)
```

- **Đọc theo dòng:** Phương thức `readline()` đọc một dòng từ file mỗi lần gọi.

```python
with open('ten_file.txt', 'r') as file:
    line = file.readline()
    print(line)
```

- **Đọc tất cả các dòng:** Phương thức readlines() đọc tất cả các dòng trong file và trả về một danh sách, mỗi phần tử là một dòng trong file.

```python
with open('ten_file.txt', 'r') as file:
    lines = file.readlines()
    print(lines)

```

#### Ghi dữ liệu vào file

Để ghi dữ liệu vào file, bạn có thể sử dụng phương thức `write()` hoặc `writelines()`.

- **Ghi một chuỗi vào file:** Phương thức `write()` dùng để ghi một chuỗi vào file.

```python
with open('ten_file.txt', 'w') as file:
    file.write('Hello, world!')

```

- **Ghi nhiều dòng vào file:** Phương thức `writelines()` dùng để ghi một danh sách các dòng vào file.

```python
lines = ['Line 1\n', 'Line 2\n', 'Line 3\n']
with open('ten_file.txt', 'w') as file:
    file.writelines(lines)

```

- **Ghi thêm dữ liệu vào cuối file:** Để ghi thêm dữ liệu vào cuối file mà không ghi đè lên dữ liệu cũ, bạn sử dụng chế độ `'a'` (append).

```python
with open('ten_file.txt', 'a') as file:
    file.write('This is an additional line.\n')
```

#### Quản lý file và bảo vệ việc đóng file

Khi thao tác với file, bạn cần phải đóng file sau khi sử dụng để giải phóng tài nguyên. Phương thức `close()` được sử dụng để đóng file.

```python
file = open('ten_file.txt', 'r')
# Thực hiện các thao tác với file
file.close()  # Đảm bảo đóng file

```

- Tuy nhiên, cách tốt nhất để làm việc với file trong Python là sử dụng **context manager** (`with`), vì Python tự động đóng file sau khi thoát khỏi khối mã.

```python
with open('ten_file.txt', 'r') as file:
    content = file.read()
# File sẽ tự động được đóng sau khi thoát khỏi khối `with`

```

#### Đọc và ghi file nhị phân
Ngoài việc thao tác với các file văn bản, Python cũng hỗ trợ đọc và ghi các file nhị phân (như ảnh, video, và các tệp dữ liệu khác).
- **Đọc file nhị phân:** Sử dụng chế độ `'rb'` để mở file nhị phân.

```python
with open('image.jpg', 'rb') as file:
    data = file.read()
    # Thực hiện các thao tác với dữ liệu nhị phân

```

- **Ghi file nhị phân:** Sử dụng chế độ `'wb'` để ghi vào file nhị phân

```python
with open('image_copy.jpg', 'wb') as file:
    file.write(data)  # Ghi dữ liệu nhị phân
```

####  Các lưu ý khi làm việc với file trong Python

**Xử lý lỗi**: Đảm bảo rằng bạn kiểm tra sự tồn tại của file trước khi đọc hoặc ghi. Sử dụng `try-except` để xử lý lỗi khi mở file.

```python
try:
    with open('ten_file.txt', 'r') as file:
        content = file.read()
except FileNotFoundError:
    print("File không tồn tại.")

```

**Kích thước file lớn:** Nếu bạn làm việc với các file rất lớn, bạn có thể đọc file theo từng phần thay vì đọc toàn bộ nội dung vào bộ nhớ.

```python
with open('ten_file.txt', 'r') as file:
    for line in file:
        print(line, end='')
```

## 18. Tổng Kết

Ở trên là tất cả những kiến thức cơ bản về Python, nhưng còn môt phần quan trọng nữa trong Python là phần OOP và tôi sẽ viết riêng một bài về nó. Python là một ngôn ngữ dễ dàng nhất để sử dụng và phổ biến nhất để học AI. Chúc mừng bạn, bạn đã biết thêm một ngôn ngữ lập trình nữa 🍀 🍀 🍀

