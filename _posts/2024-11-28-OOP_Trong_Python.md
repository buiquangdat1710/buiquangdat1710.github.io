---
title: "OOP Trong Python"
date: 2024-11-28 00:00:00  + 0800
categories: [Giáo Trình Dạy AI ProPTIT]
tags: [proptit]
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


Lập trình hướng đối tượng (OOP - Object-Oriented Programming) là một phương pháp lập trình dựa trên khái niệm "đối tượng" (object). Đối tượng là các thực thể có trạng thái (thuộc tính) và hành vi (phương thức). OOP giúp tổ chức mã nguồn một cách trực quan và logic, đồng thời tăng khả năng tái sử dụng và bảo trì. Python là một ngôn ngữ hỗ trợ OOP mạnh mẽ. Trong Python, mọi thứ đều là đối tượng, kể cả các kiểu dữ liệu cơ bản như số nguyên, chuỗi, danh sách, v.v.

## 1. Tổng quan về lập trình hướng đối tượng (OOP)

Ý tưởng chung của lập trình hướng đối tượng xuất phát từ mô hình thực tế xung quanh chúng ta: Tất cả những đồ vật, vật dụng hàng ngày, xung quanh chúng ta đều nằm trong các nhóm riêng của mình. Ví dụ: cái bàn, ghế nằm trong nhóm đồ dùng gia đình; máy tính, điện thoại, máy in đều nằm trong các nhóm thiết bị văn phòng. Trong xã hội con ngươi cũng vậy, mỗi con người đều nằm trong các nhóm, ví dụ dân tộc, quốc tịch. Điều này cũng được mang vào các ngôn ngữ lập trình. Nhưng trong ngôn ngữ lập trình các tên gọi sẽ được thay đổi:

Từ **đồ vật** sẽ được thay đổi trong lập trình **Đối tượng** (Objects).

Từ **nhóm** sẽ được thay đổi trong lập trình thành **Lớp** (Class)

Ví dụ, chúng ta đã biết về các kiểu dữ liệu như xâu ký tự (str), kiểu số nguyên (int) hay kiểu số thực (float). Khi một biến nhớ được khai báo, ví dụ x, thì x là **đối tượng**, kiểu dữ liệu của x sẽ là **lớp**:

```python
x = "Việt Nam"
print(type(x)) # <class 'str'>
x = 12
print(type(x)) # <class 'int'>
x = 12.3
print(type(x)) # <class 'float'>
```

## 2. Khởi tạo Lớp và Đối Tượng

Trong mục này chúng ta sẽ làm quen với các lệnh khởi tạo và làm việc với các Lớp và Đối tượng trong Python. Lệnh tạo Lớp (class) trong Python đơn giản như sau:

```python
class <Tên Lớp>:
    <phần mô tả lớp>
```

Trong đó \<phần mô tả Lớp\> là các khai báo thuộc tính và phương thức cho lớp này. Xét một ví dụ sau:

```python
class SinhVien:
    name = "Đặng Huyền Trang"
    age = 20
    gender = "Nữ"
    school = "PTIT"
```

Trong ví dụ trên một Lớp có tên là **SinhVien** đã được khai báo. Lớp này có 4 thuộc tính name, age, gender, school. Các thuộc tính này được tạo và gán giá trị mặc định trong phần thân của lớp. Chúng ta sẽ khai báo một đối tượng mới **sv1** của lớp **SinhVien**:

```python
sv1 = SinhVien()
print(sv1.name) 
```

> Đặng Huyền Trang

Để truy cập thuộc tính cụ thể của đối tượng, chúng ta viết như sau:

```python
<Đối tương>.<thuộc tính>
```

Có thể dễ dàng thay đổi thuộc tính bằng lệnh gán trực tiếp:

```python
sv2 = SinhVien()
sv2.name = "Bùi Quang Đạt"
print(sv2.name)
```

> Bùi Quang Đạt

Bây giờ chúng ta bổ sung khai báo phương thức cho lớp **SinhVien** này:

```python
class SinhVien:
    name = "Đặng Huyền Trang"
    age = 20
    gender = "Nữ"
    school = "PTIT"
    def show(self):
        print("Name = ", self.name)
        print("Age = ", self.age)
        print("School = ", self.school)
        print("Gender = ", self.gender)
    def update(self,name,age,gender,school):
        self.name = name
        self.age = age
        self.gender = gender
        self.school = school
```

Từ khóa `self` là từ dùng để chỉ bản thân đối tượng, rất hay dùng trong khai báo phương thức. Có thể dùng tên khác nhưng Python khuyên nên sử dụng tên `self` này. Hãy nhớ rằng mọi phương thức đều có tham số `self`. Ví dụ sử dụng phương thức:

```python
sv1 = SinhVien()
sv1.show()
sv1.update("Dương", 18, "Nam" , "THPT Hoài Đức A")
print(sv1.name)
```

> Name =  Đặng Huyền Trang
> 
> Age =  20
> 
> School =  PTIT
> 
> Gender =  Nữ
> 
> Dương


#### Hàm __init__()

Hàm `__init__()` là một hàm đặc biệt trong Python, thường được gọi là **constructor** (hàm khởi tạo). Nó được tự động gọi mỗi khi một đối tượng mới được tạo từ một lớp.
- Mục đích chính của `__init__()` là khởi tạo (initialize) các thuộc tính của đối tượng.
- Đây không phải là bắt buộc, nhưng nếu cần gán giá trị mặc định hoặc thực hiện các thiết lập ban đầu cho đối tượng, bạn nên sử dụng hàm này.

Khi một đối tượng được tạo, Python gọi hàm `__init__()` với các tham số được truyền vào lúc khởi tạo đối tượng. Điều này cho phép bạn thiết lập trạng thái ban đầu của đối tượng ngay từ khi nó được tạo. Dưới đây ví dụ cơ bản:

```python
class Dog:
    def __init__(self, name, breed):
        self.name = name  # Gán giá trị cho thuộc tính 'name'
        self.breed = breed  # Gán giá trị cho thuộc tính 'breed'

    def info(self):
        return f"{self.name} is a {self.breed}"

# Tạo đối tượng
dog1 = Dog("Buddy", "Golden Retriever")
dog2 = Dog("Charlie", "Labrador")

# Truy cập thuộc tính và phương thức
print(dog1.info())  # Output: Buddy is a Golden Retriever
print(dog2.info())  # Output: Charlie is a Labrador

```

Nếu một tham số không được truyền khi tạo đối tượng, bạn có thể thiết lập giá trị mặc định trong `__init__()`:

```python
class Person:
    def __init__(self, name, age=18):
        self.name = name
        self.age = age

# Tạo đối tượng với giá trị mặc định
person1 = Person("Alice")
person2 = Person("Bob", 25)

print(person1.name, person1.age)  # Output: Alice 18
print(person2.name, person2.age)  # Output: Bob 25

```

Bạn có thể thực hiện thêm các tính toán hoặc kiểm tra trong `__init__()`.

```python
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner
        if balance >= 0:  # Kiểm tra giá trị số dư ban đầu
            self.balance = balance
        else:
            self.balance = 0
            print("Balance cannot be negative. Setting balance to 0.")

# Tạo đối tượng
account1 = BankAccount("Alice", 500)
account2 = BankAccount("Bob", -100)

print(account1.owner, account1.balance)  # Output: Alice 500
print(account2.owner, account2.balance)  # Output: Bob 0

```

`__init__()` có thể được sử dụng để khởi tạo các thuộc tính dạng danh sách hoặc đối tượng phức tạp:

```python
class Classroom:
    def __init__(self, students=None):
        if students is None:
            self.students = []  # Khởi tạo danh sách rỗng nếu không có tham số
        else:
            self.students = students

    def add_student(self, student):
        self.students.append(student)

# Tạo đối tượng
classroom = Classroom(["Alice", "Bob"])
classroom.add_student("Charlie")
print(classroom.students)  # Output: ['Alice', 'Bob', 'Charlie']

```

#### Toán Tử Chồng (Operator Overloading)

Toán tử chồng (Operator Overloading) là kỹ thuật cho phép định nghĩa lại cách hoạt động của các toán tử như `+`, `-`, `*`, `==`, v.v., đối với các đối tượng của một lớp. Trong Python, toán tử chồng được thực hiện bằng cách sử dụng các phương thức đặc biệt (magic methods), còn gọi là dunder methods (double underscore methods).

Ví dụ:
- Thay vì sử dụng toán tử + để cộng hai số, bạn có thể định nghĩa lại để cộng hai đối tượng tùy chỉnh.

Lợi ích của toán tử chồng:
- Tăng tính dễ đọc và trực quan khi làm việc với các lớp tùy chỉnh.
- Cho phép sử dụng các toán tử quen thuộc trên các đối tượng phức tạp.
- Làm cho mã ngắn gọn và dễ hiểu hơn.

Bảng Các Phương Thức Toán Tử Chồng Trong Python:

| **Toán Tử** | **Phương Thức Đặc Biệt**    | **Mô Tả**                                         |
|-------------|-----------------------------|--------------------------------------------------|
| `+`         | `__add__(self, other)`      | Cộng hai đối tượng                                |
| `-`         | `__sub__(self, other)`      | Trừ hai đối tượng                                 |
| `*`         | `__mul__(self, other)`      | Nhân hai đối tượng                                |
| `/`         | `__truediv__(self, other)`  | Chia hai đối tượng                                |
| `//`        | `__floordiv__(self, other)` | Chia lấy phần nguyên                             |
| `%`         | `__mod__(self, other)`      | Chia lấy phần dư                                  |
| `**`        | `__pow__(self, other)`      | Lũy thừa                                          |
| `==`        | `__eq__(self, other)`       | So sánh bằng nhau                                 |
| `!=`        | `__ne__(self, other)`       | So sánh khác nhau                                 |
| `<`         | `__lt__(self, other)`       | So sánh nhỏ hơn                                   |
| `<=`        | `__le__(self, other)`       | So sánh nhỏ hơn hoặc bằng                         |
| `>`         | `__gt__(self, other)`       | So sánh lớn hơn                                   |
| `>=`        | `__ge__(self, other)`       | So sánh lớn hơn hoặc bằng                         |
| `[]`        | `__getitem__(self, key)`    | Truy cập phần tử bằng chỉ số                      |
| `[] =`      | `__setitem__(self, key, value)` | Gán giá trị cho phần tử theo chỉ số              |
| `del []`    | `__delitem__(self, key)`    | Xóa phần tử theo chỉ số                           |
| `len()`     | `__len__(self)`             | Lấy độ dài của đối tượng                          |
| `repr()`    | `__repr__(self)`            | Trả về chuỗi đại diện của đối tượng               |
| `str()`     | `__str__(self)`             | Trả về chuỗi mô tả đối tượng                      |
| `in`        | `__contains__(self, item)`  | Kiểm tra xem phần tử có trong đối tượng hay không |
| `callable()`| `__call__(self, *args, **kwargs)` | Gọi một đối tượng như hàm                        |
| `+ (unary)` | `__pos__(self)`             | Dấu dương của đối tượng                           |
| `- (unary)` | `__neg__(self)`             | Dấu âm của đối tượng                              |
| `abs()`     | `__abs__(self)`             | Giá trị tuyệt đối của đối tượng                   |
| `~`         | `__invert__(self)`          | Toán tử bitwise NOT                               |
| `&`         | `__and__(self, other)`      | Toán tử bitwise AND                               |
| `|`         | `__or__(self, other)`       | Toán tử bitwise OR                                |
| `^`         | `__xor__(self, other)`      | Toán tử bitwise XOR                               |
| `<<`        | `__lshift__(self, other)`   | Toán tử dịch trái bit                             |
| `>>`        | `__rshift__(self, other)`   | Toán tử dịch phải bit                             |
| `+=`        | `__iadd__(self, other)`     | Cộng và gán                                       |
| `-=`        | `__isub__(self, other)`     | Trừ và gán                                        |
| `*=`        | `__imul__(self, other)`     | Nhân và gán                                       |
| `/=`        | `__itruediv__(self, other)` | Chia và gán                                       |
| `//=`       | `__ifloordiv__(self, other)`| Chia lấy phần nguyên và gán                       |
| `%=`        | `__imod__(self, other)`     | Chia lấy phần dư và gán                           |
| `**=`       | `__ipow__(self, other)`     | Lũy thừa và gán                                   |
| `&=`        | `__iand__(self, other)`     | Bitwise AND và gán                                |
| `|=`        | `__ior__(self, other)`      | Bitwise OR và gán                                 |
| `^=`        | `__ixor__(self, other)`     | Bitwise XOR và gán                                |
| `<<=`       | `__ilshift__(self, other)`  | Dịch trái và gán                                  |
| `>>=`       | `__irshift__(self, other)`  | Dịch phải và gán   

Ví dụ 1: Toán tử + (Cộng hai đối tượng)

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

# Tạo hai đối tượng Vector
v1 = Vector(2, 3)
v2 = Vector(4, 5)

# Sử dụng toán tử +
v3 = v1 + v2
print(v3)  # Output: Vector(6, 8)

```

Ví dụ 2: Toán tử * (Nhân hai đối tượng)

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

# Tạo một đối tượng Vector
v = Vector(2, 3)

# Sử dụng toán tử *
v2 = v * 3
print(v2)  # Output: Vector(6, 9)

```

Ví dụ 3: So sánh với toán tử ==

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

# Tạo hai đối tượng Point
p1 = Point(1, 2)
p2 = Point(1, 2)
p3 = Point(3, 4)

# So sánh các đối tượng
print(p1 == p2)  # Output: True
print(p1 == p3)  # Output: False

```

Bạn có thể chồng toán tử [] bằng cách sử dụng phương thức `__getitem__()`:

```python
class MyList:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

# Tạo một đối tượng MyList
my_list = MyList([1, 2, 3, 4])

# Sử dụng toán tử []
print(my_list[2])  # Output: 3

```

Bạn có thể định nghĩa lại toán tử `len()` bằng cách sử dụng phương thức `__len__()`:

```python
class MyList:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

# Tạo một đối tượng MyList
my_list = MyList([1, 2, 3, 4])

# Sử dụng len()
print(len(my_list))  # Output: 4

```

Lưu ý khi sử dụng toán tử chồng
- Sử dụng toán tử chồng hợp lý:
  - Không nên định nghĩa lại toán tử theo cách mơ hồ hoặc không trực quan.
  - Ví dụ: Nếu toán tử + không thực hiện cộng dồn mà lại xóa giá trị, sẽ gây nhầm lẫn.
- Kiểm tra kiểu dữ liệu:
  - Đảm bảo kiểm tra kiểu của đối tượng đầu vào để tránh lỗi runtime.
```python
def __add__(self, other):
    if not isinstance(other, Vector):
        raise TypeError("Operand must be an instance of Vector")
    return Vector(self.x + other.x, self.y + other.y)

```

- Kế thừa toán tử từ lớp cha:
  - Nếu lớp con kế thừa lớp cha, bạn có thể sử dụng hoặc ghi đè toán tử đã được định nghĩa trong lớp cha.

## 3. Bốn tính chất OOP trong Python

Trong lập trình hướng đối tượng (OOP), có 4 tính chất chính mà bạn cần hiểu: **Encapsulation (Đóng gói)**, **Abstraction (Trừu tượng hóa)**, **Inheritance (Kế thừa)** và **Polymorphism (Đa hình)**. Dưới đây là giải thích chi tiết và ví dụ code cho từng tính chất này trong Python.

#### Encapsulation (Đóng gói)

Đóng gói là quá trình ẩn các dữ liệu và chỉ cho phép truy cập thông qua các phương thức công khai. Điều này giúp bảo vệ dữ liệu khỏi việc bị thay đổi trái phép và kiểm soát cách thức truy cập dữ liệu.

```python
class Car:
    def __init__(self, make, model, year):
        self.make = make        # Dữ liệu công khai
        self.model = model      # Dữ liệu công khai
        self.__year = year      # Dữ liệu bị ẩn (private)

    # Phương thức công khai để truy cập dữ liệu bị ẩn
    def get_year(self):
        return self.__year

    # Phương thức công khai để thay đổi dữ liệu bị ẩn
    def set_year(self, year):
        if year > 0:
            self.__year = year
        else:
            print("Năm không hợp lệ")

car = Car("Toyota", "Corolla", 2020)
print(car.get_year())  # Sử dụng phương thức để truy cập dữ liệu ẩn
car.set_year(2022)     # Thay đổi năm thông qua phương thức
print(car.get_year())  # In ra năm mới

```

Giải thích:
- Dữ liệu `__year` được ẩn (private) bằng cách sử dụng dấu gạch dưới đôi (__), và không thể truy cập trực tiếp từ bên ngoài lớp. Nếu bạn cố sử dụng lệnh `car.year` thì bạn sẽ gặp lỗi.
- Các phương thức `get_year()` và `set_year()` được cung cấp để truy cập và thay đổi giá trị `__year`.

#### Abstraction (Trừu tượng hóa)

Trừu tượng hóa là quá trình ẩn đi những chi tiết cài đặt không cần thiết và chỉ cung cấp các chức năng cơ bản cần thiết cho người sử dụng. Điều này giúp giảm độ phức tạp.

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def sound(self):
        pass

class Dog(Animal):
    def sound(self):
        return "Bark"

class Cat(Animal):
    def sound(self):
        return "Meow"

dog = Dog()
cat = Cat()
print(dog.sound())  # In ra: Bark
print(cat.sound())  # In ra: Meow
```

Giải thích:
- Lớp `Animal` là một lớp trừu tượng với phương thức `sound()` được đánh dấu là `abstractmethod`. Điều này có nghĩa là mọi lớp kế thừa từ `Animal` phải định nghĩa phương thức `sound()`.
- Lớp `Dog` và `Cat` kế thừa từ lớp `Animal` và cài đặt phương thức `sound()` của riêng chúng.

#### Inheritance (Kế thừa)

Kế thừa là cơ chế cho phép một lớp con kế thừa thuộc tính và phương thức của lớp cha. Điều này giúp tái sử dụng mã nguồn và xây dựng các lớp mới dựa trên các lớp đã có.

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound."

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # Gọi hàm khởi tạo của lớp cha
        self.breed = breed

    def speak(self):
        return f"{self.name} barks."

class Cat(Animal):
    def __init__(self, name, color):
        super().__init__(name)
        self.color = color

    def speak(self):
        return f"{self.name} meows."

dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", "Grey")

print(dog.speak())  # In ra: Buddy barks.
print(cat.speak())  # In ra: Whiskers meows.
```

Giải thích:
- Lớp `Dog` và `Cat` đều kế thừa từ lớp `Animal` và mỗi lớp con có thể cài đặt phương thức `speak()` riêng của mình.
- Hàm `super().__init__(name)` trong lớp con gọi phương thức khởi tạo của lớp cha `Animal`.

#### Polymorphism (Đa hình)

Đa hình cho phép các đối tượng khác nhau có thể gọi cùng một phương thức mà hành vi của chúng có thể khác nhau tùy thuộc vào lớp đối tượng. Điều này giúp tăng tính linh hoạt và khả năng mở rộng của chương trình.


```python
class Bird:
    def sound(self):
        return "Tweet"

class Dog:
    def sound(self):
        return "Bark"

class Cow:
    def sound(self):
        return "Moo"

def make_sound(animal):
    print(animal.sound())

bird = Bird()
dog = Dog()
cow = Cow()

make_sound(bird)  # In ra: Tweet
make_sound(dog)   # In ra: Bark
make_sound(cow)   # In ra: Moo

```

Giải thích:`
- Mặc dù cả ba lớp `Bird`, `Dog`, và `Cow` đều có phương thức `sound()`, nhưng hành vi của phương thức này lại khác nhau.
- Hàm `make_sound()` có thể nhận bất kỳ đối tượng nào từ các lớp này và gọi phương thức `sound()` mà không cần biết lớp của đối tượng đó, điều này thể hiện tính đa hình.

Tóm tắt 4 thuộc tính OOP trong Python:

- **Encapsulation (Đóng gói)** giúp bảo vệ dữ liệu và điều khiển quyền truy cập.
- **Abstraction (Trừu tượng)** giúp ẩn các chi tiết cài đặt không cần thiết và chỉ cung cấp các chức năng cơ bản.
- **Inheritance (Kế thừa)** cho phép lớp con kế thừa và mở rộng các thuộc tính và phương thức của lớp cha.
- **Polymorphism (Đa hình)** cho phép các đối tượng khác nhau thể hiện hành vi khác nhau thông qua cùng một phương thức.

## 4. Tổng kết

Vậy là bạn đã học xong phần kiến thức cơ bản của Python. Nhưng để học AI thì còn rất nhiều thứ phải học nữa, bao gồm các thư viện và framework trong Python. Nhưng đừng nản chí, bạn đã đi được một quãng đường khá xa rồi đó 🍀 🍀 🍀
