---
title: "Biểu thức chính quy"
date: 2024-09-30 00:00:00  + 0800
categories: [NLP]
tags: [Regrex]
---
---

**Biểu thức chính quy (Regular Expression)**, thường viết tắt là regex, là một chuỗi các ký tự đặc biệt dùng để mô tả một mẫu (pattern) tìm kiếm trong văn bản. Biểu thức chính quy được sử dụng để tìm kiếm, so khớp hoặc thay thế các mẫu cụ thể trong dữ liệu văn bản. Với khả năng linh hoạt và mạnh mẽ, regex giúp xử lý các tác vụ phức tạp như kiểm tra định dạng email, số điện thoại, hoặc trích xuất các thông tin cụ thể từ văn bản một cách nhanh chóng và hiệu quả. Regex là công cụ phổ biến trong nhiều ngôn ngữ lập trình và hệ thống xử lý văn bản.

## 1. Giới thiệu

<table style="margin-left: auto; margin-right: auto;">
  <tr>
    <td><strong>User:</strong></td>
    <td>I am unhappy.</td>
  </tr>
  <tr>
    <td><strong>ELIZA:</strong></td>
    <td>DO YOU THINK COMING HERE WILL HELP YOU NOT TO BE UNHAPPY?</td>
  </tr>
  <tr>
    <td><strong>User:</strong></td>
    <td>I need some help, that much seems certain.</td>
  </tr>
  <tr>
    <td><strong>ELIZA:</strong></td>
    <td>WHAT WOULD IT MEAN TO YOU IF YOU GOT SOME HELP?</td>
  </tr>
  <tr>
    <td><strong>User:</strong></td>
    <td>Perhaps I could learn to get along with my mother.</td>
  </tr>
  <tr>
    <td><strong>ELIZA:</strong></td>
    <td>TELL ME MORE ABOUT YOUR FAMILY.</td>
  </tr>
  <tr>
    <td><strong>User:</strong></td>
    <td>My mother takes care of me.</td>
  </tr>
  <tr>
    <td><strong>ELIZA:</strong></td>
    <td>WHO ELSE IN YOUR FAMILY TAKES CARE OF YOU?</td>
  </tr>
  <tr>
    <td><strong>User:</strong></td>
    <td>My father.</td>
  </tr>
  <tr>
    <td><strong>ELIZA:</strong></td>
    <td>YOUR FATHER.</td>
  </tr>
  <tr>
    <td><strong>User:</strong></td>
    <td>You are like my father in some ways.</td>
  </tr>
</table>




Đoạn hội thoại trên là từ **ELIZA**, một trong chatbot đầu tiên trên thế giới. Nó trò chuyện với con người bằng cách bắt chước phản hồi của một nhà trị liệu tâm lý theo phong cách của  Rogerian. ELIZA là một chương trình rất đơn giản, sử dụng kĩ thuật so sánh các xâu con để trả lời theo câu trả lời mặc định. Ví dụ như nếu nó tìm thấy trong câu hỏi của người dùng có xâu con: "I need X" thì nó sẽ trả lời rằng: "WHAT WOULD IT MEAN TO YOU IF YOU GOT X". Nghe buồn cười đúng không, chính xác đây là chatbot if else mà nhiều người nói đến nhưng ELIZA lại rất thành công trong lĩnh vực trị liệu tâm lý vì nhiều người tương tác với ELIZA và tin rằng nó thực sự **hiểu** họ và các vấn đề của họ. Nhiều người vẫn tin vào khả năng của ELIZA ngay cả khi họ biết cách nó hoạt động (chỉ đơn giản là if else). Cho đến nay, những chatbot đượt lập trình giống như ELIZA vẫn luôn là một cách lập trình chatbot đơn giản mà không cần đến ngữ nghĩa. Và để có thể tìm được các xâu con hay cụ thể hơn là tìm một mẫu nào đó trong văn bản, người ta gọi kĩ thuật đó là **Biểu thức chính quy**.

Biểu thức chính quy, thường được gọi là **regex** (viết tắt của *regular expression*), là một công cụ mạnh mẽ giúp tìm kiếm và thao tác trên chuỗi văn bản. Regex giúp bạn xác định các mẫu (patterns) phức tạp để tìm kiếm trong văn bản, chẳng hạn như kiểm tra định dạng email, số điện thoại, hay trích xuất thông tin từ văn bản lớn.

Biểu thức chính quy có ứng dụng rộng rãi trong hầu hết các ngôn ngữ lập trình và hệ thống xử lý dữ liệu, từ việc lọc văn bản đến phân tích dữ liệu. Trong bài viết này, chúng ta sẽ tìm hiểu chi tiết về regex và cách sử dụng nó trong Python.

Bạn có thể thực hành các biểu thức chính quy tại: [Regex](https://regex101.com/)

Thực hành Python về Regex tại: [Python](https://www.w3schools.com/python/python_regex.asp)

## 2. Các toán tử cơ bản
- **Tìm kiếm thông thường**: Bạn có thể tìm kiếm một xâu con trong một từ bằng cách đơn giản là gõ xâu con bạn muốn tìm kiếm. Ví dụ như câu:
> Con chó đang tung tăng vui vẻ trên đường.
- Khi bạn truy cập vào [Regex](https://regex101.com/) và gõ xâu con: "Con chó" ở phần Regular Expression và gõ câu ví dụ ở phần Test String, bạn sẽ nhận được câu sau: 
> <u>Con chó</u> đang tung tăng vui vẻ trên đường.
- Dưới đây là code python cho ví dụ trên:

```python
import re
txt = "Con chó đang tung tăng vui vẻ trên đường."
x = re.search("Con chó", txt)

if x:
  print("Có từ Con chó trong câu trên")
else:
  print("Không có từ Con chó trong câu trên")

```
> Output: Có từ Con chó trong câu trên

- Lưu ý là biểu thức chính quy sẽ tìm ra tất cả các mẫu mà khớp với đầu vào, ví dụ như đầu vào là kí tự "t" thì nó sẽ tìm tất cả các kí tự "t". Câu sau khi nhập từ "t" sẽ trở thành:
> Con chó đang <u>t</u>ung <u>t</u>ăng vui vẻ <u>t</u>rên đường.
- Regular Expression phân biệt dấu hoa dấu thường, cho nên nếu mẫu là "Con chó" sẽ không khớp với chuỗi "con chó" trong câu. Để giải quyết vấn đề hoa thường bạn có thể sử dụng ngoặc [] cho việc lựa chọn, lúc này mẫu sẽ trở thành: [Cc]on chó.  Đây là ví dụ 2 câu khác nhau sau khi bạn nhập mẫu "[Cc]on chó":
> <u>Con chó</u> đang tung tăng vui vẻ trên đường.

> <u>con chó</u> đang tung tăng vui vẻ trên đường.

- Kết luận dấu [] ngoặc này cho ta sự lựa chọn các ký tự nằm trong dấu. Ví như mẫu "[aàảãáạ]nh" sẽ chi ra kết quả:
> <u>anh</u> <u>ảnh</u> <u>ánh</u> <u>ạnh</u> <u>ãnh</u> <u>ành</u>
- Mẫu [1235] sẽ cho kết quả:
> <u>1</u> cộng <u>1</u> bằng <u>2</u> <u>2</u> thêm <u>2</u> bằng <u>4</u>
- Mẫu [12356789] có hạn chế về việc khoảng rộng cho nên ta có thể sử dụng dấu "-" để tìm trong đoạn dài ví dụ như mẫu "[1-9]" sẽ trả về kết quả:
> Số pi có giá trị bằng: <u> 3.141592653 </u>
- Ký hiệu ^ đứng đầu mẫu trong ngoặc vuông có ý nghĩa là không phải. Ví dụ [^a-z] có nghĩa là không phải các ký tự chữ cái. Ví dụ như mẫu "[a-z]" sẽ trả về kết quả (lưu ý là mẫu này sẽ không khớp được các chữ cái tiếng việt có dấu):
> 54 <u>d</u>ự á<u>n</u>, <u>t</u>ổ<u>ng</u> <u>c</u>ô<u>ng</u> <u>su</u>ấ<u>t</u> 10.521 MW
- [^a-z] trả về kết quả là các kí tự không nằm trong a-z (tức là các kí tự còn lại ở câu trên):
> <u>54 </u>d<u>ự </u><u>á</u>n, t<u>ổ</u>ng c<u>ô</u>ng su<u>ấ</u>t <u>10.521 MW</u>
- "[^A-Z]": trả về các ký tự không nằm trong đoạn [A-Z]
- "[^Ss]": trả về các ký tự không phải S hoặc s
- "[^.]": trả về không phải dấu chấm
- "[e^]": trả về e hoặc ^. Trường hợp này ^ không đứng đầu mẫu nên không có tính chất phủ định

- **Ký hiệu số lượng**:
  - Ký hiệu "?"
    - Ý nghĩa: Không hoặc có một ký tự phía trước ký hiệu ?.
    - Ví dụ a?m có thể tìm ra <u>m</u> hoặc <u>am</u>.
    - phu? có thể tìm ra <u>phu</u> hoặc <u>ph</u>
  
  - Ký hiệu "*"
    - Ý nghĩa: Không hoặc có nhiều ký tự phía trước ký hiệu *.
    - Ví dụ a*m có thể tìm ra <u>m</u>, <u>am</u>, <u>aam</u>, <u>aaam</u> hoặc <u>aaaaaam</u>.
    - Tuy nhiên a* sẽ khớp với chuỗi rỗng.
    - Phức tạp hơn ta sẽ có [em]* có nghĩa Không có hoặc có nhiều e hoặc nhiều m. Khớp với <u>eeee</u>, <u>mmmm</u>, <u>emememem</u>, thậm chí <u>eeeemmmm</u> và cuối cùng là chuỗi rỗng.
  - Ký hiệu "+"
    - Ý nghĩa: Có một hoặc nhiều ký tự phía trước ký hiệu + (Ít nhất một)
    - Ví dụ a+m sẽ không khớp với m vì bắt buộc ít nhất phải có một a. Mẫu này khớp với <u>am</u>, <u>aam</u>, <u>aaam</u> hoặc <u>aaaaaam</u>. 
  - Ký hiệu "."
    - Ý nghĩa: Đại diện cho bất cứ ký hiệu nào.
    - Ví dụ .oc có thể khớp <u>hoc</u>, <u>doc</u>, <u>coc</u>.
    - Nâng cao: Vậy một chuỗi bất kỳ sẽ được viết thế nào. Bạn có thể sử dụng mẫu ".*"

## 3. Tổng kết
- Còn rất nhiều các mẫu khác bạn có thể tìm tại [Python](https://www.w3schools.com/python/python_regex.asp), bài viết này chỉ giới thiệu qua về kĩ thuật biểu thức chính quy. 
- Vậy chúng ta sử dụng **Regex** khi nào ? Như các bạn đã biết, để huấn luyện một mô hình AI thứ quan trọng nhất là dữ liệu, dữ liệu phải nhiều và phong phú và phải **"sạch"** để mô hình có thể dễ huấn luyện. Ví dụ như trong quá trình crawl data từ một trang web nào đó (phần crawl data tôi sẽ có một bài viết riêng) thì các đoạn text thường bị dính các kí tự đặc biệt hoặc có nhiều khoảng trắng thừa thãi... Và để loại bỏ hay nói cách khác là làm sạch dữ liệu thì chúng ta sẽ phải sử dụng Regex. Đây là ví dụ một đoạn code làm sạch văn bản:

```python
import re
import string

def clean_text(text):
  """
  Hàm này làm sạch văn bản bằng cách loại bỏ các thành phần không mong muốn như:
  * URL: https://example.com, www.example.com
  * Thẻ HTML: <p>...</p>, <b>...</b>
  * Dấu câu: .,?!;... (trừ dấu apostrophe ')
  * Ký tự xuống dòng: \n
  * Từ chứa số: alphanumeric123, word456 (tùy chọn)
  * Khoảng trắng thừa: nhiều khoảng trắng liên tiếp

  **Tham số:**
  * text: Chuỗi văn bản cần làm sạch.

  **Trả về:**
  * Chuỗi văn bản đã được làm sạch.
  """

  # Chuyển đổi tất cả chữ cái thành chữ thường để dễ xử lý
  text = text.lower()

  # Loại bỏ các URL (https:// hoặc www.)
  text = re.sub(r'https?://\S+|www\.\S+', '', text)
 

  # Loại bỏ các thẻ HTML
  text = re.sub('<.*?>+', '', text)


  # Loại bỏ dấu câu (trừ dấu apostrophe)
  text = re.sub(r'[^\w\s\']', '', text)
  # Loại bỏ ký tự xuống dòng
  text = re.sub('\n', '', text)


  # Loại bỏ khoảng trắng thừa
  text = re.sub(r'\s+', ' ', text)
  return text
text = "RT   @user: This  ## &  is a #great tweet    with     a https://example.com link! 123"
cleaned_text = clean_text(text)
print(cleaned_text)
```
>Output: rt user this is a great tweet with a link 


## 4. Tài liệu tham khảo
- [1] [Biểu thức chính quy - ProtonX](https://protonx.coursemind.io/courses/66b0895e02b79700126975cd/topics/66badefb58f9530012731a4b?activeAId=66badefb58f9530012731a59)
- [2] [Speech and Language Processing Book](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)
