---
title: "Các kĩ thuật tách từ (Tokenizer) Phần 1"
date: 2024-09-30 00:00:00  + 0800
categories: [NLP]
tags: [Token]
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

**Tách từ (Tokenizer)** là quá trình chia văn bản thành các đơn vị nhỏ hơn, thường là từ hoặc ký tự, để dễ dàng xử lý trong các bài toán ngôn ngữ tự nhiên (NLP). Kỹ thuật này giúp mô hình máy học hiểu và phân tích văn bản một cách có cấu trúc, từ đó hỗ trợ các tác vụ như phân loại, dịch thuật, hoặc trích xuất thông tin hiệu quả hơn.

## 1. Giới thiệu
- Trong bài toán xử lý ngôn ngữ tự nhiên, sau khi qua bước làm **sạch**, hay nói cách khác sau khi áp dụng biểu thức chính quy hóa lên văn bản, chúng ta sẽ phải **tách token**, tức là chia văn bản thành những đoạn nhỏ.
- Cụ thể hơn, tách token (tokenizer) là nhiệm vụ tách câu thành những đơn vị nhỏ, những đơn vị nhỏ này gọi là **token**.
- Các đơn vị token này sẽ thuộc một trong bốn loại dưới:
  - Tách từng kí tự:
    - Ví dụ như câu:
    > Tôi đi học 
    - Sẽ được tách thành:
    > T\|ô\|i\| \|đ\|i\| \|h\|ọ\|c   
  - Tách thành các cụm kí tự, đây là cách thường hay được sử dụng nhất:
    - Ví dụ:
    > Tô\|i đ\|i \|họ\|c
  - Tách thành các từ:
    - Ví dụ:
    > Tôi\| đi \| học
  - Tách thành cụm các từ:
    - Ví dụ:
    > Tôi đi \| học

## 2. Mô hình N-gram:
- N-grams là mô hình dự đoán token kế tiếp dựa trên N token trước đó. Mô hình này sẽ tính tần suất xuất hiện của các cặp N token liên tiếp rồi dự đoán token tiếp theo dựa trên phân bố xác suất. Với phần này thì ta sẽ coi token như một kí tự, bạn có thể hoàn toàn coi token như một từ.
- Một số mô hình N-grams phổ biến:
  - **unigram**: mô hình với n=1, tức là ta sẽ tính tần suất xuất hiện của một kí tự (từ), như: "k", "a",…
  - **bigrams**: mô hình với n=2 , là mô hình được sử dụng nhiều trong việc phân tích các hình thái cho ngôn ngữ
  - **trigrams**: với n = 3, với n càng lớn thì độ chính xác càng cao tuy nhiên đi kèm với đó thì độ phức tạp cũng lớn hơn.
- Lấy ví dụ như với mô hình **bigrams**, ta sẽ tính phải tính số lần xuất hiện của các cặp "ab", "ac", "ad", ... "yz", tức là số lần xuất hiện của mọi cặp có thể. Ta thêm vào các câu kí tự "." ở đầu dòng và cuối dòng để xác định đây là kí tự bắt đầu và kí tự kết thúc. 
- Điểu hiểu rõ hơn cách **bigrams** hoạt động, giả tử ta có câu văn sau:
>  am dat and i fell happy when learning ai
- Ta sẽ thêm kí tự "." vào đầu và cuối câu, bạn có thể thêm kí tự khác kí tự "." cũng được, nói chung là kí tự này chỉ để xác định điểm đầu và điểm cuối của câu:
> .i am dat and i fell happy when learning ai.
- Tiêp theo, ta sẽ phải tính số lần xuất hiện của các cặp có 2 kí tự liên tiếp trong câu, lưu ý khoảng trắng cũng coi là 1 kí tự, ở dưới tôi ghi kí kiệu "/_" tượng trưng cho khoảng trắng:

| Cặp 2 kí tự liên tiếp | Số lần xuất hiện |
|-----------------------|------------------|
| .i                    | 1                |
| i_                    | 2                |
| _a                    | 3                |
| am                    | 1                |
| m_                    | 1                |
| _d                    | 1                |
| da                    | 1                |
| at                    | 1                |
| t_                    | 1                |
| an                    | 1                |
| nd                    | 1                |
| d_                    | 1                |
| _i                    | 1                |
| _f                    | 1                |
| fe                    | 1                |
| el                    | 1                |
| ll                    | 1                |
| l_                    | 1                |
| _h                    | 1                |
| ha                    | 1                |
| ap                    | 1                |
| pp                    | 1                |
| py                    | 1                |
| y_                    | 1                |
| _w                    | 1                |
| wh                    | 1                |
| he                    | 1                |
| en                    | 1                |
| n_                    | 1                |
| _l                    | 1                |
| le                    | 1                |
| ea                    | 1                |
| ar                    | 1                |
| rn                    | 1                |
| ni                    | 1                |
| in                    | 1                |
| ng                    | 1                |
| g_                    | 1                |
| ai                    | 1                |
| i.                    | 1                |

- Sau đó, để sinh ra một câu mới thì ta sẽ bắt đầu từ kí tự "." vì kí tự đó là kí tự bắt đầu một câu. Ta sẽ tìm xem sau kí tự "." thì kí tự nào có khả năng xuất hiện dựa vào tần suất xuất hiện các cặp: ".a" , ".b", ... , ".z", ".." . Lưu ý rằng ví dụ trên chỉ có cặp ".i" xuất hiện nhưng tôi chỉ lấy ví dụ một câu, trên thực tế có cả trăm nghìn câu như thế và mỗi câu dài hơn rất nhiều. Giờ cứ coi như cặp ".i" có tần suất cao nhất, tiếp theo ta lại tìm xem kí tự nào có khả năng xuất hiện sau từ "i" dựa vào tần suất xuất hiện các cặp: "ia" , "ib" , ... "iz" , "i." . Và cứ tiếp tục như thế đến khi ta chọn được kí tự ".", tương đương với việc kết thúc một câu. 

⚠️ **Lưu ý:** Thường thì mô hình N-grams sẽ không phải lúc nào cũng chọn cặp có tần xuất suất hiện cao nhất mà nó sẽ chọn dựa trên phân bố xác suất. Ví dự như cặp "ba" có xác suất xuất hiện là 30%, cặp "bi" có xác suất suất hiện là 15%, ... mô hình N-grams không phải lúc nào cũng sẽ chọn cặp "ba" mà có thể nó sẽ chọn cặp "bi". Thực ra điều này phụ thuộc vào cách bạn lập trình, nhưng chọn như trên sẽ làm tăng thêm tính phong phú của câu. Ví dụ như câu sau:

> Tôi đi

- Nếu mô hình tính cặp "đi học" là có xác suất xuất hiện cao nhất thì nó sẽ sinh ra câu "Tôi đi học" mãi mãi:

> Tôi đi học

- Nhưng nếu ta để mô hình chọn theo phân bố xác suất thì nó có thể sẽ sinh ra các câu:

> Tôi di học

> Tôi đi chơi

> Tôi đi ngủ

## 3. Lập trình mô hình N-grams để sinh ra tên người.
- Ta sẽ lập trình một mô hình N-grams đơn giản để sinh ra các tên người (tên tiếng anh), bạn hoàn toàn có thể áp dụng mô hình này để sinh ra các tên tiếng việt nếu bạn tìm được dữ liệu về tiếng việt.
- Đầu tiên, bạn hãy tải file dữ liệu chứa hơn 32000 tên tiếng anh: [Dữ liệu](https://github.com/karpathy/makemore/blob/master/names.txt)
- Tiếp theo ta sẽ mở file này và lưu vào biến words:
```python
words = open('names.txt', 'r').read().splitlines()
```
- Ta có thể xem qua 10 tên đầu tiên trong file này:
```python
words[:10]
```
> ['emma',
 'olivia',
 'ava',
 'isabella',
 'sophia',
 'charlotte',
 'mia',
 'amelia',
 'harper',
 'evelyn']
- Tiếp theo, ta tạo ra một bảng hai chiều N để lưu trữ các cặp 2 kí tự xuất hiện liên tiếp. Hiểu đơn giản là ta sẽ có 26 chữ cái tiếng anh, cộng thêm kí tự "." sẽ là 27 kí tự. Ta sẽ tạo ra bảng N có kích thước là 27x27, ta sẽ coi như hàng 0 sẽ là kí tự "." quản lý, tức là vị trí N[0][0] sẽ là số lần xuất hiện của cặp "..", vị trí N[0][1] sẽ là số lần xuất hiện của cặp ".a", ... , vị trí N[0][26] là số lần xuất hiện của cặp ".z". Tiếp tục, hàng 1 sẽ là các cặp mà có kí tự a ở đầu, vị trí N[1][0] sẽ là số lần xuất hiện của cặp "a." ... 
- Ban đầu ta sẽ khởi tạo bảng N bằng 0:
```python
import torch
N = torch.zeros((27, 27), dtype=torch.int32)
```
- Ta tạo ra biển chars để lưu các kí tự khác nhau trong dữ liệu, biến stoi là từ điển để ánh xạ từ kí tự sang số, biển itos là từ điển để ánh xạ từ số sang kí tự:
```python
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
```
- Tiếp theo, ta sẽ tính các tần xuất suất hiện của các cặp kí tự và lưu vào bảng N:
```python
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1
```
- Tiếp theo ta sẽ trực quan hóa bảng N:
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
plt.show()
```
![Alt text](image/n-grams.png)

- Tiếp theo ta sẽ sinh 5 tên dựa vào mô hình N-grams xem kết quả ra sao:
  
```python
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    p = N[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
```
> mor.,
axx.,
minaymoryles.,
kondlaisah.,
anchshizarie.,

- Như bạn thấy, kết quả khá tệ, tại sao lại thế ? Vì mô hình N-grams ta lập trình rất đơn giản và dữ liệu chưa đủ nhiều. Mô hình này chỉ dựa trên xác suất các cặp từ, nó còn thậm chí không có bước gọi là huấn luyện. Nhưng hãy tin rằng bạn vẫn đang đi đúng hướng, giờ hãy thử xem nếu mô hình dự đoán ngẫu nhiêu thì sao ? Hãy chạy thử đoạn code dưới:

```python
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    p = torch.ones(27) / 27.0
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
```

- Kết quả thậm chí còn tệ hơn rất nhiều. Có một điều cần cải thiện là chúng ta tính xác suất sau mỗi vòng lặp, điều này khá tốn thời gian. Chúng ta sẽ tạo ra một mảng P mới chứa xác suất như sau (Tạo sao lại là N+1 thì tôi sẽ giải thích ở phần 4):

```python
P = (N+1).float()
P /= P.sum(1, keepdims=True)
```

- Và đoạn code thay thế vòng lặp:

```python
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
``` 
  
## 4. Vẻ đẹp của toán học đằng sau mô hình N-grams.

- Trước khi đi đến phần toán, tôi có một lưu ý cho các bạn. Các xác suất trong bảng P chính là xác suất có điều kiện. Ví dự như vị trí P[1][2] sẽ lưu xác suất của cặp (a , b) và nó chính là p(a\|b) chứ không phải là p(ab) hay p(ab), lưu ý p ở đây chỉ xác suất chứ không phải biến p ở code. Tại vì như bạn thấy trên code, ta tính xác suất của cặp (a,b) bằng cách đếm số lần xuất hiện của cặp (a,b) và chia cho tổng số lần xuất hiện của các cặp (a,.), (a,b), (a,c) ,... (a,z) nên đây chính là xác suất khi ta biết trước a rồi chứ không phải xác suất a và b xuất hiện đồng thời.
- Vậy để tính xác suất của một tên người thì sẽ tính như nào ? Trước tiên thì ta sẽ đi đến một chút lý thuyết toán học nhẹ nhàng. Giả sử ta có $w_{1},w_{2},...,w_{n}$ là n kí tự ngẫu nhiên, khi đó xác suất của từ $w_{1}w_{2}...w_{n}$ là:
  
$$p(w_1 \dots w_n) = p(w_1) \cdot p(w_2 \mid w_1) \cdot p(w_3 \mid w_1 w_2) \cdot \dots \cdot p(w_n \mid w_1 \dots w_{n-1})  \tag{1}$$    

- Nhưng ta chỉ biết các xác suất có điền kiện gồm 2 kí tự như p(a \| b) thôi thì làm như nào, lúc đó ta sẽ sử dụng một phương trình ma thuật. Phương trình đó gọi là **Giả định Markov**.
- **Giả định Markov** cho ta một phương trình sau:
  
$$p(w_i \mid w_1 \dots w_{i-1}) = p(w_i \mid w_{i-n+1} \dots w_{i-1}) \tag{2}$$

- Phát biểu đơn giản thì phương trình trên nói là xác suất của kí tự $w_i$ không cần nhất thiết phải dựa vào i-1 kí tự đã biết mà chỉ cần dựa vào n - 1 kí tự đã biết gần nhất thôi là đủ. Điều này nghe vẻ không đúng lắm vì rõ ràng theo như công thức tính xác suất điều kiện thì vế trái với vế phải của phương trình là khác nhau. Giả định Markov không phải lúc nào cũng đúng, mà nó chỉ là một giả định được áp dụng cho một số loại mô hình ngẫu nhiên cụ thể. Giả định này đơn giản hóa việc tính toán và dự đoán trong các hệ thống, nhưng không phải lúc nào cũng phản ánh thực tế của mọi quá trình ngẫu nhiên.
- Giả định Markov khẳng định rằng trạng thái tương lai chỉ phụ thuộc vào trạng thái hiện tại, không phụ thuộc vào quá khứ. Điều này thường đúng với các mô hình được thiết kế dựa trên giả định này, chẳng hạn như chuỗi Markov, nơi các quá trình ngẫu nhiên được mô tả bởi các bước kế tiếp có quy tắc chuyển trạng thái cố định.
- Quay lại với phương trình giả định Markov, nếu ta chọn $n = 2$ thì phương trình sẽ thành:
  
$$p(w_i \mid w_1 \dots w_{i-1}) = p(w_i \mid w_{i-1}) \tag{3}$$

- Như vậy phương trình (1) sẽ trở thành:
  
$$p(w_1 \dots w_n) = p(w_1) \cdot p(w_2 \mid w_1) \cdot p(w_3 \mid w_2) \cdot \dots \cdot p(w_n \mid w_{n-1})  \tag{4}$$

- Thực tế thì ta sẽ phải thêm kí tự "." ở đầu và cuối của một từ nên từ lúc này sẽ trở thành "$.w_{1}w_{2}...w_{n}.$" và ta có công thức sau khi thêm kí tự "." là:

$$p(.w_1 \dots w_n.) = p(.) \cdot p(w_1 \mid .) \cdot p(w_2 \mid w_1) \cdot p(w_3 \mid w_2) \cdot \dots \cdot p(w_n \mid w_{n-1}) \cdot p(. \mid w_{n})   \tag{5}$$


- Theo như công thức trên thì p(.) sẽ được tính bằng cách đếm xem có bao nhiêu kí tự "." trong văn bản rồi chia cho số tất cả các kí tự. Nhưng nên một cách hợp lý thì ta nên tính xác suất các từ có thể xuất hiện ở đầu, vì ở đầu mỗi từ luôn luôn là kí tự "." nên ta có thể coi $p(.) = 1$. Phương trình lúc đó có thể viết gọn lại thành, ta coi $w_0$ và $w_{n+1}$ là kí tự ".":
 
$$p(w_1 \dots w_n) = \prod_{i=1}^{n+1} p(w_i \mid w_{i-1})  \tag{5}$$
  
- Gỉa sử từ $w_{1}w_{2}...w_{n}$ là một tên người thì chúng ta sẽ muốn xác suất này lớn nhất vì chúng ta muốn model sinh ra được từ này. Làm việc với các phép nhân sẽ có 2 vấn đề chính là tràn số trên (overflow) và tràn số dưới (underflow), tôi sẽ viết một blog khác nói riêng về vấn đề này. Nên một cách hợp lý, chúng ta sẽ chuyển phép nhân thành phép cộng bằng cách lấy log 2 vế:

$$log(p(w_1 \dots w_n)) = log(\prod_{i=1}^{n+1} p(w_i \mid w_{i-1})) = \sum_{i=1}^{n+1}log(p(w_i \mid w_{i-1}))  \tag{6}$$

- Có một vấn đề là xác suất của $p(w_i \mid w_{i-1})$ có thể bằng 0 nếu như cặp kí tự $(w_(i-1) , w_{i})$ chưa từng xuất hiện trong tập huấn luyện (training set), nếu như $p(w_i \mid w_{i-1}) = 0$ thì $log(p(w_i \mid w_{i-1}))$ sẽ tiến tới $\infty$. Để giải quyết vấn đề đó, chúng ta sẽ đảm bảo rằng tất cả các tần suất của mọi cặp từ đều dường, đó chính là lý do tại sao lại đặt P = (N+1).float(), N+1 để đảm bảo tất cả các giá trị trong N đều dương, bạn có thể cộng với số dương nào cũng được. Điều này không làm thay đổi thứ tự của các xác suất, kĩ thuật thêm một hằng số để các phần tử xác suất dương được gọi là **Laplace Smoothing**.

- Phương trình (6) được gọi là log likelihood, ta muốn maximize phương trình này, nhưng thường thì ta sẽ chuyển phương trình (6) sang dạng negative log likelihood bằng cách đổi dấu phương trình:

$$-log(p(w_1 \dots w_n)) = -\sum_{i=1}^{n+1}log(p(w_i \mid w_{i-1}))  \tag{7}$$

- Đến đây phương trình trông không khác gì hàm loss entropy trong softmax. Vậy một điều tự nhiên nhất là ta có thể nghĩ đến ý tưởng dùng Neural Network để làm bài toán này.

## 5. Từ xác suất điều kiện đến Neural Network.

- Ý tưởng khá là rõ ràng. Ta sẽ tạo tập huấn luyện bằng cách lấy x là từ thứ nhất và y là từ thứ hai. Đầu vào của Neural Network sẽ là kí tự x và đầu ra là xác suất của 27 cặp: (x , .) , (x, a) , ... (x , z). Đương nhiên ta sẽ phải mã hóa kí tự x dưới dạng số, ta sẽ dùng **one-hot encoding** để mã hóa vị trí. Ví dụ như kí tự a ta sẽ ánh xạ sang vị trí 1 thì khi đó one-hot của a sẽ có 27 phần từ đều bằng 0 ngoại trừ phần tử thứ 1 thì bằng 1. 
- Ta sẽ xây dựng một Neural Network đơn giản nhất thế giới, chỉ có một tầng ẩn. Đầu tiên ta sẽ chuyển hết tập huấn luyện sang dạng one-hot và ta kí hiệu là ma trận E, đặt n là số lượng dữ liệu:

$$E \in R^{n \times 27} \tag{8} $$

- Ma trận W chính là ma trận ma ta cần phải nhân khi đưa qua tầng ẩn, ta sẽ đặt ma trận W có kích thước 27*27. Phương trình dưới biểu diễn khi ta cho E qua tầng ẩn:

$$O = E*W , O \in R^{n \times 27} \tag{9} $$

- Lưu ý trong bài này tôi không thêm hệ số bias vì tôi muốn giữ nó đơn giản. Lúc này chúng ta có thể áp dụng Softmax trên từng hàng của ma trận O:

$$S_{i,j} = \frac{e^{O_{i,j}}}{\displaystyle \sum_{\text{j}} e^{O_{i,j}}} \in R^{n \times 27} \tag{10}$$

- Sau đó, chúng ta có hàm mất mát L tính bằng công thức sau, $\otimes$ là **tích Hadamard**:

$$L = -\sum_{\text{i}} \sum_{\text{j}} log(S \otimes Y) \tag{11}$$

- Oke, chúng ta hãy đi vào lập trình Neural Network, trước tiên chúng ta hãy tính thử xem loss của cách tiếp cận theo xác suất điều kiện ra bao nhiêu:

```python
log_likelihood = 0.0
n = 0

for w in words:
#for w in ["andrejq"]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')
```
>log_likelihood=tensor(-564996.8125, grad_fn=<AddBackward0>)
nll=tensor(564996.8125, grad_fn=<NegBackward0>)
2.476470470428467

- Loss ra 2.47, cũng không quá tệ. Lưu ý rằng chúng ta đang muốn hàm loss này càng thấp càng tốt. Giờ hãy tạo tập huấn luyện bằng cách xs để lưu các kí từ đầu tiên, ys là nhãn để lưu kí tự thứ hai.

```python
# create the dataset
xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
```

- Giờ chúng ta hãy huấn luyện mô hình bằng thuật toán đơn giản nhất trên đời là **Gradient Descent**:
```python
lr = 1
for k in range(100):
  
  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
  print(loss.item())
  
  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()
  
  # update
  W.data += -lr * W.grad
```

- Như bạn thấy kết quả tốt hơn cách đầu tiên một chút nhưng nhìn chung thì vẫn tệ. Thứ nhất vì mô hình này chỉ có 1 tầng ẩn, thứ hai là chưa có chính quy hóa trong hàm loss, thứ ba là mô hình này chưa có các hàm kích hoạt, thứ tư là mô hình này chỉ sử dụng Gradient Descent. Bạn có thể scale mô hình này nên và xem kết quả có cải thiện không nhé !
- Dưới đây là code sinh ra các từ mới do mô hình Neural Network:

```python
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    
    # ----------
    # BEFORE:
    #p = P[ix]
    # ----------
    # NOW:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
    # ----------
    
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
```
> mor.
axx.
minaymoryles.
kondlaisah.
anchthizarie.

- Trông có vẻ không khác gì mô hình chúng ta dùng xác suất điều kiện. Hãy thử nghiệm thêm nhiều thứ mới và gửi lại cho tôi kết quả tại email: [buiquangdat1458@gmail.com](https://mail.google.com/mail/u/0/#inbox) nhé.

## 6. Tài liệu tham khảo.
- [1] [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&t=3178s)