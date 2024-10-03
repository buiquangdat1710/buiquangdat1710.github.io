---
title: "Các kĩ thuật tách từ (Tokenizer) Phần 2"
date: 2024-10-01 00:00:00  + 0800
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

## 1. Lập trình tách từng từ một.
- Đầu tiên ta tạo một class có tên là SimpleTokenizer, class này có các phương thức:
  - add_word: Thêm một từ vào vocab, lưu ý là khi khởi tạo thì vocab đã được thêm hai từ là <PAD> đại diện cho từ được thêm vào để các câu có độ dài bằng nhau và <UNK> đại diện cho từ không có trong vocab.
  - tokenize: Phương thức để tách các từ.
  - convert_tokens_to_ids: Chuyển các từ sang dạng ids.
  - convert_ids_to_tokens: Chuyển các ids sang dạng từ.

```python
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.add_word('<PAD>')  # Padding token
        self.add_word('<UNK>')  # Unknown word token # OOV

    def add_word(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)

    def tokenize(self, text):
        return [word if word in self.vocab else '<UNK>' for word in text.split()]

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        reverse_vocab = {id: word for word, id in self.vocab.items()}
        return [reverse_vocab[id] for id in ids]

```

- Tiếp theo ta lấy ví dụ biến sentences chứa 4 câu và ta lần lượt thực hiện các phương thức:

```python
sentences = [
             'I love Vietnam',
             'Vietnamese people are pretty friendly',
             'My mom loves cooking',
             'I am Vietnamese'
]

tokenizer = SimpleTokenizer()

for sentence in sentences:
    for word in sentence.split():
        tokenizer.add_word(word)

print("Vocabulary:", tokenizer.vocab)

# Tokenizing a sentence
sentence = "I love Vietnam so"
tokens = tokenizer.tokenize(sentence)
print("Tokens:", tokens)

# Converting tokens to IDs
ids = tokenizer.convert_tokens_to_ids(tokens)
print("IDs:", ids)

# Converting back
tokens_back = tokenizer.convert_ids_to_tokens(ids)
print("Tokens from IDs:", tokens_back)
```
> Vocabulary: {'\<PAD\>': 0, '\<UNK\>': 1, 'I': 2, 'love': 3, 'Vietnam': 4, 'Vietnamese': 5, 'people': 6, 'are': 7, 'pretty': 8, 'friendly': 9, 'My': 10, 'mom': 11, 'loves': 12, 'cooking': 13, 'am': 14}

> Tokens: ['I', 'love', 'Vietnam', '\<UNK\>']

> IDs: [2, 3, 4, 1]

> Tokens from IDs: ['I', 'love', 'Vietnam', '\<UNK\>']

- Tiếp theo, ta sẽ thử lập trình tách từ sử dụng Pytorch. Đầu tiên ta cần phải cài đặt đúng thư viện torch và torchtext:

`pip install torch`

`pip install -U torchtext==0.12.0`

Dưới đây là đoạn code tách từng từ sử dụng Pytorch, các đoạn code đã được tôi thêm phần comment cho dễ hiểu:
```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Các câu mẫu
sentences = [
    'I love Vietnam',
    'Vietnamese people are pretty friendly',
    'My mom loves cooking',
    'I am Vietnamese'
]

# Bước 1: Khởi tạo bộ tách từ (tokenizer)
tokenizer = get_tokenizer('basic_english')

# Bước 2: Các token đặc biệt
unk_token = '<unk>'  # Token cho từ không xác định (dành cho những từ không có trong từ điển)
pad_token = '<pad>'  # Token để đệm (sử dụng để căn chỉnh độ dài các chuỗi)
bos_token = '<bos>'  # Token bắt đầu câu (đánh dấu bắt đầu của câu)
eos_token = '<eos>'  # Token kết thúc câu (đánh dấu kết thúc của câu)

# Bước 3: Xây dựng từ điển từ các câu đã được tách từ
vocab = build_vocab_from_iterator(
    map(tokenizer, sentences),  # Áp dụng tokenizer lên từng câu
    specials=[unk_token, pad_token, bos_token, eos_token]  # Bao gồm các token đặc biệt trong từ điển
)

# Bước 4: Thiết lập chỉ số mặc định cho các từ không xác định
vocab.set_default_index(vocab[unk_token])  # Nếu một từ không có trong từ điển, nó sẽ được thay thế bằng token <unk>
```
- Giờ hãy in thử ra vocab:
```python
print(vocab.get_stoi())
```

>{'pretty': 15,

>'\<unk\>': 0,

>'love': 10,

>'loves': 11,

>'vietnamese': 5,

>'vietnam': 16,

>'friendly': 9,

>'\<bos\>': 2,

>'\<eos\>': 3,

>'people': 14,

>'\<pad\>': 1,

>'i': 4,

>'my': 13,

>'are': 7,

>'am': 6,

>'cooking': 8,

>'mom': 12}

- Thử bạn có thể thấy thì Pytorch sẽ lưu từ dưới dang chữ thường và không theo một thứ tự nào cả. Hãy in thêm một số câu lệnh nữa:
```python
print(vocab['love'])
print(vocab.lookup_indices(['i', 'love', 'vietnamese']))
print(vocab['junkong']) # không có từ này trong vocab
```

> 10

> [4, 10, 5]

> 0

- Tiếp theo ta sẽ thử lập trình tách từ bằng thư viện Tensorflow:

```python
from tensorflow.keras.preprocessing.text import Tokenizer  # Import lớp Tokenizer từ tensorflow.keras để xử lý văn bản.

# Các câu mẫu với một số từ được lặp lại nhiều lần.
sentences = [
    'I I I I I I love Vietnam Vietnam Vietnam Vietnam',
    'Vietnamese people are pretty friendly',
    'My My My My My My My My mom loves cooking',
    'I am Vietnamese'
]

# Khởi tạo bộ tách từ (Tokenizer)
# num_words=100: Giới hạn số lượng từ được đưa vào từ điển (ở đây là tối đa 100 từ).
# oov_token="<OOV>": Sử dụng token "<OOV>" để đại diện cho những từ không nằm trong từ điển (out-of-vocabulary).
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")

# Huấn luyện tokenizer trên các câu trong danh sách 'sentences'
# Tokenizer sẽ đếm tần suất xuất hiện của các từ và tạo từ điển chỉ mục cho các từ.
tokenizer.fit_on_texts(sentences)

# Lấy ra từ điển ánh xạ giữa từ và chỉ số tương ứng (word -> index)
# Các từ sẽ được ánh xạ với các chỉ số dựa trên tần suất xuất hiện, từ có tần suất cao hơn sẽ có chỉ số thấp hơn.
word_index = tokenizer.word_index
print(word_index)
```
>{'<OOV>': 1,

>'my': 2,

>'i': 3,

>'vietnam': 4,

>'vietnamese': 5,

>'love': 6,

>'people': 7,

>'are': 8,

>'pretty': 9,

>'friendly': 10,

>'mom': 11,

>'loves': 12,

>'cooking': 13,

>'am': 14}

- Các từ có tần suất xuất hiện cao hơn  được gán chỉ số thấp hơn trong Tensorflow. Đây là một cơ chế rất hay vì trong nhiều thuật toán xử lý ngôn ngữ tự nhiên (NLP), thứ tự ưu tiên của các từ rất quan trọng. Dưới đây là lý do và lợi ích của việc gán chỉ số thấp hơn cho các từ có tần suất xuất hiện cao hơn:
  - Tối ưu hóa không gian bộ nhớ:
  
    - Trong nhiều mô hình NLP, các từ có chỉ số thấp sẽ được xử lý trước hoặc ưu tiên khi phân bổ tài nguyên (như bộ nhớ hoặc cache). Bằng cách gán chỉ số thấp hơn cho các từ phổ biến, mô hình có thể tối ưu hóa việc lưu trữ và truy cập những từ này một cách hiệu quả hơn.

  - Tối ưu hóa tốc độ tính toán:

    - Những từ phổ biến như "I", "the", "is", "and" thường xuất hiện rất nhiều trong các văn bản. Gán chỉ số thấp cho những từ này có thể giúp mô hình xử lý chúng nhanh hơn, bởi các phép toán như embedding (gắn nhãn số cho từ) thường xử lý các chỉ số nhỏ nhanh hơn các chỉ số lớn.

  - Tầm quan trọng trong việc học từ:

    - Trong một số mô hình học sâu (deep learning), như các mô hình dựa trên RNN, LSTM, hoặc Transformer, các từ có tần suất cao thường có ý nghĩa quan trọng để nắm bắt cấu trúc của ngôn ngữ (ví dụ, các từ ngữ chức năng như "is", "the", "of" giúp định hình cú pháp). Bằng cách gán chỉ số thấp hơn cho những từ này, mô hình có thể học các mối quan hệ giữa các từ phổ biến một cách nhanh chóng hơn.
  
- Dưới đây là một số đoạn code nữa sử dụng thư viện Tensorflow:
```python
new_sentences = [
    'I love dog',
    'I live in Hanoi'
]
new_sequences = tokenizer.texts_to_sequences(new_sentences)
print(new_sequences) # 2 câu có độ dài khác nhau
from tensorflow.keras.preprocessing.sequence import pad_sequences
padding_sequences = pad_sequences(new_sequences) # 2 câu có độ dài bằng nhau sau khi được thêm id pad ở đầu
print(padding_sequences)
```

> [[3, 6, 1], [3, 1, 1, 1]]

> array([[0, 3, 6, 1], [3, 1, 1, 1]], dtype=int32)


Đây là ảnh so sánh giữa hai framework Tensorflow và Pytorch được lấy từ khóa học NLP của ProtonX:

![Mô tả hình ảnh](image/PytorchvsTensorflow.png)

## 2. Thuật toán tách token Byte-Pair Encoding (BPE).
- BPE là một thuật toán, được Philip Gage mô tả lần đầu tiên vào năm 1994 để mã hóa các chuỗi văn bản thành dạng bảng để sử dụng trong mô hình hóa hạ lưu, hay nói cách khác, BPE là thuật toán dùng để tách token. Ý tưởng của BPE rất đơn giản, đầu tiên ta sẽ tách từng từ một, sau đó nhóm 2 từ liên tiếp lại với nhau và đếm tần suất xuất hiện giữa chúng. Sau đó ta sẽ tìm cặp từ có tần suất xuất hiện nhiều nhất và thay thế nó với 2 từ liên tiếp ban đầu. Ta làm thuật toán thế trong K lần, sau K lần thì ta sẽ có các nguyên tắc để tách token.
- Ví dụ như dữ liệu của ta là:

> cam_cam_cam_cam_cam_

> nham_nham_nham_nham_

> tam_tam_tam_

> cam_can_can_can_can_can_

> ham_ham_ham_han_

- Sau bước tách từng kí tự một, ta sẽ có được:

> c,a,m,\_,c,a,m,\_,c,a,m,\_,c,a,m,\_,c,a,m,\_

> n,h,a,m,\_,n,h,a,m,\_,n,h,a,m,\_,n,h,a,m,\_  

> t,a,m,\_,t,a,m,\_,t,a,m,\_
  
> c,a,m,\_,c,a,n,\_,c,a,n,\_,c,a,n,\_,c,a,n,\_,c,a,n,\_
  
> h,a,m,\_,h,a,m,\_,h,a,m,\_,h,a,n,\_

- Sau đó ta đếm các cặp hai kí tự liên tiếp nhau:

> 'am': 16, 'm_': 16, 'ca': 11, 'ha': 8, 'an': 6, 'n_': 6, 'nh': 4, 'ta': 3

- Ta thấy kí tự 'am' là xuất hiện nhiều nhất, ta sẽ thay kí tự 'am' này như một kí tự. khi đó thì các kí tự có thể biểu diễn như sau:

> c,am,\_,c,am,\_,c,am,\_,c,am,\_,c,am,\_

> n,h,am,\_,n,h,am,\_,n,h,am,\_,n,h,am,\_  

> t,am,\_,t,am,\_,t,am,\_
  
> c,am,\_,c,a,n,\_,c,a,n,\_,c,a,n,\_,c,a,n,\_,c,a,n,\_
  
> h,am,\_,h,am,\_,h,am,\_,h,a,n,\_

- Cứ tiếp lúc làm các bước như thế trong K lần nữa. Dưới đây là thuật toán BPE được code bằng python từ đầu:

```python
# Tạo danh sách 'words' chứa các danh sách con từ các từ cụ thể: 'cam_', 'nham_', 'tam_', 'can_', 'ham_'
words = [list("cam_")] * 5 + [list("nham_")] * 4 + [list("tam_")] * 3 + [list("can_")] * 6 + [list("ham_")] * 4

# Tạo một tập hợp để lưu trữ các ký tự duy nhất từ danh sách 'words'
aSet = set()
for word in words:
    for ch in word:
        aSet.add(ch)  # Thêm từng ký tự vào tập hợp aSet

# Import Counter từ thư viện collections để đếm các cặp ký tự
from collections import Counter

# Hàm đếm các cặp ký tự liên tiếp phổ biến nhất trong danh sách 'words'
def most_common(words):
    # Tạo một Counter để đếm các cặp ký tự liên tiếp
    counter = Counter()
    # Duyệt qua từng từ trong danh sách words
    for word in words:
        # Duyệt qua các ký tự trong từ, ngoại trừ ký tự cuối cùng
        for i in range(len(word) - 1):
            # Tạo cặp ký tự liên tiếp và tăng số đếm cho cặp đó
            counter["{}{}".format(word[i], word[i+1])] += 1
    # In ra tất cả các cặp ký tự và số lần xuất hiện của chúng
    print(counter)
    # Trả về cặp ký tự phổ biến nhất và số lần xuất hiện của nó
    return counter.most_common(1)

# Hàm hợp nhất cặp ký tự phổ biến nhất thành một ký tự trong danh sách 'words'
def merge_tokens(words, token):
    token_to_merge = token  # Cặp ký tự phổ biến nhất được truyền vào để hợp nhất
    # Duyệt qua từng từ trong danh sách words
    for i in range(len(words)):
        word = words[i]
        # Duyệt qua từng cặp ký tự trong từ
        for j in range(len(word) - 1):
            # Kiểm tra nếu cặp ký tự hiện tại khớp với cặp ký tự cần hợp nhất
            if "{}{}".format(word[j], word[j+1]) == token_to_merge:
                # Tạo từ mới với cặp ký tự đã hợp nhất
                new_word = word[:j] + [token_to_merge] + word[j+2:]
                words[i] = new_word  # Cập nhật từ trong danh sách words
    return words

# Số lần lặp để thực hiện quá trình hợp nhất
num = 3
# Lặp lại quá trình tìm cặp ký tự phổ biến nhất và hợp nhất chúng
for i in range(num):
    # Tìm cặp ký tự phổ biến nhất
    common_tokens_and_cnt = most_common(words)
    # Hợp nhất cặp ký tự đó trong danh sách words
    words = merge_tokens(words, common_tokens_and_cnt[0][0])

# In ra danh sách words sau khi hợp nhất cặp ký tự
print(words)

```

> Counter({'am': 16, 'm_': 16, 'ca': 11, 'ha': 8, 'an': 6, 'n_': 6, 'nh': 4, 'ta': 3})

>Counter({'am_': 16, 'ham': 8, 'ca': 6, 'an': 6, 'n_': 6, 'cam': 5, 'nh': 4, 'tam': 3})

>Counter({'ham_': 8, 'ca': 6, 'an': 6, 'n_': 6, 'cam_': 5, 'nh': 4, 'tam_': 3})

>[['c', 'am_'],

>['c', 'am_'],

>['c', 'am_'],

>['c', 'am_'],

>['c', 'am_'],

>['n', 'ham_'],

>['n', 'ham_'],


>['n', 'ham_'],

>['n', 'ham_'],

>['t', 'am_'],

>['t', 'am_'],

>['t', 'am_'],

>['c', 'a', 'n', '_'],

>['c', 'a', 'n', '_'],

>['c', 'a', 'n', '_'],

>['c', 'a', 'n', '_'],

>['c', 'a', 'n', '_'],

>['c', 'a', 'n', '_'],

>['ham_'],

>['ham_'],

>['ham_'],

>['ham_']]

- Chúng ta sẽ lập trình đầy đủ hơn như sau:
  
```python
# -*- coding: utf-8 -*-
# Chỉ định rằng file mã nguồn này được mã hóa bằng UTF-8.

from collections import Counter
# Nhập lớp Counter từ module collections để đếm tần suất của các cặp ký tự.

# Danh sách các từ ban đầu
words = [list("cam_")] * 5 + [list("nham_")] * 4 + [list("tam_")] * 3 + [list("can_")] * 6 + [list("ham_")] * 4
# Định nghĩa danh sách các từ ban đầu dưới dạng danh sách các ký tự. Phép nhân sẽ nhân bản từng từ.
# Ví dụ, "cam_" xuất hiện 5 lần, "nham_" xuất hiện 4 lần, v.v.

# Hàm tìm cặp ký tự thông dụng nhất
def most_common(words_):
    counter = Counter()
    # Tạo một đối tượng Counter để theo dõi tần suất của các cặp ký tự.

    for word in words_:
        # Duyệt qua từng từ (là danh sách các ký tự).

        for i in range(len(word) - 1):
            # Duyệt qua các ký tự trong từng từ, trừ ký tự cuối cùng (để tránh lỗi tràn chỉ số).

            counter["{}{}".format(word[i], word[i+1])] += 1
            # Với mỗi cặp ký tự liền kề trong từ, tạo cặp ký tự và tăng tần suất của nó trong Counter.

    return counter.most_common(1)
    # Trả về cặp ký tự phổ biến nhất.

# Khởi tạo một tập rỗng để lưu các ký tự duy nhất
aSet = set()
# Tạo một tập hợp rỗng tên 'aSet' để lưu các ký tự duy nhất từ tất cả các từ.

# Thêm từng ký tự từ danh sách các từ ban đầu vào 'aSet'
for word in words:
    for ch in word:
        aSet.add(ch)
# Duyệt qua từng từ và từng ký tự trong từ.
# Thêm ký tự vào 'aSet', đảm bảo mỗi ký tự chỉ xuất hiện một lần trong tập hợp.

# Hiển thị tập hợp các ký tự duy nhất
print(f"Initial Set of Unique Characters: {aSet}")
# In ra tập hợp các ký tự duy nhất thu thập được trong 'aSet'.

# Khởi tạo bpe_vocab dưới dạng từ điển với tần suất của từng ký tự
bpe_vocab = {char: 0 for char in aSet}
# Tạo một từ điển 'bpe_vocab', trong đó mỗi ký tự từ 'aSet' là một khóa và tần suất ban đầu được đặt là 0.

# Hàm trích xuất từ vựng BPE (Byte Pair Encoding) theo thứ tự tần suất
def extract_bpe_vocab(words, num_iterations=10):
    from collections import Counter
    # Định nghĩa một hàm để trích xuất từ vựng BPE.
    # 'num_iterations' mặc định là 10, kiểm soát số lần lặp lại của thuật toán BPE.

    # Thực hiện các vòng lặp BPE
    for _ in range(num_iterations):
        # Lặp lại quá trình sau 'num_iterations' lần:

        # Tìm cặp ký tự thông dụng nhất
        counter = Counter()
        # Khởi tạo một Counter để theo dõi tần suất của các cặp ký tự trong vòng lặp hiện tại.

        for word in words:
            for i in range(len(word) - 1):
                token_pair = "{}{}".format(word[i], word[i+1])
                # Với mỗi từ, tạo cặp ký tự liền kề (tokens).

                counter[token_pair] += 1
                # Tăng tần suất của mỗi cặp ký tự.

        # Nếu không tìm thấy cặp ký tự nào nữa, thoát khỏi vòng lặp
        if not counter:
            break
        # Nếu không còn cặp ký tự nào trong Counter, thoát khỏi vòng lặp sớm.

        # Lấy cặp ký tự phổ biến nhất
        most_frequent_pair, frequency = counter.most_common(1)[0]
        # Lấy cặp ký tự phổ biến nhất và tần suất của nó từ Counter.

        # Thêm cặp ký tự phổ biến nhất vào bpe_vocab với tần suất của nó
        bpe_vocab[most_frequent_pair] = frequency
        # Lưu cặp ký tự phổ biến nhất vào 'bpe_vocab' với tần suất của nó.

        # Gộp cặp ký tự phổ biến nhất trong tất cả các từ
        words = merge_tokens(words, most_frequent_pair)
        # Thay thế các cặp ký tự phổ biến nhất trong 'words' bằng hàm 'merge_tokens'.

    # Trả về bpe_vocab dưới dạng từ điển đã được sắp xếp theo tần suất giảm dần
    bpe_vocab_sorted = dict(sorted(bpe_vocab.items(), key=lambda item: -item[1]))
    # Sắp xếp 'bpe_vocab' theo tần suất giảm dần và trả về từ điển đã sắp xếp.

    return bpe_vocab_sorted

# Hàm gộp các cặp ký tự dựa trên cặp ký tự phổ biến nhất
def merge_tokens(words, token):
    token_to_merge = token
    # Lưu cặp ký tự phổ biến nhất thành 'token_to_merge'.

    for i in range(len(words)):
        word = words[i]
        # Duyệt qua từng từ trong 'words'.

        for j in range(len(word) - 1):
            # Duyệt qua các ký tự trong từ, trừ ký tự cuối cùng.

            if "{}{}".format(word[j], word[j + 1]) == token_to_merge:
                # Nếu một cặp ký tự liền kề trong từ khớp với 'token_to_merge':

                new_word = word[:j] + [token_to_merge] + word[j + 2:]
                # Thay thế các ký tự liền kề bằng cặp ký tự 'token_to_merge'.

                words[i] = new_word
                # Cập nhật từ với cặp ký tự mới được gộp.

    return words
    # Trả về danh sách các từ đã được cập nhật.

# Trích xuất từ vựng BPE đã được sắp xếp theo tần suất
bpe_vocab = extract_bpe_vocab(words)
# Gọi hàm 'extract_bpe_vocab' để thực hiện thuật toán BPE và nhận từ vựng BPE đã được sắp xếp.

print(f"BPE Vocabulary (Token and Frequency): {bpe_vocab}")
# In ra từ vựng BPE và tần suất của chúng.

# Tạo ánh xạ từ token sang chỉ số dựa trên từ vựng BPE đã được sắp xếp theo tần suất
token_to_index = {token: idx for idx, token in enumerate(bpe_vocab.keys())}
# Tạo ánh xạ từ mỗi token trong 'bpe_vocab' đến chỉ số của nó trong từ điển đã sắp xếp.

print(f"Token to Index Mapping: {token_to_index}")
# In ra ánh xạ từ token sang chỉ số.
```

> Initial Set of Unique Characters: {'c', 'n', 'h', 't', '_', 'm', 'a'}

> BPE Vocabulary (Token and Frequency): {'am': 16, 'am_': 16, 'ham_': 8, 'ca': 6, 'can': 6, 'can_': 6, 'cam_': 5, 'nham_': 4, 'tam_': 3, 'c': 0, 'n': 0, 'h': 0, 't': 0, '_': 0, 'm': 0, 'a': 0}

>  Token to Index Mapping: {'am': 0, 'am_': 1, 'ham_': 2, 'ca': 3, 'can': 4, 'can_': 5, 'cam_': 6, 'nham_': 7, 'tam_': 8, 'c': 9, 'n': 10, 'h': 11, 't': 12, '_': 13, 'm': 14, 'a': 15}

- Tiếp theo ta sẽ xem xét cách thuật toán BPE tách token một câu đầu vào, trước tiếp ta sẽ lập trình hàm encode:
  
```python
# Hàm mã hóa một câu cho trước bằng cách sử dụng từ vựng BPE đã học, trả về các chỉ số dạng số
def encode_to_numbers(sentence, bpe_vocab, token_to_index):
    # Chuyển đổi câu đầu vào thành danh sách các ký tự riêng lẻ
    words = list(sentence)

    # Bắt đầu một vòng lặp vô hạn để liên tục gộp các tokens cho đến khi không còn có thể gộp nữa
    while True:
        merged = False  # Biến cờ để theo dõi xem có gộp nào xảy ra trong vòng lặp này không

        # Duyệt qua từng token trong từ vựng BPE được sắp xếp theo tần suất giảm dần
        for token in sorted(bpe_vocab.keys(), key=lambda k: -bpe_vocab[k]):
            token_len = len(token)  # Lấy độ dài của token hiện tại

            # Bỏ qua việc xử lý nếu token chỉ có một ký tự
            if token_len == 1:
                continue

            # Duyệt qua các ký tự trong 'words', kiểm tra sự xuất hiện của 'token'
            for i in range(len(words) - token_len + 1):

                # Kiểm tra xem chuỗi con từ vị trí 'i' có khớp với token hiện tại không
                if "".join(words[i:i+token_len]) == token:
                    # Nếu tìm thấy sự trùng khớp, gộp token vào danh sách 'words'
                    words = words[:i] + [token] + words[i+token_len:]
                    print(f"Gộp token '{token}' tại vị trí {i}: {words}")  # Câu lệnh in để debug

                    # Đặt 'merged' thành True vì đã có gộp xảy ra và thoát khỏi vòng lặp bên trong
                    merged = True
                    break

            # Nếu có một gộp đã xảy ra, thoát khỏi vòng lặp bên ngoài để khởi động lại với 'words' đã cập nhật
            if merged:
                break

        # Nếu không có gộp nào được thực hiện trong vòng lặp này, thoát khỏi vòng lặp
        if not merged:
            break

    print('---token_to_index', token_to_index)
    # Chuyển danh sách cuối cùng của các tokens trong 'words' thành các chỉ số số tương ứng
    encoded_numbers = [token_to_index[token] for token in words if token in token_to_index]

    # In ra các tokens cuối cùng đã được gộp và mã hóa dạng số của chúng để debug
    print('--- Các từ đã được gộp cuối cùng:', words)
    print('--- Dạng mã hóa số:', encoded_numbers)

    # Trả về danh sách các chỉ số số biểu diễn câu đã được mã hóa
    return encoded_numbers

```

- Sau đây là một ví dụ về tách token:

```python

# Example usage
sentence = "cam cam nham tam can ham"
sentence = sentence.replace(" ", "_")

print('--sentence', sentence)

encoded_numbers = encode_to_numbers(sentence, bpe_vocab, token_to_index)
print(f"Encoded Sentence as Numbers: {encoded_numbers}")
```

> --sentence cam_cam_nham_tam_can_ham

>Merging token 'am' at position 1: ['c', 'am', '\_', 'c', 'a', 'm', '\_', 'n', 'h', 'a', 'm', '\_', 't', 'a', 'm', '\_', 'c', 'a', 'n', '\_', 'h', 'a', 'm']

>Merging token 'am' at position 4: ['c', 'am', '\_', 'c', 'am', '\_', 'n', 'h', 'a', 'm', '\_', 't', 'a', 'm', '\_', 'c', 'a', 'n', '\_', 'h', 'a', 'm']

>Merging token 'am' at position 8: ['c', 'am', '\_', 'c', 'am', '\_', 'n', 'h', 'am', '\_', 't', 'a', 'm', '\_', 'c', 'a', 'n', '\_', 'h', 'a', 'm']

>Merging token 'am' at position 11: ['c', 'am', '\_', 'c', 'am', '\_', 'n', 'h', 'am', '\_', 't', 'am', '\_', 'c', 'a', 'n', '\_', 'h', 'a', 'm']

>Merging token 'am' at position 18: ['c', 'am', '\_', 'c', 'am', '\_', 'n', 'h', 'am', '\_', 't', 'am', '\_', 'c', 'a', 'n', '\_', 'h', 'am']

>Merging token 'ca' at position 13: ['c', 'am', '\_', 'c', 'am', '\_', 'n', 'h', 'am', '\_', 't', 'am', '\_', 'ca', 'n', '\_', 'h', 'am']

>---token_to_index {'am': 0, 'am_': 1, 'ham_': 2, 'ca': 3, 'can': 4, 'can_': 5, 'cam_': 6, 'nham_': 7, 'tam_': 8, 'c': 9, 'n': 10, 'h': 11, 't': 12, '_': 13, 'm': 14, 'a': 15}

>--- Final merged words: ['c', 'am', '\_', 'c', 'am', '\_', 'n', 'h', 'am', '\_', 't', 'am', '\_', 'ca', 'n', '\_', 'h', 'am']

>--- Encoded numeric representation: [9, 0, 13, 9, 0, 13, 10, 11, 0, 13, 12, 0, 13, 3, 10, 13, 11, 0]

>Encoded Sentence as Numbers: [9, 0, 13, 9, 0, 13, 10, 11, 0, 13, 12, 0, 13, 3, 10, 13, 11, 0]

- Đoạn code sau để giải mã từ ids sang một câu:

```python
# Function to decode a list of numeric indices back into the original sentence
def decode_from_numbers(encoded_numbers, index_to_token):
    # Map each numeric index back to its corresponding token
    decoded_tokens = [index_to_token[idx] for idx in encoded_numbers]

    # Join the list of tokens to form the decoded sentence
    decoded_sentence = "".join(decoded_tokens)

    # Return the final decoded sentence
    return decoded_sentence

# Create the index-to-token mapping from the token_to_index dictionary
index_to_token = {idx: token for token, idx in token_to_index.items()}

# Example usage: Decode the previously encoded numbers
decoded_sentence = decode_from_numbers(encoded_numbers, index_to_token)
decoded_sentence = decoded_sentence.replace("_", " ")
print(f"Decoded Sentence: {decoded_sentence}")
```
> Decoded Sentence: cam cam nham tam can ham

- Một điều thú vị là chat GPT bạn vẫn hay sử dụng hàng ngày dùng BPE để làm thuật toán tách token. Giờ chúng ta hãy thử tách token bằng model GPT thông qua Hugging Face. Đầu tiên chúng ta sẽ tải vocab của gpt2 từ câu lệnh terminal dưới đây:
  
`wget https://huggingface.co/gpt2/raw/main/vocab.json`

- Tiếp theo chúng ta sẽ load vocab bằng câu lệnh dưới:

```python
import json

# open the file in read mode with UTF-8 encoding
with open('vocab.json', 'r', encoding='utf-8') as file:
    # load the JSON data into a variable
    data = json.load(file)
```

- Bạn có thể in thử ra data để biết nó nhiều như nào, đây là một từ điển chứa hơn 50000 kí tự đã được ánh xạ sang ids. Giờ chúng ta sẽ load tokenizer bằng thư viện transformers:

```python
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "AI is the best thing ever !"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
```
> {'input_ids': tensor([[20185,   318,   262,  1266,  1517,  1683,  5145]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}

- Lưu ý ở đây output của ta là một từ điển, khóa thứ nhất chính là input_ids biểu thị cho ids của từng token, khóa thứ hai chính là attention_mask biểu thị cho token có phải được padding vào không, nếu ids là 0 thì có nghĩa là đây là token đc padding vào (thường được biểu hiện là <PAD>), nếu là 1 thì token ban đầu. 
- Để xem câu của chúng ta được chia token như thế nào thì ta có thể giải mã ngược lại, từ ids sang dạng text:

```python
import itertools
arr = encoded_input["input_ids"].tolist()
flat_arr = list(itertools.chain(*arr))
tokenizer.convert_ids_to_tokens(flat_arr)
```
> ['AI', 'Ġis', 'Ġthe', 'Ġbest', 'Ġthing', 'Ġever', 'Ġ!']

- Lưu ý là GPT sử dụng kí hiệu Ġ thay cho khoảng trắng, tiếp theo chúng ta đến với thuật toán tách token nữa là **WordPiece**.
  
## 3. Thuật toán tách token WordPiece.

- Chúng ta sẽ cùng nghiên cứu một thuật toán có tên WordPiece. Một mô hình rất nổi tiếng dùng thuật toán này đó chính là **BERT** - [Bidirectional Encoder Representations from Transformers](https://arxiv.org/pdf/1810.04805).
- Tách từ thành các ký tự:
  - Ví dụ trong mô hình Bert, một từ "ProtonX" sẽ được tách thành các ký tự và thêm ## vào phía trước:
  > ProtonX
  - Sau khi tách sẽ trở thành:
  > P ##r ##o ##t ##o ##n ##X
  - Lưu ý: Chỉ riêng từ ở đầu là không thêm "##"
- Điểm khác biệt với thuật toán BPE:
  - Thuật toán BPE đếm số cặp hai token xuất hiện liền nhau và lưu token hợp của chúng vào trong từ điển. Thuật toán WordPiece cũng đếm tuy nhiên thay vì chỉ lựa chọn cặp token xuất hiện nhiều nhất, thuật toán này sẽ thêm yếu tố tần suất của từng ký tự trong cặp này.

  - Công thức sẽ như sau:

    - Điểm (score) = (Số lần xuất hiện của cặp token) / (Số lần xuất hiện của token 1 x Số lần xuất hiện của token 2)

  - Các cặp token có điểm càng cao sẽ ưu tiên hợp (merge) trước.

  - Ý nghĩa của việc này là gì nhỉ ? Khi nhìn vào công thức chúng ta sẽ thấy nếu chúng ta có hai cặp token:

    >"g ##a"

    >Cặp 2: "g #ấu"

  - Nếu số lượng xuất hiện của cả hai cặp này tuơng đương nhau, bộ học token (Token learner) sẽ ưu tiên cặp "g ##ấu" hơn vì token "##ấu" trong cặp này sẽ xuất hiện ít hơn nhiều so với token "##a" trong cặp "g ##a". Cả hai cặp token cùng chung token đầu "g".

  - Thậm chí cặp 1 cũng không cần hợp vì đơn giản là hai token xuất hiện rất nhiều trong bộ từ điển.
  
### Thuật toán 

- Sau đây, chúng ta sẽ đi đến thuật toán chính của WordPiece, giờ hãy lấy một ví dụ đơn giản, giả sử dữ liệu ban đầu của ta là:

> ga ga ga ga ga 
 
> gấu gấu gấu gấu gấu gấu

> gan gan gan gan gan gan gan gan

> gấm gấm gấm gấm gấm gấm gấm

> ha ha ha

- Chúng ta sẽ có tập từ điển sau:

> '##a', '##m', '##n', '##u', '##ấ', 'g', 'h'

- Trong ngữ liệu này:

  - Token "g" xuất hiện 5 + 6 + 7 + 8 = 26 lần

  - Token "##a" xuất hiện 5 + 8 + 3 = 16 lần

  - Token "##ấ" xuất hiện 6 + 7 = 13 lần

  - Cặp token "g ##a" xuất hiện 5 + 8 = 13 lần

  - Cặp token "g ##ấ" xuất hiện 6 + 7 = 13 lần

- Điểm của cặp "g ##a" = (Số lần xuất hiện của cặp "g ##a") / (Số lần xuất hiện của token "g" x Số lần xuất hiện của token "##a") = 13 / (26 x 16) = 0.03125

- Điểm của cặp "g ##ấ" = (Số lần xuất hiện của cặp "g ##ấ") / (Số lần xuất hiện của token "g" x Số lần xuất hiện của token "##ấ") = 13 / (26 x 13) = 0.0385

- Trong trường hợp này điểm của cặp "g ##ấ" lớn hơn nên sẽ ưu tiên hợp cặp này vào từ điển hơn.

- Chúng ta sẽ nói qua về cách hợp của WordPiece:
  - Nếu ta hợp "g" và "##ấ" ta sẽ bỏ "##" trước "##ấ" ta sẽ có "gấ".
  - Nếu ta hợp "##ấ" và "##m" ta sẽ bỏ "##" trước "##m" ta sẽ có "##ấm".
  - Sau đó token hợp này sẽ được thêm vào từ điển.

- Chúng ta sẽ di đến một ví dụ phức tạp hơn để hiểu rõ về cách hoạt động của WordPiece, giờ giả sử dữ liệu của chúng ta gồm hai câu:

> "ProtonX là một công ty AI",

> "ProtonX là một nơi ươm mầm tài năng AI"

- Bộ đếm:
  
> 'ProtonX': 2,

> 'là': 2,

> 'một': 2,

> 'công': 1,

> 'ty': 1,

> 'AI': 2,

> 'nơi': 1,

> 'ươm': 1,

> 'mầm': 1,

> 'tài': 1,

> 'năng': 1

- Những ký tự đứng đầu câu thì không thêm '##'.

- Những ký tự đứng sau ký tự đầu đều thêm '##' vào phía trước.

Ví dụ với từ "ProtonX" ta có kết quả như sau:

>'ProtonX': ['P', '##r', '##o', '##t', '##o', '##n', '##X']

Kết quả ta có trên cả hai câu như sau:

> ['##I', '##X', '##g', '##i', '##m', '##n', '##o', '##r', '##t', '##y', '##à', '##ô', '##ă', '##ơ', '##ầ', '##ộ', 'A', 'P', 'c', 'l', 'm', 'n', 't', 'ư']

- Ta bắt đầu đi tính điểm cho các cặp token cạnh nhau trong cách từ:

>('P', '##r'): 0.5

>('##r', '##o'): 0.25

>('##o', '##t'): 0.125

>('##t', '##o'): 0.125

>('##o', '##n'): 0.125

>('##n', '##X'): 0.25

>('l', '##à'): 0.3333333333333333

>('m', '##ộ'): 0.3333333333333333

>('##ộ', '##t'): 0.25

>('c', '##ô'): 1.0

>('##ô', '##n'): 0.25

>('##n', '##g'): 0.25

>('t', '##y'): 0.5

>('A', '##I'): 0.5

>('n', '##ơ'): 0.25

>('##ơ', '##i'): 0.25

>('ư', '##ơ'): 0.5

>('##ơ', '##m'): 0.25

>('m', '##ầ'): 0.3333333333333333

>('##ầ', '##m'): 0.5

>('t', '##à'): 0.16666666666666666

>('##à', '##i'): 0.16666666666666666

>('n', '##ă'): 0.5

>('##ă', '##n'): 0.25

- Cặp token:

> ('c', '##ô'): 1.0

- Có điểm cao nhất hơn hẳn cặp ('A', '##I') mặc dù cặp AI xuất hiện hai lần trong ngữ liệu nhưng chỉ có điểm là (2) / (2 x 2) = 0.5.

- Điểm thấp nhất là 3 cặp:

> ('##o', '##t'): 0.125
> ('##t', '##o'): 0.125
> ('##o', '##n'): 0.125

- Ví dụ cặp ('##o', '##t') có điểm (2) / (4 x 4) = 0.125. Cặp ('c', '##ô') được hợp thành "cô" và thêm vào từ điển. Từ "công" sẽ được biểu diễn như sau sau lần hợp này:

> 'công': ['cô', '##n', '##g']

- Sau một số lượng vòng lặp cài đặt sẵn ta sẽ thu được bộ từ điển sau:

> ['##I', '##X', '##g', '##i', '##m', '##n', '##o', '##r', '##t', '##y', '##à', '##ô', '##ă', '##ơ', '##ầ', '##ộ', 'A', 'P', 'c', 'l', 'm', 'n', 't', 'ư', 'cô', 'Pr', 'ty', 'AI', 'ươ', 'nơ', 'nă', 'nơi', 'ươm', '##ầm', 'là', 'tà', 'tài', 'mộ', 'mầm', 'Pro', 'Prot', 'Proto', 'một', 'Proton', 'ProtonX', 'côn', 'năn', 'công', 'năng'] 

###  Quá trình mã hóa (encode)

- Với bộ từ điển trên thì với đầu vào câu:

> "Thả tym cho ProtonX nào"

- sau khi tách token sẽ có kết quả là:

> ['[UNK]', 'ty', '##m', '[UNK]', 'ProtonX', 'n', '##à', '##o']

- Từ "Thả" tính từ đầu (cố định vị trí đầu) đến cuối không chứa bất kỳ từ con (subword) nào nằm bên trong từ điển nên sẽ được kết luận là từ không biết với token "[UNK]".

- Cách lặp:

    - "T" không nằm trong từ điển

    - "Th" không nằm trong từ điển

    - "Thả" không nằm trong từ điển

- Cho nên kết luận "Thả" là "[UNK]"

- Từ "tym" được tách thành token "ty" và "##m"

    - Từ đầu đến "y" từ "ty" là một token nằm trong từ điển. Tiến hành tách khỏi từ ban đầu ra thành "ty" và phần còn lại là "m". Thêm "##" vào "m" thành "##m" và tiếp tục tách.

    - "##m" nằm trong từ điển và tách thành chính nó.
- Vậy cách mã hóa ở đây sẽ là gì ?
  - Chúng ta sẽ đặt mốc cố định là đầu chuỗi, tìm xem từ đầu đến vị trí nào chứa token dài nhất trong vocab thì ta bắt đầu tách token. Phần đằng sau sẽ thêm "##" vào phía trước.

  - Tiếp tục tách phần sau.

  - Trường hợp nào từ đầu đến đuôi của phần chuỗi đang làm việc không hề nằm trong từ điển, trả về token [UNK]

## 4. Lập trình thuật toán WordPiece

- Chúng ta sẽ thử lập trình thuật toán WordPiece từ đầu, trước tiên chúng ta sẽ tạo corpus như sau:
  
```python
from collections import defaultdict
corpus = ["ga"] * 5 + ["gấu"] * 6 + ["gan"] * 8 + ["gấm"] * 7 + ["ha"] * 3
```
- Tiếp theo chúng ta sẽ đếm tần suất xuất hiện các từ trong câu:

```python
# Khởi tạo từ điển
word_freqs = defaultdict(int)

# Lặp qua ngữ liệu
for word in corpus:
    # Đếm từ
    word_freqs[word] += 1

print(word_freqs)
```
> defaultdict(int, {'ga': 5, 'gấu': 6, 'gan': 8, 'gấm': 7, 'ha': 3})

- Tiếp theo chúng ta sẽ tạo Splits cho từng từ:

```python
def generate_splits(word_freqs):
    """
    Tạo các splits cho mỗi từ trong dictionary word_freqs.

    Mỗi từ được chia thành các ký tự riêng lẻ, trong đó ký tự đầu tiên được giữ nguyên 
    và các ký tự tiếp theo được thêm tiền tố '##'.

    Ví dụ:
    Giả sử`word_freqs = {'cat': 5, 'dog': 3}

    Kết quả sẽ là:
    {
        'cat': ['c', '##a', '##t'],
        'dog': ['d', '##o', '##g']
    }

    Tham số:
        word_freqs (dict): Một dictionary với các từ là key và tần suất của chúng là value.

    Trả về:
        dict: Một dictionary với các từ là key và danh sách các split là value.
    """
    splits = {}

    # Lặp qua các từ trong word_freqs
    for word in word_freqs:
        # Chia từ thành các splits
        split_list = [word[0]] + [f"##{char}" for char in word[1:]]
        splits[word] = split_list

    return splits
splits = generate_splits(word_freqs)
print(splits)

```
>{'ga': ['g', '##a'],

>'gấu': ['g', '##ấ', '##u'],

>'gan': ['g', '##a', '##n'],

>'gấm': ['g', '##ấ', '##m'],

>'ha': ['h', '##a']}

- Chúng ta sẽ đếm số lượng ký tự trong ngữ liệu:
  
```python
from collections import defaultdict

def compute_letter_frequencies(splits, word_freqs):
    """
    Tính tần suất xuất hiện của các ký tự dựa trên các splits đã cho và tần suất từ.

    Ví dụ:
    Giả sử:
    splits = {
        'cat': ['c', '##a', '##t'],
        'dog': ['d', '##o', '##g']
    }

    word_freqs = {
        'cat': 5,
        'dog': 3
    }

    Kết quả sẽ là:
    {
        'c': 5,
        '##a': 5,
        '##t': 5,
        'd': 3,
        '##o': 3,
        '##g': 3
    }

    Tham số:
        splits (dict): Dictionary chứa các từ và danh sách các ký tự đã được split.
        word_freqs (dict): Dictionary chứa các từ và tần suất tương ứng của chúng.

    Trả về:
        defaultdict(int): Một dictionary với các ký tự là key và tần suất của chúng là value.
    """
    letter_freqs = defaultdict(int)

    # Lặp qua các từ và lấy tần suất
    for word, freq in word_freqs.items():
        # Lấy danh sách các splits cho mỗi từ
        split = splits[word]
        
        # Đếm số lần xuất hiện của các ký tự này
        for letter in split:
            letter_freqs[letter] += freq

    return letter_freqs

# Ví dụ chạy hàm
letter_freqs = compute_letter_frequencies(splits, word_freqs)
print(letter_freqs)

```
>defaultdict(int,
            {'g': 26,
             '##a': 16,
             '##ấ': 13,
             '##u': 6,
             '##n': 8,
             '##m': 7,
             'h': 3})
            
- Chúng ta sẽ tính tần suất xuất hiện của các cặp ký tự dựa trên các splits đã cho:
  
```python

def compute_pair_frequencies(splits, word_freqs):
    """
    Tính tần suất xuất hiện của các cặp ký tự dựa trên các splits đã cho và tần suất từ.

    Ví dụ:
    Giả sử:
    splits = {
        'cat': ['c', '#a', '#t'],
        'dog': ['d', '#o', '#g']
    }

    word_freqs = {
        'cat': 5,
        'dog': 3
    }

    Kết quả sẽ là:
    {
        ('c', '#a'): 5,
        ('#a', '#t'): 5,
        ('d', '#o'): 3,
        ('#o', '#g'): 3
    }

    Tham số:
    splits (dict): Dictionary chứa các từ và danh sách các ký tự đã được split.
    word_freqs (dict): Dictionary chứa các từ và tần suất tương ứng của chúng.

    Trả về:
    defaultdict(int): Một dictionary với các cặp ký tự là key và tần suất của chúng là value.
    """
    pair_freqs = defaultdict(int)  # Tạo dictionary mặc định với giá trị là số nguyên

    # Lặp qua các phần tử của word_freqs
    for word, freq in word_freqs.items():
        # Lấy splits của từ hiện tại
        split = splits.get(word, [])
        
        # Kiểm tra nếu splits chứa nhiều hơn 1 phần tử thì mới tính cặp ký tự
        if len(split) > 1:
            # Lặp qua từng cặp của splits
            for i in range(len(split) - 1):
                # Lấy cặp ký tự hiện tại
                pair = (split[i], split[i + 1])
                
                # Cập nhật số lượng cặp ký tự với tần suất của từ
                pair_freqs[pair] += freq

    return pair_freqs
pair_freqs = compute_pair_frequencies(splits, word_freqs)
print(pair_freqs)
```
> defaultdict(int,
            {('g', '##a'): 13,
             ('g', '##ấ'): 13,
             ('##ấ', '##u'): 6,
             ('##a', '##n'): 8,
             ('##ấ', '##m'): 7,
             ('h', '##a'): 3})

- Tiếp theo chúng ta sẽ lập trình hàm tính scores theo công thức ở trên:
  
```python
def calculate_pair_scores(pair_freqs, letter_freqs):
    """
    Tính điểm số cho mỗi cặp ký tự dựa trên tần suất của cặp và tần suất của từng ký tự,
    đảm bảo tránh chia cho 0.

    Tham số:
        pair_freqs (dict): Dictionary chứa các cặp ký tự và tần suất tương ứng của chúng.
        letter_freqs (dict): Dictionary chứa các ký tự và tần suất tương ứng của chúng.

    Trả về:
        dict: Một dictionary với các cặp ký tự là key và điểm số là value.
    """
    scores = {}
    for pair, freq in pair_freqs.items():
        # Lấy số lượng token 1 trong ngữ liệu
        token1, token2 = pair
        freq1 = letter_freqs.get(token1, 0)
        # Lấy số lượng token 2 trong ngữ liệu
        freq2 = letter_freqs.get(token2, 0)
        # Kiểm tra tránh chia cho 0
        if freq1 > 0 and freq2 > 0:
            # Áp dụng công thức bên trên
            scores[pair] = freq / (freq1 * freq2)
    return scores
```
- Kết hợp các hàm bên trên:
  
```python
def compute_pair_scores(splits, word_freqs):
    letter_freqs = compute_letter_frequencies(splits, word_freqs)
    pair_freqs = compute_pair_frequencies(splits, word_freqs)
    scores = calculate_pair_scores(pair_freqs, letter_freqs)
    return scores


def display_pair_scores(pair_scores, max_display=10):
    for i, (key, score) in enumerate(pair_scores.items()):
        print(f"{key}: {score}")
        if i == max_display - 1:
            break

# Example usage
pair_scores = compute_pair_scores(splits, word_freqs)
display_pair_scores(pair_scores)
```
>('g', '##a'): 0.03125

>('g', '##ấ'): 0.038461538461538464

>('##ấ', '##u'): 0.07692307692307693

>('##a', '##n'): 0.0625

>('##ấ', '##m'): 0.07692307692307693

- Ta sẽ tìm cặp có score cao nhất và lưu lại:
  
```python
best_pair = ""
max_score = None
for pair, score in pair_scores.items():
    if max_score is None or max_score < score:
        best_pair = pair
        max_score = score

print(best_pair, max_score)
```
> ('##ấ', '##u') 0.07692307692307693

- Ta sẽ lập trình hàm hợp 2 tokens:

```python
def merge_tokens(a, b):
    """
    Gộp hai token thành một dựa trên quy tắc BPE.

    Tham số:
        a (str): Token đầu tiên.
        b (str): Token thứ hai.

    Trả về:
        str: Token sau khi đã gộp.
    """
    # Hợp hai token bất kỳ
    if b.startswith("##"):
        return a + b[2:]  # Bỏ đi "##" ở token b và nối vào token a
    else:
        return a + b
def process_split(split, a, b):
    """
    Xử lý một split để gộp hai token liên tiếp a và b.

    Tham số:
        split (list): Danh sách các token cần xử lý.
        a (str): Token đầu tiên cần gộp.
        b (str): Token thứ hai cần gộp.

    Trả về:
        list: Danh sách các token sau khi đã gộp.
    """
    i = 0
    while i  <  len(split) - 1:
        if split[i] == a and split[i + 1] == b:
            merged_token = merge_tokens(a, b)
            split = split[:i] + [merged_token] + split[i + 2:]
        else:
            i += 1
    return split
def merge_pair(a, b, splits, word_freqs):
    """
    Lặp qua tất cả các từ và gộp cặp (a, b) trong các splits của chúng.

    Tham số:
        a (str): Token đầu tiên trong cặp cần gộp.
        b (str): Token thứ hai trong cặp cần gộp.
        splits (dict): Dictionary chứa các từ và danh sách các token đã được split.
        word_freqs (dict): Dictionary chứa các từ và tần suất tương ứng của chúng.

    Trả về:
        dict: Dictionary chứa các từ với danh sách các token đã được gộp.
    """
    for word in word_freqs:
        split = splits[word]
        if len(split) > 1:
            splits[word] = process_split(split, a, b)
    return splits
```
- Giờ ta sẽ đóng gói tất cả mọi thứ lại trong một hàm như sau:

```python
def build_vocab(vocab_size, splits, word_freqs):
    """
    Xây dựng từ vựng dựa trên tần suất của các cặp token.

    Tham số:
        vocab_size (int): Số lượng token tối đa cho từ vựng.
        splits (dict): Dictionary chứa các từ và danh sách các token đã được split.
        word_freqs (dict): Dictionary chứa các từ và tần suất tương ứng của chúng.

    Trả về:
        list: Danh sách các token trong từ vựng đã được xây dựng.
    """

    # Lấy các ký tự trong ngữ liệu
    alphabet = []
    for word in word_freqs.keys():
        if word[0] not in alphabet:
            alphabet.append(word[0])
        for letter in word[1:]:
            if f"##{letter}" not in alphabet:
                alphabet.append(f"##{letter}")

    alphabet.sort()

    vocab = alphabet
    print(vocab)


    # Tính tần suất xuất hiện của từng ký tự trong các splits
    letter_freqs = compute_letter_frequencies(splits, word_freqs)

    # Lặp lại quá trình cho đến khi số lượng token trong từ vựng đạt đến giới hạn 'vocab_size'
    while len(vocab) < vocab_size:
        # Tính điểm số cho từng cặp ký tự
        scores = compute_pair_scores(splits, word_freqs)
        print(scores)

        # Tìm cặp token có điểm số cao nhất
        best_pair, max_score = "", None
        for pair, score in scores.items():
            if max_score is None or max_score < score:
                best_pair = pair,
                max_score = score

        # Nếu không còn cặp token nào để gộp, thoát khỏi vòng lặp
        if len(best_pair) == 0:
            break

        # Gộp cặp token tốt nhất vào các splits
        token1, token2 = best_pair[0][0], best_pair[0][1]
        splits = merge_pair(token1, token2, splits, word_freqs)

        # Tạo token mới sau khi gộp và thêm vào từ vựng
        new_token = merge_tokens(best_pair[0][0],best_pair[0][1])
        vocab.append(new_token)

    return vocab

# Ví dụ sử dụng hàm vừa tạo
final_vocab = build_vocab(60, splits, word_freqs)
print(final_vocab)
```
> ['##a',
 '##m',
 '##n',
 '##u',
 '##ấ',
 'g',
 'h',
 '##ấu',
 '##ấm',
 '##an',
 'ha',
 'ga',
 'gấu',
 'gan',
 'gấm']

- Giờ là phần cuối cùng, chúng ta sẽ lập trình để chia token:

```python
def find_longest_token(word, vocab):
    """
    Tìm token dài nhất có trong từ vựng phù hợp với từ đầu vào.

    Tham số:
        word (str): Từ cần tìm token.
        vocab (set): Tập hợp các token trong từ vựng.

    Trả về:
        str: Token dài nhất được tìm thấy hoặc None nếu không tìm thấy token.
    """
    for i in range(len(word), 0, -1):  # Bắt đầu từ token dài nhất có thể
        token_candidate = word[:i]  # Lấy phần đầu của từ
        if token_candidate in vocab:
            return token_candidate
    return None

def encode_subword(word):
    """
    Thêm tiền tố '##' vào phần của từ chưa được mã hóa.

    Tham số:
        word (str): Phần từ cần thêm tiền tố.

    Trả về:
        str: Phần từ với tiền tố '##'.
    """
    return  f"##{word}"

def encode_word(word, vocab):
    """
    Mã hóa một từ thành các token dựa trên từ vựng đã có.

    Tham số:
        word (str): Từ cần mã hóa.
        vocab (set): Tập hợp các token trong từ vựng.

    Trả về:
        list: Danh sách các token sau khi mã hóa. Trả về ["[UNK]"] nếu từ không thể mã hóa.
    """
    tokens = []  # Khởi tạo danh sách các token

    while len(word) > 0:
        token = find_longest_token(word, vocab)

        if token is None:
            return ["[UNK]"]  # Trả về ["[UNK]"] nếu không tìm thấy token phù hợp

        # Thêm token đã tìm thấy vào danh sách
        tokens.append(token)

        # Cập nhật từ bằng cách loại bỏ phần token đã mã hóa
        word = word[len(token):]

        # Nếu còn phần từ chưa mã hóa, thêm tiền tố '##'
        if len(word) > 0:
            word = encode_subword(word)

    return tokens
print(encode_word("haấu", final_vocab))
```  

> ['ha', '##ấu']

- Giờ hãy thử sử dụng thuật toán WordPiece thông qua -thư viện HuggingFace. Đầu tiên, chúng ta se tải vocab của mô hình BERT:

`wget https://huggingface.co/bert-base-uncased/raw/main/tokenizer.json`

- Tiếp theo chúng ta load vocab bằng câu lệnh dưới:

```python
import json

# open the file in read mode with UTF-8 encoding
with open('tokenizer.json', 'r', encoding='utf-8') as file:
    # load the JSON data into a variable
    data = json.load(file)
```

- Giờ hãy thử tách token một câu văn bản:

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "unhappyness housewife"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
```
> {'input_ids': tensor([[  101, 12511,  2791,  2160, 19993,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}

- input_ids và attention_mask đã được tôi giải thích ở trên. Khóa token_type_ids để phân biệt hai câu văn bản. Văn bản thứ nhất sẽ được đánh dấu là 0 đối với token thuộc văn bản thứ nhất, token thuộc văn bản thứ hai sẽ đánh dấu là một. Dưới đây là ví dụ gồm 2 câu văn bản:

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_1 = "AI is the future"
text_2 = "Robots will assist humans"
encoded_input = tokenizer(text_1, text_2, return_tensors='pt')
print(encoded_input)

```
> {
    'input_ids': tensor([[  101,  9932,  2003,  1996,  2920,   102,  12365,  2097,  5026,  4286,   102]]),
    'token_type_ids': tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}

- Để mã hóa từ ids sang token, chúng ta dùng câu lệnh bên dưới:

```python
import itertools
arr = encoded_input["input_ids"].tolist()
flat_arr = list(itertools.chain(*arr))
tokenizer.convert_ids_to_tokens(flat_arr)
```
> ['[CLS]', 'unhappy', '##ness', 'house', '##wife', '[SEP]']

- Như các bạn thấy thì nó có thêm token [CLS] để báo hiệu bắt đầu câu và token [SEP] để kết thúc câu.

## 5. Tài liệu tham khảo.
- [1] [Biểu thức chính quy - ProtonX](https://protonx.coursemind.io/courses/66b0895e02b79700126975cd/topics/66badf03e1f5ad00195e901b?activeAId=66badf0a58f9530012731ae6)