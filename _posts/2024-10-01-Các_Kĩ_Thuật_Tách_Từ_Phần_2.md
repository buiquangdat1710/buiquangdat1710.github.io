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

- Thuật toán WordPiece là một thuật toán chia nhỏ từ vựng thành các subword để giúp xử lý các từ chưa từng gặp và làm giảm kích thước từ vựng cần lưu trữ trong các mô hình ngôn ngữ. WordPiece nhằm mục tiêu tạo ra một tập từ vựng hiệu quả, có thể chia nhỏ từ thành các đơn vị con gọi là subword. Nó giúp mô hình ngôn ngữ xử lý được các từ không có trong từ vựng (OOV - out-of-vocabulary) bằng cách chia chúng thành các phần nhỏ hơn có thể dự đoán được từ ngữ liệu huấn luyện.

- Thuật toán WordPiece:
  - Chuẩn bị Dữ liệu:
      - Thuật toán bắt đầu bằng việc thu thập dữ liệu văn bản lớn và xử lý trước văn bản này để có được danh sách các từ. Mỗi từ sẽ được coi là một chuỗi ký tự ban đầu.
  - Xây dựng Từ vựng Ban đầu:
      - Ban đầu, từ vựng chỉ bao gồm các ký tự đơn lẻ từ tập dữ liệu (ví dụ, a, b, c, …).
      - Từ điển này cũng có thể chứa một số từ phổ biến đã xuất hiện nguyên vẹn trong dữ liệu huấn luyện (nếu có). Nhưng nói chung, phần lớn là các ký tự ban đầu.
  - Thuật toán Huấn luyện:
      - Bước 1: Đếm tần suất cặp token (bi-gram):
          - Với mỗi từ trong tập dữ liệu huấn luyện, thuật toán bắt đầu bằng cách chia nó thành các token nhỏ nhất có thể (ban đầu là ký tự đơn lẻ). Ví dụ: từ banana sẽ được chia thành ['b', 'a', 'n', 'a', 'n', 'a'].
          Thuật toán sau đó sẽ đếm tần suất xuất hiện của từng cặp token liền kề (bi-gram). Ví dụ: từ banana sẽ tạo ra các bi-gram: ['b', 'a'], ['a', 'n'], ['n', 'a'], ['a', 'n'], ['n', 'a'].
      - Bước 2: Chọn bi-gram phổ biến nhất:

          - Thuật toán sẽ chọn bi-gram có tần suất xuất hiện cao nhất trong toàn bộ tập dữ liệu. Đây là cặp token phổ biến nhất.
          - Ví dụ: nếu bi-gram ['n', 'a'] xuất hiện nhiều nhất, thuật toán sẽ hợp nhất chúng thành một token mới ['na'].
      - Bước 3: Cập nhật từ vựng:

          - Sau khi hợp nhất bi-gram phổ biến nhất thành một token, từ vựng sẽ được cập nhật để bao gồm token mới này. Token mới này sẽ thay thế các cặp token ban đầu trong văn bản.
          - Ví dụ: sau khi hợp nhất ['n', 'a'], từ banana sẽ được chia thành ['b', 'a', 'na', 'na'].
      - Bước 4: Lặp lại quá trình:

          - Thuật toán tiếp tục quá trình này, đếm lại tần suất của các bi-gram mới và hợp nhất cặp token phổ biến nhất vào từ vựng.
          - Quá trình lặp lại này tiếp diễn cho đến khi đạt đến kích thước từ vựng mong muốn hoặc khi không còn bi-gram nào có tần suất đủ cao để hợp nhất nữa.
  - Xử lý Từ Mới trong Tokenization:
      - Khi từ vựng đã được học xong, trong quá trình sử dụng (inference), nếu một từ mới (chưa có trong từ vựng) xuất hiện, thuật toán sẽ chia từ này thành các subword đã có trong từ vựng.
      - Ví dụ: nếu từ "happiness" không có trong từ vựng đầy đủ, thuật toán có thể chia nó thành ['happy', '##ness'], trong đó ## là tiền tố cho biết đây là phần tiếp theo của một từ lớn hơn.
  - Tiền xử lý với Subword:
      - Một khi từ đã được chia thành các subword, các subword này sẽ được đưa vào mô hình để huấn luyện hoặc suy luận (inference). Mô hình sẽ học cách xử lý các phần từ (subword) thay vì toàn bộ từ, giúp giảm vấn đề từ mới chưa gặp phải trong quá trình huấn luyện.

### Sự khác biệt giữa WordPiece và BPE:
- Cách chọn cặp token để hợp nhất:
    - BPE: BPE chọn cặp token phổ biến nhất (về số lần xuất hiện trong văn bản) để hợp nhất mà không quan tâm đến tần suất xuất hiện của token hợp nhất trong từ vựng. Mỗi lần hợp nhất, BPE đơn giản chỉ chọn cặp token liền kề xuất hiện nhiều nhất trong tập dữ liệu và hợp nhất chúng.

    - Ví dụ: Nếu cặp ký tự "e" và "r" xuất hiện cùng nhau nhiều nhất, BPE sẽ hợp nhất chúng thành "er", bất kể "er" có mang nghĩa trong từ điển hay không.

    - WordPiece: WordPiece không chỉ dựa trên tần suất xuất hiện của cặp token mà còn xem xét xác suất điều kiện của token hợp nhất mới. Nó chọn cặp token sao cho việc hợp nhất chúng sẽ tạo ra một token mới có khả năng giải thích tốt nhất dữ liệu. Điều này có nghĩa là WordPiece sử dụng phương pháp tối ưu hóa dựa trên mô hình xác suất (hoặc tần suất điều kiện) để quyết định cặp nào nên được hợp nhất.

    - Ví dụ: WordPiece sẽ chỉ hợp nhất "un" và "happy" thành "unhappy" nếu điều này giúp mô hình hiểu rõ hơn về cấu trúc từ, thay vì chỉ dựa trên tần suất.
- Mục tiêu tối ưu hóa:
    - BPE: Mục tiêu của BPE là hợp nhất các cặp token phổ biến nhất để giảm số lượng token tổng thể. Nó không quan tâm đến ý nghĩa ngữ nghĩa hay tần suất của các token mới trong từ vựng.
    - WordPiece: WordPiece tập trung vào việc tìm ra các token có ý nghĩa ngữ nghĩa cao hơn thông qua tối ưu hóa xác suất điều kiện của các cặp token. Mục tiêu của nó là xây dựng một từ vựng subword tốt nhất dựa trên sự tối ưu hóa ngữ nghĩa hơn là chỉ đơn giản giảm số lượng token.
- Cấu trúc từ vựng:
    - BPE: Từ vựng của BPE được xây dựng hoàn toàn dựa trên tần suất cặp token trong tập dữ liệu, không quan tâm nhiều đến ý nghĩa hay sự liên quan giữa các token. Do đó, nó thường có thể sinh ra những token không có ý nghĩa về mặt ngữ nghĩa nhưng lại phổ biến.

    - WordPiece: Từ vựng của WordPiece được xây dựng không chỉ dựa vào tần suất mà còn dựa vào các yếu tố ngữ nghĩa. Điều này giúp từ vựng WordPiece có tính liên kết ngữ nghĩa tốt hơn so với BPE.

- Cách hợp nhất:
    - BPE: Cứ mỗi vòng lặp, BPE hợp nhất cặp bi-gram phổ biến nhất (dựa trên tần suất), bất kể mức độ "hữu ích" của cặp đó. Điều này có thể dẫn đến sự hợp nhất của các token có tần suất cao nhưng không mang nhiều thông tin ngữ nghĩa.

    - WordPiece: Trong WordPiece, quá trình hợp nhất được thực hiện dựa trên sự đánh giá xác suất của việc hợp nhất đó mang lại lợi ích cho mô hình. Điều này làm cho quá trình hợp nhất trong WordPiece được tối ưu hóa hơn so với BPE.

- Ứng dụng thực tế:
    - BPE: BPE được sử dụng trong các mô hình như GPT và các mô hình liên quan đến OpenAI. BPE có tốc độ xử lý nhanh hơn so với WordPiece vì thuật toán đơn giản hơn, chỉ cần dựa trên tần suất bi-gram.

  - WordPiece: WordPiece được sử dụng trong các mô hình như BERT và một số mô hình ngôn ngữ khác do Google phát triển. WordPiece có xu hướng tạo ra từ vựng tốt hơn về mặt ngữ nghĩa, giúp mô hình học các biểu diễn ngữ nghĩa sâu hơn.

- Ví dụ minh họa sự khác biệt. Giả sử chúng ta có từ happiness và từ unhappiness:

    - BPE: BPE có thể hợp nhất từng cặp ký tự dựa trên tần suất, chẳng hạn nó có thể hợp nhất h, a, p, p, i, n, e, s, s thành một số subword như hap, pi, ness, mà không quan tâm đến sự liên kết ngữ nghĩa giữa các phần của từ.

    - WordPiece: WordPiece sẽ ưu tiên hợp nhất các token có khả năng xuất hiện cùng nhau để tạo ra các phần từ có ý nghĩa hơn như un, happy, ness. Điều này giúp mô hình hiểu rõ hơn về ngữ nghĩa của từ và cải thiện khả năng tổng quát hóa. 
  
- Giờ hãy thử sử dụng thuật toán WordPiece thông qua thư viện HuggingFace. Đầu tiên, chúng ta se tải vocab của mô hình BERT:

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

## 4. Tài liệu tham khảo.
- [1] [Biểu thức chính quy - ProtonX](https://protonx.coursemind.io/courses/66b0895e02b79700126975cd/topics/66badf03e1f5ad00195e901b?activeAId=66badf0a58f9530012731ae6)