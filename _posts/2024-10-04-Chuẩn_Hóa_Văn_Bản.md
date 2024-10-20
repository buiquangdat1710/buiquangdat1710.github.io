---
title: "Chuẩn Hóa Văn Bản (Text Normalization)"
date: 2024-10-04 00:00:00  + 0800
categories: [NLP]
tags: [text normalization]
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

**Chuẩn hóa văn bản (Text Normalization)** là quá trình xử lý và chuyển đổi dữ liệu văn bản để đảm bảo tính nhất quán, loại bỏ các yếu tố không cần thiết, và chuẩn bị cho các bước xử lý tiếp theo trong các bài toán ngôn ngữ tự nhiên (NLP). Quá trình này thường bao gồm các bước như chuyển đổi chữ hoa thành chữ thường, loại bỏ dấu câu, chuẩn hóa khoảng trắng, và thay thế các từ viết tắt hoặc từ đồng nghĩa. Chuẩn hóa văn bản giúp cải thiện độ chính xác của các mô hình NLP bằng cách giảm thiểu sự đa dạng không cần thiết trong dữ liệu.

## 1. Chuẩn hóa văn bản.

- Chuẩn hóa văn bản là một chuỗi việc chuyển văn bản sang dạng chuẩn, thuận tiện để sử dụng trong các bài toán khác nhau.
- Các quy trình có thể có trong công việc chuẩn hóa văn bản:
  - **Tách câu (sentence segmentation)**: Chia văn bản thành các câu.
  - **Tách token (Tokenization)**: Chia văn bản thành các token.
  - **Lemmatization**: Đưa về dạng từ gốc.
  - **Stemming**: Cắt hậu tố từ. Ít được sử dụng hơn Lemmatization.
  - **Lọc stop words**: Lọc những từ hay xuất hiện và ít ngữ nghĩa như "the", "is", "at", "which", và "on". Thường là các giới từ, lưu ý có những câu mà giới từ để chỉ vị trí quan trọng thì không được lọc.
  - **Sửa sai từ (Word Corection)**: Sai thứ tự chữ trong từ Tiếng Anh hoặc sai dấu trong Tiếng Việt.

- Chuẩn hóa văn bản khác nhau giữa các bài toán khác nhau. Ví dụ như bài toán sinh từ (Text Generation) thì chuẩn hóa sẽ giữ njieuef token nhất có thể, đưa văn bản về chung một format, như lùi đầu dòng, viết hoa đầu câu. Một ví dụ khác là bài toán phân loại cảm xúc (Sentiment Classification) thì chuẩn hóa sẽ loại bỏ những stop-words như a, to,... và giữ lại các biểu tượng cảm xúc như :), :D, =)).

## 2. Lemmatization.
- Lemmatization là việc xác định từ gốc của các từ, ví dự như các từ say, said, saying có từ gốc là say. Ta sẽ nói qua về ưu điểm và nhược điểm của Lemmatization.
- Ưu điểm:
  - **Tìm kiếm tốt hơn**: Khi người dùng tìm kiếm văn bản từ sing, thuật toán có thể cùng tìm kiếm thêm từ sang, sung.
  - **Phân loại tốt hơn**: Chuẩn hóa về từ gốc giúp thu hẹp không gian phân tích và tạo ra độ chính xác cao hơn.
- Nhược điểm:
  - **Đánh mất thông tin ngữ pháp**: Nếu bộ dữ liệu có sự mập mờ lớn thì việc xử lý này sẽ đánh mất thông tin ngữ pháp làm giảm độ chính xác của mô hình.
  - Giờ ta sẽ lập trình Lemmatization sử dụng các thư viện khác nhau, đầu tiên hãy thử lập trình với thư viện nltk (Natural Language Toolkit).

```python
import nltk
nltk.download('wordnet')  # Tải về bộ từ điển WordNet, cần thiết cho lemmatization
from nltk.stem import WordNetLemmatizer  # Import lớp WordNetLemmatizer từ thư viện nltk

# Tạo đối tượng WordNetLemmatizer
wnl = WordNetLemmatizer()

# Ví dụ về lemmatization (chuyển từ về dạng gốc) cho từng từ
list1 = ['kites', 'babies', 'dogs', 'flying', 'smiling',  # Danh sách từ cần lemmatize
         'driving', 'died', 'tried', 'feet']

# Vòng lặp qua từng từ trong list1 để lemmatize
for words in list1:
    print(words + " ---> " + wnl.lemmatize(words))  # In ra từ gốc của từng từ

```

> kites ---> kite

> babies ---> baby

>dogs ---> dog

>flying ---> flying

>smiling ---> smiling

>driving ---> driving

>died ---> died

>tried ---> tried

>feet ---> foot

- Giờ ta hãy thử sử dụng thư viện textblob:

```python
from textblob import TextBlob, Word  # Import thư viện TextBlob và lớp Word

my_word = 'cats'  # Từ cần lemmatize (dạng số nhiều)

# Tạo một đối tượng Word
w = Word(my_word)

print(w.lemmatize())  # Lemmatize từ 'cats', kết quả là 'cat'
#> cat

sentence = 'the bats saw the cats with stripes hanging upside down by their feet.'  
# Câu ví dụ chứa nhiều từ dạng số nhiều và động từ

s = TextBlob(sentence)  # Tạo đối tượng TextBlob từ câu đã cho

# Lemmatize từng từ trong câu và kết hợp lại thành chuỗi hoàn chỉnh
lemmatized_sentence = " ".join([w.lemmatize() for w in s.words])

print(lemmatized_sentence)  # In ra câu đã được lemmatize

```
> cat

> the bat saw the cat with stripe hanging upside down by their foot

- Có vẻ thư viện textblob chỉ chuyển các từ số nhiều về dạng số ít. Giờ chúng ta hãy thử lập trình với thư viện Spacy:

```python
import spacy  # Import thư viện spaCy
nlp = spacy.load('en_core_web_sm')  # Tải mô hình ngôn ngữ tiếng Anh 'en_core_web_sm'

# Tạo đối tượng Doc từ chuỗi đầu vào
doc = nlp(u'the bats saw the cats with best stripes hanging upside down by their feet')

# Tạo danh sách các token từ chuỗi đã cho
tokens = []
for token in doc:
    tokens.append(token)  # Thêm từng token vào danh sách

print(tokens)
#> [the, bats, saw, the, cats, with, best, stripes, hanging, upside, down, by, their, feet]
# In ra danh sách các token (từ) đã tách từ chuỗi đầu vào

# Lemmatize từng token trong câu và nối lại thành câu hoàn chỉnh
lemmatized_sentence = " ".join([token.lemma_ for token in doc])

print(lemmatized_sentence)
```

> [the, bats, saw, the, cats, with, best, stripes, hanging, upside, down, by, their, feet]

> the bat see the cat with good stripe hang upside down by their foot

- Có thể thấy thư viện Spacy đã chuyển được các từ số nhiều về dạng số ít, càng từ ở thì quá khứ về hiện tại, từ so sánh nhất về dạng gốc.

- Giờ chúng ta hãy thử so sánh việc sử dụng Lemmatization và không sử dụng Lemmatization với bài toán phân loại cảm xúc bình luận phim. Trước tiên hãy import các thứ viện cần thiết:

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import pandas as pd
```
- Tiếp theo ta sẽ tải bộ dữ liệu các bình luận về phim:

`wget https://storage.googleapis.com/protonx-cloud-storage/datasets/IMDB%20Dataset.csv`

- Tiếp theo, ta sẽ mở dữ liệu dưới dạng pandas:

```python
df = pd.read_csv('/content/IMDB Dataset.csv')
print(df.iloc()[0]['review'])
```

> One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side

- Giờ chúng ta hãy lập trình hàm xử lý Lemmatization và Stop Words:

```python
lemmatizer = WordNetLemmatizer()  # Tạo đối tượng lemmatizer từ WordNetLemmatizer để lemmatize các từ
stop_words = set(stopwords.words('english'))  # Tạo danh sách các stop words (từ dừng) tiếng Anh

def preprocess_text(text):
    # Tokenize văn bản (tách văn bản thành các từ riêng lẻ)
    tokens = nltk.word_tokenize(text)
    # Lemmatize các từ và loại bỏ các stop words
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    # Kết hợp lại các từ đã lemmatize thành một chuỗi
    return ' '.join(lemmatized_tokens)

# Áp dụng tiền xử lý cho từng đánh giá (review) trong cột 'review' của DataFrame
df['processed_review_with_lemmatization'] = df['review'].apply(preprocess_text)
```

- Giờ hãy lập trình một hàm chỉ xử lý Stop Words:

```python
def preprocess_text_no_lemmatization(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Lemmatize and remove stop words
    clean_tokens = [token for token in tokens if token not in stop_words]
    # Re-join tokens into a string
    return ' '.join(clean_tokens)
df['processed_review_no_lemmatization'] = df['review'].apply(preprocess_text_no_lemmatization)
```

- Giờ hãy tạo một model đơn giản để train với hàm xử lý Lemmatization và xem kết quả:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split

# Assuming df['processed_review'] and df['sentiment'] are your data
X = df['processed_review_with_lemmatization']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # Convert sentiment to binary

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# Convert text to sequences and pad them to ensure uniform length
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

padded_train = pad_sequences(sequences_train, maxlen=200, truncating='post', padding='post')
padded_test = pad_sequences(sequences_test, maxlen=200, truncating='post', padding='post')

# Define the model
model_with_lemma = Sequential([
    Embedding(input_dim=5000, output_dim=16, input_length=200),
    Bidirectional(LSTM(64)),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model_with_lemma.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model_with_lemma.fit(padded_train, y_train, epochs=10, validation_data=(padded_test, y_test))

# Evaluate the model
loss, accuracy = model_with_lemma.evaluate(padded_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

```

> Loss: 0.4142063558101654, Accuracy: 0.8694000244140625

- Giờ hãy thử train với mô hình không sử dụng Lemmatization:

```python
# Assuming df['processed_review'] and df['sentiment'] are your data
X = df['processed_review_no_lemmatization']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # Convert sentiment to binary

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# Convert text to sequences and pad them to ensure uniform length
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

padded_train = pad_sequences(sequences_train, maxlen=200, truncating='post', padding='post')
padded_test = pad_sequences(sequences_test, maxlen=200, truncating='post', padding='post')

# Define the model
model_with_no_lemma = Sequential([
    Embedding(input_dim=5000, output_dim=16, input_length=200),
    Bidirectional(LSTM(64)),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model_with_no_lemma.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model_with_no_lemma.fit(padded_train, y_train, epochs=10, validation_data=(padded_test, y_test))

# Evaluate the model
loss, accuracy = model_with_no_lemma.evaluate(padded_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
```

> Loss: 0.4729436933994293, Accuracy: 0.8672999739646912

- Kết luận: Việc sử dụng Lemmatization có kết quả tốt trên từng bài toán khác nhau. Việc sử dụng có thể làm mất đi thông tin về ngữ pháp
- Các bạn có thể xem 9 cách lập trình Lemmatization tại đây: [Lemmatization](https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/).

## 3. Chia câu (Sentence Segmentation).

- Sentence Segmentation là việc chia văn bản thành câu, việc chia này có thể dựa vào dấu chấm (.), hỏi chấm (?), chấm than (!). Vấn đề khó khăn xảy ra như từ viết tắt trong Tiếng Anh sử dụng dấu chấm. Ví dụ như Mr. hay Mrs.
- Giờ chúng ta hãy thử lập trình Sentence Segmentation bằng các thư viện khác nhau. Đầu tiên là thư viện nltk:

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
s = '''Good muffins cost $3.88\nin New York.  Please buy me two of them.\n\nThanks! Who'''
print(sent_tokenize(s))
```

>['Good muffins cost $3.88\nin New York.',

>'Please buy me two of them.',

>'Thanks!',

>'Who']

- Tiếp theo, thử lập trình bằng thư viện Spacy:

```python
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# The text you want to segment into sentences
text = "This is a sentence. This is another one."

# Process the text
doc = nlp(text)

# Iterate over the sentences
for sent in doc.sents:
    print(sent.text)
```

> This is a sentence.
> This is another one.

- Tiếp theo,ta sẽ sử dụng thư viện underthesea để xử lý Tiếng Việt. Trước tiên ta cần phải tải thư viện underthesea:

`pip install underthesea`

```python
from underthesea import sent_tokenize
text = 'Bạn là hướng nội hay hướng ngoại? Tôi hướng lung tung. Mr. and Mrs'
sent_tokenize(text)
```

> ['Bạn là hướng nội hay hướng ngoại?', 'Tôi hướng lung tung.', 'Mr. and Mrs']

## 4. Lọc Stop Words.

- Đầu tiên, ta sẽ lậ trình lọc Stop Words bằng thư viện nltk:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
- Tiếp theo ta sẽ khởi tạo các từ stop words tiếng anh và chia token câu đầu vào, ta sẽ chỉ lấy các token không phải là stop words:

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sample text
text = "This is an example sentence. However, it contains stop words!"
stop_words = set(stopwords.words('english'))


word_tokens = word_tokenize(text)


filtered_text = [word for word in word_tokens if not word in stop_words]

print("Original text:", text)
print("Filtered text:", " ".join(filtered_text))
```

> Original text: This is an example sentence. However, it contains stop words!

> Filtered text: This example sentence . However , contains stop words !
- Tiếp theo, ta sẽ sử dụng thư viện Spacy:

```python
import spacy

# Load the language model
nlp = spacy.load("en_core_web_sm")

text = "This is an example sentence... However, it contains stop words!"
doc = nlp(text)

filtered_text = [token.text for token in doc if not token.is_stop]

print("Original text:", text)
print("Filtered text:", " ".join(filtered_text))
```

> Original text: This is an example sentence... However, it contains stop words!

> Filtered text: example sentence ... , contains stop words !

- Oke, giờ ta hãy thử huấn luyện giữa hai mô hình, một mô hình sử dụng lọc stop words và một mô hình không sử dụng:

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import pandas as pd
```

- Tải bộ dữ liệu đánh giá phim giống phần trên:

`wget https://storage.googleapis.com/protonx-cloud-storage/datasets/IMDB%20Dataset.csv`

- Load dữ liệu dưới dạng pandas:
  
```python
df = pd.read_csv('/content/IMDB Dataset.csv')
```

- Lập trình hai hàm xử lý, một hàm lọc stop words và một hàm không làm gì:

```python
stop_words = set(stopwords.words('english'))

def preprocess_text_with_stopwords_filtering(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Lemmatize and remove stop words
    lemmatized_tokens = [token for token in tokens if token not in stop_words]
    # Re-join tokens into a string
    return ' '.join(lemmatized_tokens)

def preprocess_text_with_no_stopwords_filtering(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Lemmatize and remove stop words
    lemmatized_tokens = [token for token in tokens]
    # Re-join tokens into a string
    return ' '.join(lemmatized_tokens)


# Apply preprocessing to each review
df['processed_review_with_stopwords_filtering'] = df['review'].apply(preprocess_text_with_stopwords_filtering)
df['processed_review_with_no_stopwords_filtering'] = df['review'].apply(preprocess_text_with_no_stopwords_filtering)
```

- Thử nghiệm huấn luyện mô hình sử dụng Stopwords Filtering:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split

# Assuming df['processed_review'] and df['sentiment'] are your data
X = df['processed_review_with_stopwords_filtering']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # Convert sentiment to binary




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# Convert text to sequences and pad them to ensure uniform length
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

padded_train = pad_sequences(sequences_train, maxlen=200, truncating='post', padding='post')
padded_test = pad_sequences(sequences_test, maxlen=200, truncating='post', padding='post')
# Define the model
model_with_stopwords_filtering = Sequential([
    Embedding(input_dim=5000, output_dim=16, input_length=200),
    Bidirectional(LSTM(64)),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])


# Compile the model
model_with_stopwords_filtering.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model_with_stopwords_filtering.fit(padded_train, y_train, epochs=10, validation_data=(padded_test, y_test))

# Evaluate the model
loss, accuracy = model_with_stopwords_filtering.evaluate(padded_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

```

> Loss: 0.3831138610839844, Accuracy: 0.8629000186920166



- Thử nghiệm huấn luyện mô hình không sử dụng Stopwords Filtering:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split

# Assuming df['processed_review'] and df['sentiment'] are your data
X = df['processed_review_with_no_stopwords_filtering']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # Convert sentiment to binary

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# Convert text to sequences and pad them to ensure uniform length
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

padded_train = pad_sequences(sequences_train, maxlen=200, truncating='post', padding='post')
padded_test = pad_sequences(sequences_test, maxlen=200, truncating='post', padding='post')

# Define the model
model_with_no_stopwords_filtering = Sequential([
    Embedding(input_dim=5000, output_dim=16, input_length=200),
    Bidirectional(LSTM(64)),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model_with_no_stopwords_filtering.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model_with_no_stopwords_filtering.fit(padded_train, y_train, epochs=10, validation_data=(padded_test, y_test))

# Evaluate the model
loss, accuracy = model_with_no_stopwords_filtering.evaluate(padded_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

```

> Loss: 0.4349353015422821, Accuracy: 0.8571000099182129

- Kết luận trong trường hợp này sử dụng Stopwords Filtering cho kết quả tốt hơn.

## 5. Sửa Từ.
- Sửa từ cũng là một bài toán quan trọng trong chuẩn hóa văn bản hay đề xuất tin nhắn. Ví dụ như từ happpy có thể sửa lại thành happy, từ azmaing có thể sửa lại thành amazing, từ intelliengt có thể sửa lại thành intelligent. Thực ra phần này mình thấy thiên về thuật toán hơn là AI vì ta sẽ tính khoảng cách của từ cần sửa với các từ trong từ điển, từ nào mà khoảng cách gần nhất với từ cần sửa thì là từ đúng. Giờ mình sẽ trình bày hai cách tính khoảng cách chính là: **Khoảng cách JCard** và khoảng cách **Edit Distance**.

### Khoảng cách JCard.
- Trước tiên ta sẽ nói về **hệ số Jaccard**, đây là hệ số để đo mức độ tương đồng giữa các tập mẫu hữu hạn và được định nghĩa là kích thước của giao điểm chia cho kích thước của hợp các tập mẫu:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

- Theo công thức thì $0 \leq J(A,B) \leq 1$. Nếu giao điểm A và B rỗng thì $J(A,B) = 0$. **Khoảng cách Jaccard** là khoảng cách đo sự khác biệt giữa các tập mẫu, thu được bằng cách trừ hệ số Jaccard khỏi 1:

$$
d_J(A, B) = 1 - J(A, B) = \frac{|A \cup B| - |A \cap B|}{|A \cup B|}
$$

- $d_J(A, B)$ càng nhỏ thì sự khác biệt giữa hai tập hợp A, B càng nhỏ, ngược lại, $d_J(A, B)$ càng lớn thì sự khác biệt giữa hai tập hợp A,B càng lớn.

- Hai tập A,B có thể định nghĩa là số kí tự riêng biệt trong một từ, hoặc là các cặp 2 kí tự liên tiếp trong một từ,... Cái này là do bạn tự định nghĩa.
- Giờ hãy thử định nghĩa A,B là các cặp 2 kí tự liên tiếp trong một từ, ta sẽ phần tích hai từ là hello và elhlo. Từ hello được phân tích thành tập hợp A = [he,el,ll,lo], từ elhlo được phân tích thành tập hợp B = [el,lh,hl,lo]. Hai tập hợp giao nhau sẽ ra 2 phần tử:

$$
|A \cap B| = 2
$$

- Hai tập hợp này hợp nhau sẽ thành 6 phần tử:

$$
|A \cup B| = 2
$$

- Khoảng cách Jcard:

$$
d_J(A, B) = 1 - J(A, B) = \frac{|A \cup B| - |A \cap B|}{|A \cup B|} = \frac{2}{3}
$$

- Giờ ta sẽ thử lập trình khoảng cách JCard dựa vào từng kí tự:

```python
def jaccard_distance(str1, str2):
    # Convert the strings into sets of characters
    set1 = set(str1)
    set2 = set(str2)

    # Calculate intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Calculate Jaccard distance
    distance = 1 - (len(intersection) / len(union))

    return distance

# Example usage
str1 = "hello"
str2 = "yellow"
distance = jaccard_distance(str1, str2)
print(f"Jaccard Distance: {distance}")

```

> Jaccard Distance: 0.5

- Thêm một ví dụ nữa:

```python
str3 = "azmaing"
str4 = "amazing"
distance = jaccard_distance(str3, str4)
print(f"Jaccard Distance: {distance}")
```

> Jaccard Distance: 0.0

- Giờ hãy thử lập trình bằng cách chia ra các cụm kí tự:

```python
def jaccard_distance_ngrams(str1, str2, n=2):
    # Function to generate n-grams from a given string
    def generate_ngrams(s, n):
        ngrams = set()
        for i in range(len(s) - n + 1):
            ngrams.add(s[i:i+n])
        return ngrams

    # Generate n-grams for both strings
    ngrams1 = generate_ngrams(str1, n)
    ngrams2 = generate_ngrams(str2, n)

    # Calculate intersection and union of n-grams sets
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)

    # Calculate Jaccard distance
    distance = 1 - (len(intersection) / len(union))

    return distance

# Calculate Jaccard distance for 'hello' and 'hlleo' using 2-grams
str1 = "hello"
str2 = "hlleo"
distance = jaccard_distance_ngrams(str1, str2, 2)
distance

```

> 0.8571428571428572

- Hãy lấy thêm một ví dụ nữa:

```python
str1 = "hello"
str2 = "elhlo"
distance = jaccard_distance_ngrams(str1, str2, 2)
distance
```

> 0.6666666666666667

- Tiếp theo ta sẽ đến với một khoảng cách nữa là **Edit Distance**.

### Edit Distance

- **Edit Distance** hay còn gọi là **Levenshtein distance** chỉ sự đô lường tổng số phép nhỏ nhất cần thực hiện để biến chuỗi A thành chuỗi B. Các phép biến bao gồm ba phép: Phép chèn (Insertion), Phép xóa (Deletion), Phép thay thế (Substitution).
- Ví dự như từ kitten và từ sitting có Edit distance bằng 3 vì cần ít nhất 3 phép để chuyển kitten thành sitting. Đầu tiên thay kí tự k thành s, sau đó thay kí tự e bằng i, thêm kí tự g ở cuối.

- Chúng ta sẽ nói qua về quy trình sửa từ. Đầu tiên chúng ta cần lặp qua các từ trong từ điển và tính khoảng cách của các từ này với từ mục tiêu. Từ nào trong từ điển có khoảng cách ngắn nhất thì đề xuất từ đó là từ thay thế. Có thể tối ưu hóa thời gian tìm kiếm bằng cách chỉ lặp qua các từ có độ dài bằng từ mục tiêu, hay chỉ lặp qua các từ có các kí tự giống với từ mục tiêu,...
- Chúng ta có thể lập trình Edit Distance bằng quy hoạch động:

```python
def edit_distance(str1, str2):
    # Create a table to store results of subproblems
    dp = [[0 for x in range(len(str2) + 1)] for x in range(len(str1) + 1)]

    # Fill dp[][] in bottom up manner
    for i in range(len(str1) + 1):
        for j in range(len(str2) + 1):

            # If first string is empty, insert all characters of second string
            if i == 0:
                dp[i][j] = j

            # If second string is empty, remove all characters of the first string
            elif j == 0:
                dp[i][j] = i

            # If last characters are the same, ignore the last char
            # and recur for remaining string
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]

            # If last character is different, consider all possibilities
            # and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])      # Replace

    return dp[len(str1)][len(str2)]

# Example usage
str1 = "hello"
str2 = "elhlo"
print(f"Edit Distance between '{str1}' and '{str2}':", edit_distance(str1, str2))

```

## 6. Giới thiệu thư viện Underthesea.

- 🌊 Bộ công cụ NLP tiếng Việt - Underthesea là một bộ công cụ bằng ngôn ngữ Python mã nguồn mở bao gồm các mô-đun, bộ dữ liệu và hướng dẫn hỗ trợ nghiên cứu và phát triển trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên tiếng Việt. Thư viện có các API dễ sử dụng để nhanh chóng áp dụng các mô hình NLP tiền huấn luyện cho văn bản tiếng Việt của bạn, như phân đoạn từ (Word segmentation), gán phần loại từ (PoS), nhận diện thực thể đặt tên (Named entity recognition - NER), phân loại văn bản (Text Classification) và phân tích phụ thuộc (Dependency Parsing). Chi tiết thư viện [tại đây](https://github.com/undertheseanlp/underthesea).

- Giờ hãy thử lập trình dùng thư viện underthesea, đầu tiên ta cần phải tải thư viện:

`pip install underthesea`

- **Tách văn bản thành câu**:

```python
from underthesea import sent_tokenize
text = 'Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng. Amanda cũng thoải mái với mối quan hệ này.'

print(sent_tokenize(text))
```
 
> [
  "Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng.",
  "Amanda cũng thoải mái với mối quan hệ này."
]

- **Chuẩn hóa văn bản**:

```python
from underthesea import text_normalize
text_normalize("Ðảm baỏ chất lựơng phòng thí nghịêm hoá học")
```
> "Đảm bảo chất lượng phòng thí nghiệm hóa học"

- **Tách câu thành từ**:

```python
from underthesea import word_tokenize
text = "Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò"

word_tokenize(text)

word_tokenize(sentence, format="text")

text = "Viện Nghiên Cứu chiến lược quốc gia về học máy"
fixed_words = ["Viện Nghiên Cứu", "học máy"]
word_tokenize(text, fixed_words=fixed_words)
```

> ["Chàng trai", "9X", "Quảng Trị", "khởi nghiệp", "từ", "nấm", "sò"]

> "Chàng_trai 9X Quảng_Trị khởi_nghiệp từ nấm sò"

> "Viện_Nghiên_Cứu chiến_lược quốc_gia về học_máy"

- **POS Tagging - Đánh nhãn từ**:

```python
from underthesea import pos_tag
pos_tag('Chợ thịt chó nổi tiếng ở Sài Gòn bị truy quét')
```

> [('Chợ', 'N'),

>('thịt', 'N'),

>('chó', 'N'),

>('nổi tiếng', 'A'),

>('ở', 'E'),

>('Sài Gòn', 'Np'),

>('bị', 'V'),

>('truy quét', 'V')]

- Chợ là danh từ, Sài gòn là danh từ riêng, truy quét là động từ.
- **Nhận diện tên thực thể**:
```python
from underthesea import ner
text = 'Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump'
ner(text)
```

> [('Chưa', 'R', 'O', 'O'),

>('tiết lộ', 'V', 'B-VP', 'O'),

>('lịch trình', 'V', 'B-VP', 'O'),

>('tới', 'E', 'B-PP', 'O'),

>('Việt Nam', 'Np', 'B-NP', 'B-LOC'),

>('của', 'E', 'B-PP', 'O'),

>('Tổng thống', 'N', 'B-NP', 'O'),

>('Mỹ', 'Np', 'B-NP', 'B-LOC'),

>('Donald', 'Np', 'B-NP', 'B-PER'),

>('Trump', 'Np', 'B-NP', 'I-PER')]

- **Phân loại chủ đề câu**:

```python
from underthesea import classify

classify('HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu')
classify('Hội đồng tư vấn kinh doanh Asean vinh danh giải thưởng quốc tế')
classify('Lãi suất từ BIDV rất ưu đãi', domain='bank')
```

> ['The thao']

> ['Kinh doanh']

> ['INTEREST_RATE']

- **Phân tích cảm xúc câu**:

```python
from underthesea import sentiment
sentiment('hàng kém chất lg,chăn đắp lên dính lông lá khắp người. thất vọng')
sentiment('Sản phẩm hơi nhỏ so với tưởng tượng nhưng chất lượng tốt, đóng gói cẩn thận.')
sentiment('Đky qua đường link ở bài viết này từ thứ 6 mà giờ chưa thấy ai lhe hết', domain='bank')
sentiment('Xem lại vẫn thấy xúc động và tự hào về BIDV của mình', domain='bank')
```

> 'negative'

> 'positive'

> ['CUSTOMER_SUPPORT#negative']

> ['TRADEMARK#positive']

## 7. Giới thiệu thư viện Spacy.

- SpaCy là một thư viện xử lý ngôn ngữ tự nhiên mã nguồn mở cho Python. Nó được thiết kế để thực hiện xử lý ngôn ngữ tự nhiên. SpaCy cung cấp các mô hình được đào tạo trước cho nhiều ngôn ngữ và nó nổi tiếng với tốc độ và hiệu suất của mình khi xử lý lượng văn bản lớn. 
- **Tách token (Tokenization)**: SpaCy có thể phân đoạn văn bản thành từ hoặc đơn vị con từ, giúp dễ dàng phân tích và xử lý các yếu tố ngôn ngữ.

- **POS-Tagging**: Gán các loại (ví dụ: danh từ, động từ, tính từ) cho mỗi từ trong một câu.

- **Nhận dạng thực thể tên thực thể ( Name Entity Recognition - NER)**: SpaCy có thể xác định và phân loại các thực thể như người, tổ chức, địa điểm, ngày tháng và nhiều hơn nữa trong một đoạn văn bản cho trước.

- **Phân tích phụ thuộc (Dependency Parsing)**: Nó phân tích cấu trúc ngữ pháp của một câu, xác định mối quan hệ giữa các từ và phụ thuộc ngữ pháp của chúng.


- **Lemmatization**: SpaCy có thể giảm các từ về gốc ví dụ running từ gốc là run , điều này hữu ích cho các nhiệm vụ như chuẩn hóa văn bản.


- **Vector hóa từ**: SpaCy cung cấp các vector từ đã được đào tạo trước có thể được sử dụng cho các nhiệm vụ như tính toán độ tương đồng và phân cụm tài liệu.


- **Tùy chỉnh**: Người dùng có thể đào tạo các mô hình riêng của họ bằng cách sử dụng SpaCy cho các lĩnh vực hoặc ngôn ngữ cụ thể.

- Giờ chúng ta sẽ lập trình thư viện Spacy, đầu tiên cần phải tải thư viện xuống:

`pip install spacy`

- Tải mô hình tách từ Tiếng anh:

`python -m spacy download en_core_web_sm`

- Giờ hãy lập trình tách từ:

```python
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Text to be tokenized
text = "SpaCy is a powerful natural language processing library for Python."

# Process the text with spaCy
doc = nlp(text)

# Access the tokens
tokens = [token.text for token in doc]

# Print the tokens
print(tokens)
```

> ['SpaCy', 'is', 'a', 'powerful', 'natural', 'language', 'processing', 'library', 'for', 'Python', '.']

- Ngoài ra Spacy còn hỗ trợ sử dụng bộ tách token của Bert. Trước tiên chúng ta sẽ cài đặt thư viện tokenizers:

`pip install tokenizers`

- Tải từ điển của Bert bằng đoạn code sau:

```python
import requests

url = 'https://raw.githubusercontent.com/microsoft/SDNet/master/bert_vocab_files/bert-base-uncased-vocab.txt'
response = requests.get(url)

# Lưu file xuống
with open('bert-base-uncased-vocab.txt', 'w') as file:
    file.write(response.text)

```

- Như bạn đã biết Bert sử dụng thuật toán WordPiece để tách từ. Xem chi tiết về thuật toán này [tại đây](https://buiquangdat1710.github.io/posts/C%C3%A1c_K%C4%A9_Thu%E1%BA%ADt_T%C3%A1ch_T%E1%BB%AB_Ph%E1%BA%A7n_2/).
- Giờ hãy lập trình để tách token từ mô hình Bert:
  
```python
from tokenizers import BertWordPieceTokenizer
from spacy.tokens import Doc
import spacy

class BertTokenizer:
    def __init__(self, vocab, vocab_file, lowercase=True):
        self.vocab = vocab
        self._tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=lowercase)

    def __call__(self, text):
        tokens = self._tokenizer.encode(text)
        words = []
        spaces = []
        for i, (text, (start, end)) in enumerate(zip(tokens.tokens, tokens.offsets)):
            words.append(text)
            if i < len(tokens.tokens) - 1:
                # If next start != current end we assume a space in between
                next_start, next_end = tokens.offsets[i + 1]
                spaces.append(next_start > end)
            else:
                spaces.append(True)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.blank("en")
nlp.tokenizer = BertTokenizer(nlp.vocab, "bert-base-uncased-vocab.txt")
doc = nlp("Justin Drew Bieber is a Canadian singer, songwriter, and actor.")
print([token.text for token in doc])
```

> ['[CLS]', 'justin', 'drew', 'bi', '##eber', 'is', 'a', 'canadian', 'singer', ',', 'songwriter', ',', 'and', 'actor', '.', '[SEP]']

### Kiến trúc Spacy.

![Ảnh Kiến trúc của Spacy](https://spacy.io/images/architecture.svg)

- Kiến trúc lõi của Spacy bao gồm các thành phần sau:
  - **Lớp Language**: Lớp này sử dụng để xử lý văn bản và chuyển thành đối tượng Doc. Biến sử dụng là nlp.
  - **Lớp Vocab**: Vector của từ và các thuộc tính từ vựng nằm bên trong Vocab
  - **Lớp Doc**: Đối tượng Doc chứa chuỗi tokens và nhãn tương ứng.

- Lớp **Token** là lớp cơ bản nhất của Spacy. Bạn có thể xem các hàm chi tiết tại đây: [https://spacy.io/api/token](https://spacy.io/api/token)
- Cách chuyển văn bản thành chuỗi các Token:

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Give it back! He pleaded.")
```

- Bạn hoàn toàn có thể truy cập vào vector của token thông qua thuộc tính sau:

```python
print(doc.vector)
```

- Lớp **Doc** là lớp tạo ra những đối tượng chứa chuỗi các token.

### Xây dựng từ điển từ các văn bản.

![Ảnh](https://spacy.io/images/vocab_stringstore.svg)

- Từ nhiều văn bản ta xây dựng được từ điển thông qua việc tạo ra ánh xạ giữa các token với số hoặc giá trị băm hash. Ví dụ trong trường hợp này:
  - Token I có giá trị 46904.
  - Token love có giá trị 37020.

- Dưới đây là đoạn code ví dụ:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I love coffee")
print(doc.vocab.strings["coffee"])  
print(doc.vocab.strings[3197928453018144401])  

```
> 3197928453018144401

> coffee

- Trong trường hợp này hash của token "coffee" là 3197928453018144401. Hiển thị từ đã được đưa vào từ điển:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I love coffee")
for word in doc:
    lexeme = doc.vocab[word.text]
    print(lexeme.text, lexeme.orth, lexeme.shape_, lexeme.prefix_, lexeme.suffix_,
            lexeme.is_alpha, lexeme.is_digit, lexeme.is_title, lexeme.lang_)
```
> I 4690420944186131903 X I I True False True en

> love 3702023516439754181 xxxx l ove True False False en

> coffee 3197928453018144401 xxxx c fee True False False en

- Đơn vị lưu trữ trong Vocab dưới dạng từ vựng (Lexeme). Một từ vựng có các thuộc tính sau:
  - **Văn bản:** Văn bản gốc của từ vựng.
  - **Orth:** Giá trị băm của từ vựng.
  - **Shape:** Hình dạng từ ngữ trừu tượng của từ vựng.
  - **Prefix:** Theo mặc định, là chữ cái đầu tiên của chuỗi từ.
  - **Suffix:** Theo mặc định, là ba chữ cái cuối cùng của chuỗi từ.
  - **is alpha:** Liệu từ vựng có bao gồm các ký tự chữ cái không?
  - **is digit:** Liệu từ vựng có bao gồm các chữ số không?

- Giá trị băm của một token không thể decode để ra được từ đó và luôn phải có từ điển để ánh xạ ngược lại.

```python
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab

nlp = spacy.load("en_core_web_sm")
doc = nlp("I love coffee")  # Original Doc
print(doc.vocab.strings["coffee"])  # 3197928453018144401
print(doc.vocab.strings[3197928453018144401])  # 'coffee' 👍

empty_doc = Doc(Vocab())  # New Doc with empty Vocab
# empty_doc.vocab.strings[3197928453018144401] will raise an error :(

empty_doc.vocab.strings.add("coffee")  # Add "coffee" and generate hash
print(empty_doc.vocab.strings[3197928453018144401])  # 'coffee' 👍

new_doc = Doc(doc.vocab)  # Create new doc with first doc's vocab
print(new_doc.vocab.strings[3197928453018144401])  # 'coffee' 👍
```

> 3197928453018144401

> coffee

> coffee

> coffee

### Thiết kế Pipeline.

![Anh](https://spacy.io/images/pipeline.svg)

- Các thành phần trong pipeline bạn có thể tùy biến: 
  - **Tagger**: part-of-speech tags.
  - **Parser**: Thêm phụ thuộc.
  - **Ner**: Phát hiện và đánh nhãn các thực thể.
  - **Lemmatizer**: Đưa về dạng gốc.
  - **Textcat**: Đánh nhãn văn bảncustom: Bổ sung các thuộc tính, hàm tùy chỉnh vào trong pipeline.

## 8. Mô hình đề xuất bởi Kiss và Strunk.

### Từ viết tắt - Abbreviation.
- Có ba yếu tố chính của từ viết tắt:
  - Từ viết tắt sẽ trông dưới dạng các ký từ đi liền với dấu chấm. Ví dụ Mrs. hay Mr. hay Dr.
  - Từ viết tắt có xu hướng ngắn. Từ càng dài thì khả năng là từ viết tắt càng thấp.
  - Dấu chấm xuất hiện nhiều ở giữa từ viết tắt.
- Hệ thống phát hiện dấu chấm của Punkt phát hiện được 99.38% từ viết tắt. Tuy nhiên những từ viết tắt này chưa bao gồm:
  - Chữ viết tắt tên ví dụ John F Kennedy - J.F.K hoặc Dr. J. Smith hoặc J.K. Rowling. 
  - Số thứ tự (Ordinal Numbers) như 1. 2.

- Với việc kết luận được từ viết tắt thì tất cả các dấu chấm đứng sau những từ không phải từ viết tắt có thể coi là kết thúc câu (End of a Sentence). Tuy nhiên có trường hợp dấu chấm vừa là kết thúc từ viết tắt cũng như là kết thúc câu.

### Pipeline để phát hiện dấu chấm câu.
- Bài báo gốc: [https://aclanthology.org/J06-4003.pdf](https://aclanthology.org/J06-4003.pdf)

![Ảnh](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/e6235cf0-a9eb-11ee-881b-717bb9489abc-Screen_Shot_2024_01_03_at_10.55.01_500x448.png)

#### 1. Khâu đầu tiên - Type-based classification stage.

- Phát hiện từ viết tắt thông qua:
  - Collocational bond: Từ viết tắt hay đi với dấu chấm đằng sau.
  - Length: Chiều dài của từ viết tắt.
  - Internal Periods: Từ viết tắt có dấu chấm câu bên trong.
  - Occurrences without a final period: Từ viết tắt không có dấu chấm kết thúc.
- Sau bước này các từ sẽ được phân loại thành:
  - `<A>`: Từ viết tắt.
  - `<E>`: Dấu 3 chấm (...).
  - `<S>`: Dấu chấm mà không đi sau từ viết tắt thì quy thành kết thúc câu.

#### 2. Khâu thứ hai - Token-based classification stage.

- Phát hiện từ viết tắt ở cuối câu: `<A>` thành `<A><S>`
- Phát hiện dấu ba chấm ở cuối câu: `<E>` thành `<E> <S>`
  - Cách phát hiện: sử dụng Regex: `[^\W\d]\.$`
- Phát hiện ký tự đầu câu: `<S>` thành `<A>` và `<A>` thành `<A>`. Ví dụ A. hay B.
  - Cách phát hiện: sử dụng Regex: `[^\W\d]\.$`
- Phát hiện số thứ tự: `<S>` Thành `<A>`. Ví dụ 1. hay 1000. hay -1.
  - Cách phát hiện: sử dụng Regex: `^-?[\.,]?\d[\d,\.-]*\.?$`

#### 3. Khâu thứ ba.

- Kết luật từ đó có phải từ viết tắt hay không với các nhãn. `<A>`, `<E>`, `<S>`, `<A><S>`, `<E><S>`

### Dunning log-likelihood.
- Công thức Dunning log-likelihood được sử dụng để so sánh tần suất của một đặc trưng trong hai ngữ liệu khác nhau nhằm xem liệu sự khác biệt có mang ý nghĩa thống kê hay không. Công thức này dựa trên tỷ lệ khả năng (likelihood ratio), so sánh khả năng của dữ liệu dưới hai giả thuyết khác nhau. Dưới đây là công thức:

$$
G^2 = 2 \sum_{i=1}^{k} O_i \cdot \ln \left( \frac{O_i}{E_i} \right)
$$

- Trong đó:

    - $G^2$ là thống kê tỷ lệ khả năng log (log-likelihood ratio statistic).
    - $O_i$ là tần suất quan sát được của một sự kiện cụ thể trong dữ liệu.
    - $E_i$ là tần suất kỳ vọng của sự kiện dưới giả thuyết không (null hypothesis).
    - Tổng được tính qua tất cả các sự kiện (hoặc hạng mục) đang được xem xét.
- Với trường hợp có hai sự kiện:

$$
G^2 = 2 \left[ O_1 \cdot \ln \left( \frac{O_1}{E_1} \right) + O_2 \cdot \ln \left( \frac{O_2}{E_2} \right) \right]
$$

- Trong ngữ cảnh này:

  - $O_1$ và $O_2$ là tần suất quan sát trong mỗi tập ngữ liệu.
  - $E_1$ và $E_2$ là tần suất kỳ vọng dưới giả thuyết không, giả định rằng tần suất tổng thể của đặc trưng là như nhau trong cả hai tập ngữ liệu.
  
- Các tần suất kỳ vọng $E_i$ thường được tính toán dựa trên tần suất tổng thể của sự kiện và kích thước của mỗi tập ngữ liệu. Giá trị của $G^2$, càng lớn thì càng ít khả năng là sự phân bố quan sát được là do ngẫu nhiên, cho thấy rằng sự khác biệt trong tần suất giữa hai tập ngữ liệu là có ý nghĩa thống kê.


- Đối với một kịch bản điển hình, khi bạn so sánh tần suất của một từ hoặc đặc trưng trong hai tập dữ liệu (gọi là Tập dữ liệu 1 và Tập dữ liệu 2), bạn sẽ tính $E_i$ như sau:
  - **Tổng số lần xảy ra sự kiện trong cả hai tập dữ liệu**: Đầu tiên, tìm tổng số lần xảy ra sự kiện (ví dụ: một từ cụ thể) trong cả hai tập dữ liệu đã kết hợp. Gọi số đếm này là $C$.
  - Kích thước tổng của từng tập dữ liệu: Xác định kích thước tổng của mỗi tập dữ liệu. Gọi kích thước của Tập dữ liệu 1 là $N_1$ và Tập dữ liệu 2 là $N_2$.
  - Kích thước tổng của cả hai tập dữ liệu: Tính tổng kích thước của cả hai tập dữ liệu, $N = N_1 + N_2$.
  - Tính tần suất kỳ vọng cho mỗi tập dữ liệu: Tần suất kỳ vọng của sự kiện trong mỗi tập dữ liệu dưới giả thuyết không có thể được tính bằng công thức:
  - 
  $$
  E_i = \frac{C*N_i}{N}
  $$ 
  


- Công thức Dunning log-likelihood này được dùng để xác định xem sự khác biệt trong tần suất của một sự kiện cụ thể trong hai tập dữ liệu có mang ý nghĩa thống kê hay không. Cụ thể, nó giúp so sánh sự phân bố của một đặc trưng (ví dụ: một từ hoặc một sự kiện) giữa hai tập dữ liệu khác nhau, và cho thấy sự khác biệt đó có thể chỉ là do ngẫu nhiên hoặc thực sự có sự khác biệt đáng kể.

- Công thức này hữu ích khi bạn muốn kiểm tra xem một từ hoặc sự kiện có xuất hiện khác nhau về tần suất trong hai tập dữ liệu văn bản khác nhau (ví dụ: so sánh tần suất xuất hiện của một từ trong ngữ liệu của báo chí và ngữ liệu của mạng xã hội).
- Giả sử chúng ta muốn so sánh tần suất xuất hiện của từ "AI" trong hai tập văn bản khác nhau: một tập văn bản về công nghệ và một tập văn bản về văn học.
- **Ngữ liệu về công nghệ**: Từ "AI" xuất hiện 300 lần trên tổng số 10.000 từ.
- **Ngữ liệu về văn học**: Từ "AI" xuất hiện 20 lần trên tổng số 8.000 từ.

#### Bước 1: Tính tần suất quan sát được.

- $O_1$ ​(số lần xuất hiện từ "AI" trong ngữ liệu công nghệ) = 300.
- $O_2$ số lần xuất hiện từ "AI" trong ngữ liệu văn học) = 20.

#### Bước 2:  Tính tần suất kỳ vọng.

- Dưới giả thuyết không, ta giả định tần suất xuất hiện của từ "AI" sẽ tương tự trong cả hai ngữ liệu. Tổng số lần xuất hiện của "AI" trong cả hai ngữ liệu là:
  - Tổng số lần xuất hiện của từ "AI": $300+20=320$.
  - Tổng số từ trong cả hai ngữ liệu: $10000 + 8000 = 18000$.
- Tần suất kỳ vọng của từ "AI" trong ngữ liệu công nghệ và văn học sẽ lần lượt là:
  - $E_1$ (tần suất kỳ vọng của "AI" trong ngữ liệu công nghệ) = $\frac{320*10000}{18000} \approx 177.78$.
  - $E_2$ tần suất kỳ vọng của "AI" trong ngữ liệu văn học) = $\frac{320*8000}{18000} \approx 142.22$.

#### Bước 3: Tính $G^2$

- Sử dụng công thức Dunning log-likelihood cho hai sự kiện:

$$
G^2 = 2 \left[ O_1 \cdot \ln \left( \frac{O_1}{E_1} \right) + O_2 \cdot \ln \left( \frac{O_2}{E_2} \right) \right]
$$

- Thay giá trị vào:

$$
G^2 = 2 \left[ 300 \cdot \ln \left( \frac{300}{177.78} \right) + 20 \cdot \ln \left( \frac{20}{1422.22} \right) \right] = 235.44
$$

- Giá trị $G^2 = 235.44$ cho thấy rằng sự khác biệt về tần suất xuất hiện của từ "AI" giữa ngữ liệu công nghệ và văn học rất có ý nghĩa thống kê. Điều này có nghĩa là tần suất của từ "AI" trong hai tập dữ liệu khác nhau không phải là do ngẫu nhiên, mà do sự khác biệt nội tại giữa các chủ đề của hai tập văn bản.
- Từ "AI" xuất hiện thường xuyên hơn rất nhiều trong ngữ liệu công nghệ so với ngữ liệu văn học, và sự khác biệt này là có ý nghĩa thống kê.
- **$G^2$ càng cao**, sự khác biệt giữa tần suất quan sát được và tần suất kỳ vọng càng lớn. Điều này có nghĩa là có sự khác biệt đáng kể giữa hai tập ngữ liệu về tần suất xuất hiện của đặc trưng đang được so sánh (ví dụ: một từ xuất hiện nhiều hơn trong một tập so với tập kia). Sự khác biệt này có thể không phải ngẫu nhiên mà do nội tại của hai ngữ liệu (ví dụ, do nội dung chủ đề của hai ngữ liệu khác nhau).
- **$G^2$ càng thấp**, sự khác biệt giữa tần suất quan sát được và tần suất kỳ vọng càng nhỏ, cho thấy rằng tần suất xuất hiện của đặc trưng trong cả hai ngữ liệu có xu hướng tương đồng. Điều này ngụ ý rằng sự khác biệt (nếu có) có thể chỉ là do ngẫu nhiên, và không có dấu hiệu rõ ràng về sự khác biệt giữa hai tập ngữ liệu.


### Likelihood là lõi.

- Cả khâu một và hai đều sử dụng khả năng (Likelihood) để xác định mối quan hệ giữa các từ viết tắt và dấu chấm của nó cũng như mối quan hệ của dấu chấm kết thúc câu và những từ đi sau nó và những từ xung quanh một dấu chấm. Chú ý từ w trong bài viết này là **từ viết tắt**.

#### 1. Sử dụng Likelihood trong khâu 1.

- Log-likelihood đề xuất bởi Dunning năm 1993 kiểm tra liệu xác suất của một từ có phụ thuộc với việc với loại từ ở phía trước không.

- **Giả thuyết Null ($H_0$)**:

$$
P( \cdot | w) = p = P( \cdot | \neg w)
$$

- Giả thuyết thay thế khẳng định rằng xác suất xuất hiện dấu chấm phụ thuộc vào từ đứng trước. Điều này có nghĩa là nếu từ trước thay đổi, xác suất xuất hiện dấu chấm cũng thay đổi.
- Ví dụ: Giả sử chúng ta đang kiểm tra xem dấu chấm có phụ thuộc vào từ "học" trong câu "Tôi đang học." hay không. Giả thuyết Null ($H_0$) nói rằng xác suất xuất hiện dấu chấm trong câu này sẽ không thay đổi nếu chúng ta thay thế từ "học" bằng từ khác như "chơi" (ví dụ: "Tôi đang chơi."). Tức là, xác suất xuất hiện dấu chấm sẽ giống nhau dù từ trước nó là "học" hay "chơi".

- **Giả thuyết Thay thế ($H_1$)**:

$$
P( \cdot | w) = p_1 \neq P( \cdot | \neg w) = p_2
$$

- Giả thuyết thay thế khẳng định rằng xác suất xuất hiện dấu chấm phụ thuộc vào từ đứng trước. Điều này có nghĩa là nếu từ trước thay đổi, xác suất xuất hiện dấu chấm cũng thay đổi.

- Tiếp tục ví dụ trước, giả thuyết thay thế ($H_1$) cho rằng xác suất xuất hiện dấu chấm trong câu "Tôi đang học." khác với xác suất trong câu "Tôi đang chơi.". Từ "học" và "chơi" có ảnh hưởng đến việc có dấu chấm hay không, vì vậy $P(.\|học) \neq P(.\|chơi)$.

- **Tỷ số Log-likelihood ($\lambda$)**:

$$
\log \lambda = -2 \log \frac{P_{\text{binom}}(H_0)}{P_{\text{binom}}(H_1)}
$$

- Đây là tỷ số log-likelihood được dùng để so sánh hai xác suất dựa trên giả thuyết Null và giả thuyết thay thế. Nếu giá trị của tỷ số log-likelihood lớn, điều này chỉ ra rằng giả thuyết thay thế có khả năng đúng hơn giả thuyết Null.

- $P_{\text{binom}}(H_0)$ là xác suất xảy ra hiện tượng (ví dụ: có dấu chấm) dựa trên giả thuyết Null.
- $P_{\text{binom}}(H_1)$  là xác suất xảy ra hiện tượng dựa trên giả thuyết thay thế. Tỷ số này được sử dụng để đánh giá liệu giả thuyết Null có khả năng đúng hay không. Nếu $\lambda$ lớn, thì giả thuyết thay thế sẽ có nhiều khả năng đúng hơn.
- Ví dụ: Giả sử chúng ta có một tập dữ liệu gồm 100 câu, trong đó 50 câu có dấu chấm khi từ đứng trước là "học", và 20 câu có dấu chấm khi từ đứng trước là "chơi". Chúng ta có thể tính toán tỷ số log-likelihood để xác định xem dấu chấm có phụ thuộc vào từ đứng trước hay không.

- **Giả thuyết Null đã điều chỉnh (MLE)**:

$$
P( \cdot | w) = P_{\text{MLE}}( \cdot ) = \frac{C( \cdot )}{N}
$$

- Công thức này sử dụng ước lượng hợp lý tối đa (Maximum Likelihood Estimation - MLE) để tính xác suất của dấu chấm
- $C(\cdot)$ là số lần xuất hiện dấu chấm trong tập dữ liệu.
- $N$ là tổng số câu trong tập dữ liệu. Công thức này giả định rằng xác suất xuất hiện dấu chấm là một giá trị cố định được tính dựa trên tần suất xuất hiện của nó trong dữ liệu.
  
- **Giả thuyết Thay thế đã điều chỉnh**:

$$
P( \cdot | w) = 0.99
$$

- Điều này cho thấy rằng giả thuyết thay thế khẳng định xác suất xuất hiện dấu chấm gần như chắc chắn (99%) nếu từ trước là từ đặc biệt nào đó. Việc cài đặt giá trị 0.99 thay vì 1, chúng ta cho rằng một từ viết tắt thi thoảng sẽ không có dấu chấm đi kèm trong ngữ liệu.
- Ví dụ: Nếu chúng ta cho rằng trong 100 câu, 99 câu có dấu chấm khi từ trước là "học", thì xác suất xuất hiện dấu chấm trong trường hợp từ trước là "học" gần như chắc chắn:

$$
P( \cdot | học) = 0.99
$$

- Các công thức likelihood giúp xác định mối quan hệ giữa từ đứng trước và dấu câu, cho phép ta đánh giá xem sự xuất hiện của dấu câu có phụ thuộc vào từ trước hay không. Việc tính toán log-likelihood sẽ giúp đưa ra kết luận về mức độ ảnh hưởng của các từ đối với dấu câu trong một tập dữ liệu ngôn ngữ tự nhiên.
- Cách lập trình như sau:

```python
def _dunning_log_likelihood(count_a, count_b, count_ab, N):
        """
        A function that calculates the modified Dunning log-likelihood
        ratio scores for abbreviation candidates.  The details of how
        this works is available in the paper.
        """
        p1 = count_b / N
        p2 = 0.99

        null_hypo = count_ab * math.log(p1 + 1e-8) + (count_a - count_ab) * math.log(
            1.0 - p1 + 1e-8
        )
        alt_hypo = count_ab * math.log(p2) + (count_a - count_ab) * math.log(1.0 - p2)

        likelihood = null_hypo - alt_hypo

        return -2.0 * likelihood
```

- Trong trường hợp này:
  - count_a sẽ là số lần từ này xuất hiện với dấu chấm câu ở cuối và số lần không xuất hiện dấu chấm ở cuối.
  - count_b sẽ là số lượng dấu chấm.
  - count_ab sẽ là số lần từ này xuất hiện với dấu chấm ở cuối.
  - count_a - count_ab = số lần từ này xuất hiện với dấu chấm không ở cuối.
  - N là số lượng tất cả các tokens trong dataset.
  - p1: Xác suất xuất hiện dấu chấm ở cuối.
  - p2: Xác suất không xuất hiện dấu chấm ở cuối.

- Lập trình Giả thuyết null:
  - Sử dụng phân phối nhị thức với count_a lần xảy ra:
    -  Từ đi sau dấu chấm ở cuối với xác suất p1 xảy ra count_ab lần với xác suất p1.
    -  Từ đi sau không có dấu chấm ở cuối xảy ra count_a - count_ab lần với xác suất 1 - p1.

```python
null_hypo = count_ab * math.log(p1 + 1e-8) + (count_a - count_ab) * math.log(1.0 - p1 + 1e-8)
```

- Lập trình Giả thuyết thay thế (Tương tự giả thuyết null):

```python
alt_hypo = count_ab * math.log(p2) + (count_a - count_ab) * math.log(1.0 - p2)
```

- Tính giá trị likelihood:

$$
\log \lambda = -2 \log \frac{P_{\text{binom}}(H_0)}{P_{\text{binom}}(H_A)}
\log \lambda = 2 * (\log P_{\text{binom}}(H_A) - \log P_{\text{binom}}(H_0))
$$ 

```python
likelihood = -2.0 * (null_hypo - alt_hypo)
```

#### 2. Sử dụng Likelihood trong Khâu 2.

- Với một cặp từ w1 và w2 xung quanh một dấu chấm và chúng ta kiểm tra rằng liệu có việc xuất hiện thường xuyên giữa chúng hay không.

- Giả thuyết Null: 

$$
P( \cdot | w) = p = P( \cdot | \neg w)
$$

- Giả thuyết thay thế:

$$
P( \cdot | w) = p_1 \neq P( \cdot | \neg w) = p_2
$$

- Giá trị $\log \lambda$  sẽ lớn nếu xác suất $p_1$ và $p_2$ khác nhau nhiều tuy nhiên ta sẽ quan tâm tới trường hợp $p_1$ lớn hơn nhiều $p_2$ hay $p_1$ >> $p_2$ với ý nghĩa $w_2$ xuất hiện sau w1 nhiều hơn kỳ vọng.
- Vì trường hợp $p_1$ << $p_2$ hiếm nên ta sẽ coi là $p_1$ >> $p_2$.

- Công thức sử dụng:

$$
\log \lambda = 0 if \frac{C(w_2)}{N} = \frac{C(w_1,w_2)}{C(w_1)}
$$

- Vế phải là $w_2$ hay đi sau $w_1$.
- Vế trái là $w_2$ đi một mình.
- Nếu hai vế của phương trình này khác nhau, giá trị $\log \lambda$ sẽ lớn hơn 0. Giá trị $\log \lambda$ lớn có nghĩa rằng $w_1$ và $w_2$ hay đi với nhau hay vế phải lớn hơn vế trái của công thức trên. Nếu vế trái lớn hơn vế phải, giá trị $\log \lambda$  vẫn lớn hơn 0 tuy nhiên phản ánh w2 không xuất hiện thường xuyên sau w1 như kỳ vọng.

```python
    def _col_log_likelihood(count_a, count_b, count_ab, N):
        """
        A function that will just compute log-likelihood estimate, in
        the original paper it's described in algorithm 6 and 7.

        This *should* be the original Dunning log-likelihood values,
        unlike the previous log_l function where it used modified
        Dunning log-likelihood values
        """
        # count_a: Số lượng w1 xuất hiện
        # count_b: Số lượng w2 xuất hiện
        # count_ab: Số lượng w1w2
        # N - count_a: Số lượng không phải w1
        # N - count_a - count_b + count_ab: Không có w1 và w2. Cách viết khác: N - count_a - (count_b - count_ab)
        # count_b - count_ab: w2 đứng một mình không đi với w1
        # count_a - count_ab: w1 đứng một mình không đi với w2
        # p: Xác suất w2 đứng một mình, sử dụng cho null hypothesis
        p = count_b / N

        # p1: P(w2|w1)
        p1 = count_ab / count_a
        try:
            # p2: P(w2|¬w1)
            p2 = (count_b - count_ab) / (N - count_a)
        except ZeroDivisionError:
            p2 = 1

        try:
            # Binomial null hypothesis p. P(w2|w1)
            summand1 = count_ab * math.log(p) + (count_a - count_ab) * math.log(1.0 - p)
        except ValueError:
            summand1 = 0

        try:
            # Binomial null hypothesis p. P(w2|¬w1)
            summand2 = (count_b - count_ab) * math.log(p) + (
                N - count_a - count_b + count_ab
            ) * math.log(1.0 - p)
        except ValueError:
            summand2 = 0

        if count_a == count_ab or p1 <= 0 or p1 >= 1:
            summand3 = 0
        else:
            # Binomial alternative hypothesis p1: P(w2|w1)
            summand3 = count_ab * math.log(p1) + (count_a - count_ab) * math.log(
                1.0 - p1
            )

        if count_b == count_ab or p2 <= 0 or p2 >= 1:
            summand4 = 0
        else:
            # Binomial alternative hypothesis p2 P(w2|¬w1)
            summand4 = (count_b - count_ab) * math.log(p2) + (
                N - count_a - count_b + count_ab
            ) * math.log(1.0 - p2)

        likelihood = summand1 + summand2 - summand3 - summand4

        return -2.0 * likelihood

```

### Type-based Classification
- Có ba yếu tố xác định một từ viết tắt:
  - Hay đi cùng với dấu chấm ở cuối.
  - Ngắn.
  - Rất nhiều từ viết tắt chứa dấu chấm bên trong.

![Anh](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/7b6fa1e0-aa16-11ee-9bef-cda9df06a9a6-Screen_Shot_2024_01_03_at_15.59.53_900x434.png)

- Các từ viết nghiêng không phải từ viết tắt. Tuy nhiên từ *ounces* rất hay xuất hiện đi kèm dấu chấm đằng sau (4 lần) mà không bao giờ xuất hiện nếu không có dấu chấm đi kèm. Cho nên cách làm chỉ đơn giản dùng 3 tiêu chí này chưa đủ để đánh giá từ *ounces* là một từ viết tắt. Chiều dài của từ viết tắt ngắn hơn so với từ không viết tắt. Chiều dài của từ viết tắt ngắn hơn so với từ không viết tắt.

#### Mô hình hóa các yếu tố.

- Chiều dài của từ sẽ loại bỏ số lượng các dấu chấm trong từ. Ví dụ:

$$
length(u.s.a.) = 3
$$

- Trường hợp này chiều dài sẽ bằng 3 vì đã loại bỏ số lượng dấu chấm. Tính chất từ viết tắt ngắn dựa trên số lượng ký tự không bao gồm dấu chấm.
- Giá trị mô phỏng được lựa chọn dưới dạng hàm mũ vì nó phản ánh chính xác khả năng là từ viết tắt khi chiều dài giảm đi.

$$
F_{\text{length}(w)} = \frac{1}{e^{\text{length(w)}}}
$$

- Biểu đồ thể hiện mối quan hệ giữa chiều dài của từ và khả năng từ đó là từ viết tắt, mối quan hệ dưới dạng hàm mũ. Càng ngắn thì khả năng là từ viết tắt càng cao.

![Anh](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/ebc8e1b0-aab4-11ee-9077-df5dcf1c4350-Screen_Shot_2024_01_04_at_10.54.05.png)

- Hàm đếm số lượng dấu chấm trong từ. Từ càng nhiều dấu chấm bên trong thì có khả năng càng cao là từ viết tắt:

![anh](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/20e42850-aab5-11ee-9bef-cda9df06a9a6-Screen_Shot_2024_01_04_at_10.55.34.png)

- Tổng kết các yếu tố và đưa ra kết luận:

![Anh](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/9b5952a0-ac44-11ee-9bef-cda9df06a9a6-Screen_Shot_2024_01_06_at_10.35.02.png)

## 9. Tài liệu tham khảo.
- [1] [Chuẩn hóa văn bản - ProtonX](https://protonx.coursemind.io/courses/66b0895e02b79700126975cd/topics/66badf0a58f9530012731ad6?activeAId=66badf0a58f9530012731ae6)