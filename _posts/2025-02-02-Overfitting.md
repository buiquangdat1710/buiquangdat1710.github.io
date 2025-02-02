---
title: "Overfitting"
date: 2025-02-02 00:00:00  + 0800
categories: [Machine Learning]
tags: [overfitting]
---
---


<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [['$','$'], ['\$','\$']],
            processEscapes: true
        }
    });
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML-full"></script>


Overfitting không phải là một thuật toán trong Machine Learning. Nó là một hiện tượng không mong muốn thường gặp, người xây dựng mô hình Machine Learning cần nắm được các kỹ thuật để tránh hiện tượng này.


## 1. Giới thiệu

- Có bao giờ bạn học xong hồi quy tuyến tính rồi tự hỏi bản thân là tại sao chúng ta không cố gắng tìm một hàm số mà có thể đi qua hết cả điểm dữ liệu ? Nếu tìm được thì hàm chi phí sẽ bằng $0$, chẳng phải quá tốt hay sao, chúng ta đã tìm được giá trị nhỏ nhất của hàm chi phí. Thực tế trong toán học, chúng ta có thể tìm ra ham đi qua hế tất cả các điểm dữ liệu như vậy thông qua một cái gọi là [Đa thức nội suy Lagrange](https://vuontoanblog.blogspot.com/2012/10/polynomial-interpolation-lagrange.html) (đây là kiến thức được dạy ở chương trình chuyên toán cấp 3). Nhưng cái gì mà tốt quá cũng không phải là một điều tốt, tại sao lại như vậy ? Chúng ta cũng tìm hiểu trong bài này.
- Nhắc lại một chút về Đa thức nội suy Lagrange: Với $N$ cặp điểm dữ liệu $(x_1,y_1),(x_2,y_2), ... ,(x_N,y_N)$ với các $x_i$ khác nhau đôi một, luôn tìm được một đa thức $P(x)$ có bậc không vượt quá $N-1$ sao cho $P(x_i) = y_i, \forall i = 1,2, ... , N$. Cụ thể $P(x)$ được biểu diễn dưới dạng như sau:

$$
P(x) = \sum_{i=1}^{n} y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}
$$

- Chẳng phải điều này giống với việc ta đi tìm một mô hình phù hợp (fit) với dữ liệu trong bài toán Supervised Learning hay sao? Thậm chí điều này còn tốt hơn vì trong Supervised Learning ta chỉ cần xấp xỉ thôi.

- Sự thật là nếu một mô hình quá fit với dữ liệu thì nó sẽ gây phản tác dụng! Hiện tượng quá fit này trong Machine Learning được gọi là overfitting, là điều mà khi xây dựng mô hình, chúng ta luôn cần tránh. Để có cái nhìn đầu tiên về overfitting, chúng ta cùng xem Hình dưới đây. Có $50$ điểm dữ liệu được tạo bằng một đa thức bậc ba cộng thêm nhiễu. Tập dữ liệu này được chia làm hai, $30$ điểm dữ liệu màu đỏ cho training data, $20$ điểm dữ liệu màu vàng cho test data. Đồ thị của đa thức bậc ba này được cho bởi đường màu xanh lục. Bài toán của chúng ta là giả sử ta không biết mô hình ban đầu mà chỉ biết các điểm dữ liệu, hãy tìm một mô hình “tốt” để mô tả dữ liệu đã cho.


```python
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(4)
from sklearn import datasets, linear_model

N = 30
N_test = 20 
X = np.random.rand(N, 1)*5
y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)

X_test = (np.random.rand(N_test,1) - 1/8) *10
y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)

def buildX(X, d = 2):
    res = np.ones((X.shape[0], 1))
    for i in xrange(1, d+1):
        res = np.concatenate((res, X**i), axis = 1)
    return res 

def myfit(X, y, d):
    Xbar = buildX(X, d)
    regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
    regr.fit(Xbar, y)

    w = regr.coef_
    # Display result
    w_0 = w[0][0]
    w_1 = w[0][1]
    x0 = np.linspace(-2, 7, 200, endpoint=True)
    y0 = np.zeros_like(x0)
    ytrue = 5*(x0 - 2)*(x0-3)*(x0-4)
    for i in xrange(d+1):
        y0 += w[0][i]*x0**i

    # Draw the fitting line 
    plt.scatter(X.T, y.T, c = 'r', s = 40, label = 'Training samples')     # data 
    print(X_test.shape, y_test.shape)
    plt.scatter(X_test.T, y_test.T, c = 'y', s = 40, label = 'Test samples')     # data 
    
    plt.plot(x0, y0, 'b', linewidth = 2, label = "Trained model")   # the fitting line
    plt.plot(x0, ytrue, 'g', linewidth = 2, label = "True model")   # the fitting line
    plt.xticks([], [])
    plt.yticks([], [])
    if d < 3:
        str1 = 'Underfitting'
    elif d > 4:
        str1 = 'Overfitting'
    else:
        str1 = 'Good fit'
    str0 = 'Degree = ' + str(d) + ': ' + str1
    plt.title(str0)
    plt.axis([-4, 10, np.amin(y_test) - 100, np.amax(y) + 100])
    plt.legend(loc="best")
    
    fn = 'linreg_' + str(d) + '.png'
    
    plt.xlabel('$x$', fontsize = 20);
    plt.ylabel('$y$', fontsize = 20);
    
    plt.savefig(fn, bbox_inches='tight', dpi = 600)
    
    plt.show()
    print(w)

myfit(X, y, 1)
myfit(X, y, 2)
myfit(X, y, 3)
myfit(X, y, 4)
myfit(X, y, 8)
myfit(X, y, 16)
```

![anh](./image/326.png)

- Với những gì chúng ta đã biết từ bài Linear Regression, với loại dữ liệu này, chúng ta có thể áp dụng Polynomial Regression. Bài toán này hoàn toàn có thể được giải quyết bằng Linear Regression với dữ liệu mở rộng cho một cặp điểm $(x,y)$ là $(\mathbf{x},y)$ với $\mathbf{x} = [1,x,x^2,x^3,...,x^d]^T$ cho đa thức bậc $d$. Điều quan trọng là chúng ta cần tìm bậc $d$ của đa thức cần tìm.
- Rõ ràng là một đa thức bậc không vượt quá $29$ có thể fit được hoàn toàn với $30$ điểm trong training data. Chúng ta cùng xét vài giá trị $d = 2,4,8,16$. Với $d = 2$, mô hình không thực sự tố vì mô hình dự đoán quá khác so với mô hình thực. Trong trường hợp này, ta nói mô hình bị underfitting. Với $d = 8$, với các điểm dữ liệu trong khoảng của training data, mô hình dự đoán và mô hình thực là khá giống nhau. Tuy nhiên, về phía phải, đa thức bậc $8$ cho kết quả hoàn toàn ngược với xu hướng của dữ liệu. Điều tương tự xảy ra trong trường hợp $d= 16$. Đa thức bậc $16$ này quá fit dữ liệu trong khoảng đang xét, và quá fit, tức không được mượt trong khoảng dữ liệu training. Việc quá fit trong trường hợp bậc 16 không tốt vì mô hình đang cố gắng mô tả nhiễu hơn là dữ liệu. Hai trường hợp đa thức bậc cao này được gọi là Overfitting.

> Nếu bạn nào biết về Đa thức nội suy Lagrange thì có thể hiểu được hiện tượng sai số lớn với các điểm nằm ngoài khoảng của các điểm đã cho. Đó chính là lý do phương pháp đó có từ “nội suy”, với các trường hợp “ngoại suy”, kết quả thường không chính xác.

- Với $d = 4$, ta được mô hình dự đoán khá giống với mô hình thực. Hệ số bậc cao nhất tìm được rất gần với $0$, vì vậy đa thức bậc 4 này khá gần với đa thức bậc $3$ ban đầu. Đây chính là một mô hình tốt.

- Overfitting là hiện tượng mô hình tìm được quá khớp với dữ liệu training. Việc quá khớp này có thể dẫn đến việc dự đoán nhầm nhiễu, và chất lượng mô hình không còn tốt trên dữ liệu test nữa. **Dữ liệu test được giả sử là không được biết trước, và không được sử dụng để xây dựng các mô hình Machine Learning.** 
- Về cơ bản, overfitting xảy ra khi mô hình quá phức tạp để mô phỏng training data. Điều này đặc biệt xảy ra khi lượng dữ liệu training quá nhỏ trong khi độ phức tạp của mô hình quá cao. Trong ví dụ trên đây, độ phức tạp của mô hình có thể được coi là bậc của đa thức cần tìm. Trong Multi-layer Perceptron, độ phức tạp của mô hình có thể được coi là số lượng hidden layers và số lượng units trong các hidden layers.
- Vậy, có những kỹ thuật nào giúp tránh Overfitting ?
- Trước hết, chúng ta cần một vài đại lượng để đánh giá chất lượng của mô hình trên training data và test data. Dưới đây là hai đại lượng đơn giản, với giả sử  $\mathbf{y}$ là đầu ra thực sự (có thể là vector), và $\hat{\mathbf{y}}$ là đầu ra dự đoán bởi mô hình:
- **Train error:** Thường là hàm mất mát áp dụng lên training data. Hàm mất mát này cần có một thừa số $\frac{1}{N_{\text{train}}}$ để tính giá trị trung bình, tức mất mát trung bình trên mỗi điểm dữ liệu. Với Regression, đại lượng này thường được định nghĩa:

$$
\text{train error} = \frac{1}{N_{\text{train}}} \sum_{\text{training set}} \|\mathbf{y} - \hat{\mathbf{y}}\|_p^2
$$

- Với $p$ thường bằng $1$ hoặc $2$. Với Classification, trung bình cộng của cross entropy có thể được sử dụng.
- **Test error:** Tương tự như trên nhưng áp dụng mô hình tìm được vào **test data**. Chú ý rằng, khi xây dựng mô hình, ta không được sử dụng thông tin trong tập dữ liệu test. Dữ liệu test chỉ được dùng để đánh giá mô hình. Với Regression, đại lượng này thường được định nghĩa:

$$
\text{test error} = \frac{1}{N_{\text{test}}} \sum_{\text{test set}} \|\mathbf{y} - \hat{\mathbf{y}}\|_p^2
$$

- với $p$ giống như $p$ trong cách tính train error phía trên. Việc lấy trung bình là quan trọng vì lượng dữ liệu trong hai tập hợp training và test có thể chênh lệch rất nhiều.
- Một mô hình được coi là tốt (fit) nếu cả train error và test error đều thấp. Nếu train error thấp nhưng test error cao, ta nói mô hình bị overfitting. Nếu train error cao và test error cao, ta nói mô hình bị underfitting. Nếu train error cao nhưng test error thấp, tôi không biết tên của mô hình này, vì cực kỳ may mắn thì hiện tượng này mới xảy ra, hoặc có chỉ khi tập dữ liệu test quá nhỏ. Chúng ta cùng đi vào phương pháp đầu tiên.

## 2. Validation

### 2.1. Validation


- Chúng ta vẫn quen với việc chia tập dữ liệu ra thành hai tập nhỏ: training data và test data. Và một điều tôi vẫn muốn nhắc lại là khi xây dựng mô hình, ta không được sử dụng test data. Vậy làm cách nào để biết được chất lượng của mô hình với unseen data (tức dữ liệu chưa nhìn thấy bao giờ)?

- Phương pháp đơn giản nhất là trích từ tập training data ra một tập con nhỏ và thực hiện việc đánh giá mô hình trên tập con nhỏ này. Tập con nhỏ được trích ra từ training set này được gọi là validation set. Lúc này, training set là phần còn lại của training set ban đầu. Train error được tính trên training set mới này, và có một khái niệm nữa được định nghĩa tương tự như trên validation error, tức error được tính trên tập validation.

> Việc này giống như khi bạn ôn thi. Giả sử bạn không biết đề thi như thế nào nhưng có 10 bộ đề thi từ các năm trước. Để xem trình độ của mình trước khi thi thế nào, có một cách là bỏ riêng một bộ đề ra, không ôn tập gì. Việc ôn tập sẽ được thực hiện dựa trên 9 bộ còn lại. Sau khi ôn tập xong, bạn bỏ bộ đề đã để riêng ra làm thử và kiểm tra kết quả, như thế mới “khách quan”, mới giống như thi thật. 10 bộ đề ở các năm trước là “toàn bộ” training set bạn có. Để tránh việc học lệch, học tủ theo chỉ 10 bộ, bạn tách 9 bộ ra làm training set thật, bộ còn lại là validation test. Khi làm như thế thì mới đánh giá được việc bạn học đã tốt thật hay chưa, hay chỉ là học tủ. Vì vậy, Overfitting còn có thể so sánh với việc Học tủ của con người.


- Với khái niệm mới này, ta tìm mô hình sao cho cả train error và validation error đều nhỏ, qua đó có thể dự đoán được rằng test error cũng nhỏ. Phương pháp thường được sử dụng là sử dụng nhiều mô hình khác nhau. Mô hình nào cho validation error nhỏ nhất sẽ là mô hình tốt.

- Thông thường, ta bắt đầu từ mô hình đơn giản, sau đó tăng dần độ phức tạp của mô hình. Tới khi nào validation error có chiều hướng tăng lên thì chọn mô hình ngay trước đó. Chú ý rằng mô hình càng phức tạp, train error có xu hướng càng nhỏ đi.

- Hình dưới đây mô tả ví dụ phía trên với bậc của đa thức tăng từ 1 đến 8. Tập validation bao gồm 10 điểm được lấy ra từ tập training ban đầu.


```python
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(5)
from sklearn import datasets, linear_model

N = 50
N_test = 10 
X = np.random.rand(N, 1)*5
y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)

# split train to train + valid use train_test_split 
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split( \
     X, y, test_size=0.33, random_state=0)

# generate test data
X_test = (np.random.rand(N_test,1)) *5
y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)

def buildX(X, d = 2):
    res = np.ones((X.shape[0], 1))
    for i in xrange(1, d+1):
        res = np.concatenate((res, X**i), axis = 1)
    return res 

def poly(a, x):
    """
    return a[0] + a[1]*x + a[2]*x**2 + .... 
    """
    res = np.zeros_like(x)
    for i in xrange(len(a) - 1, -1, -1):
        res = res*x + a[i] 
    return res 

x = 2
a = [1, 2, 3, 4]
print(poly(a, x))

def MSE(x, y, w):     
    d = len(w) - 1
    y_pred = poly(w, x)
    return np.mean(np.abs(y - y_pred))

def myfit(d):
    Xbar = buildX(X_train, d)
    regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
    regr.fit(Xbar, y_train)

    w = regr.coef_[0]
    train_err = MSE(X_train, y_train, w)
    valid_err = MSE(X_valid, y_valid, w)
    test_err = MSE(X_test, y_test, w)
                    
    return (train_err, valid_err, test_err)
    
    
Train_error = []
Test_error = []
Valid_error = []
degree = 9
for d in xrange(1, degree):
    (train_err, valid_err, test_err) = myfit(d) 
    Train_error.append(train_err)
    Test_error.append(test_err)
    Valid_error.append(valid_err)

degree = xrange(1, degree)

fig = plt.figure()
# fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)

plt.plot(degree, Train_error, 'b-', linewidth = 2, label = "Train error")
plt.plot(degree, Valid_error, 'r-', linewidth = 2, label = "Validation error")
plt.plot(degree, Test_error, 'g-', linewidth = 2, label = "Test error")
plt.legend(loc="best")
plt.xlabel('degree')
plt.ylabel('error')


plt.plot([3.5, 3.5], [6, 10], color='k', linestyle='-', linewidth=2)
ax.text(0.7, 0.01, 'overfitting',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='purple', fontsize=15)
ax.text(0.27, 0.01, 'underfitting',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='purple', fontsize=15)
plt.savefig('linreg_val.png', bbox_inches='tight', dpi = 600)
plt.show()
```

![anh](./image/327.png)

- Chúng ta hãy tạm chỉ xét hai đường màu lam và đỏ, tương ứng với train error và validation error. Khi bậc của đa thức tăng lên, train error có xu hướng giảm. Điều này dễ hiểu vì đa thức bậc càng cao, dữ liệu càng được fit. Quan sát đường màu đỏ, khi bậc của đa thức là 3 hoặc 4 thì validation error thấp, sau đó tăng dần lên. Dựa vào validation error, ta có thể xác định được bậc cần chọn là 3 hoặc 4. Quan sát tiếp đường màu lục, tương ứng với test error, thật là trùng hợp, với bậc bằng 3 hoặc 4, test error cũng đạt giá trị nhỏ nhất, sau đó tăng dần lên. Vậy cách làm này ở đây đã tỏ ra hiệu quả.

- Việc không sử dụng test data khi lựa chọn mô hình ở trên nhưng vẫn có được kết quả khả quan vì ta giả sử rằng validation data và test data có chung một đặc điểm nào đó. Và khi cả hai đều là unseen data, error trên hai tập này sẽ tương đối giống nhau.

- Nhắc lại rằng, khi bậc nhỏ (bằng 1 hoặc 2), cả ba error đều cao, ta nói mô hình bị underfitting.

### 2.2. Cross-validation

- Trong nhiều trường hợp, chúng ta có rất hạn chế số lượng dữ liệu để xây dựng mô hình. Nếu lấy quá nhiều dữ liệu trong tập training ra làm dữ liệu validation, phần dữ liệu còn lại của tập training là không đủ để xây dựng mô hình. Lúc này, tập validation phải thật nhỏ để giữ được lượng dữ liệu cho training đủ lớn. Tuy nhiên, một vấn đề khác nảy sinh. Khi tập validation quá nhỏ, hiện tượng overfitting lại có thể xảy ra với tập training còn lại. Có giải pháp nào cho tình huống này không?

- Câu trả lời là cross-validation.

- Cross validation là một cải tiến của validation với lượng dữ liệu trong tập validation là nhỏ nhưng chất lượng mô hình được đánh giá trên nhiều tập validation khác nhau. Một cách thường đường sử dụng là chia tập training ra $k$ tập con không có phần tử chung, có kích thước gần bằng nhau. Tại mỗi lần kiểm thử , được gọi là run, một trong số $k$ tập con được lấy ra làm validate set. Mô hình sẽ được xây dựng dựa vào hợp của $k - 1$ tập con còn lại. Mô hình cuối được xác định dựa trên trung bình của các train error và validation error. Cách làm này còn có tên gọi là **k-fold cross validation**. Khi $k$ bằng với số lượng phần tử trong tập training ban đầu, tức mỗi tập con có đúng $1$ phần tử, ta gọi kỹ thuật này là **leave-one-out**.

- Sklearn hỗ trợ rất nhiều phương thức cho phân chia dữ liệu và tính toán scores của các mô hình. Bạn đọc có thể xem thêm tại [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html).

## 3. Regularization
  
- Một nhược điểm lớn của cross-validation là số lượng training runs tỉ lệ thuận với $k$. Điều đáng nói là mô hình polynomial như trên chỉ có một tham số cần xác định là bậc của đa thức. Trong các bài toán Machine Learning, lượng tham số cần xác định thường lớn hơn nhiều, và khoảng giá trị của mỗi tham số cũng rộng hơn nhiều, chưa kể đến việc có những tham số có thể là số thực. Như vậy, việc chỉ xây dựng một mô hình thôi cũng là đã rất phức tạp rồi. Có một cách giúp số mô hình cần huấn luyện giảm đi nhiều, thậm chí chỉ một mô hình. Cách này có tên gọi chung là regularization.

- Regularization, một cách cơ bản, là thay đổi mô hình một chút để tránh overfitting trong khi vẫn giữ được tính tổng quát của nó (tính tổng quát là tính mô tả được nhiều dữ liệu, trong cả tập training và test). Một cách cụ thể hơn, ta sẽ tìm cách di chuyển nghiệm của bài toán tối ưu hàm mất mát tới một điểm gần nó. Hướng di chuyển sẽ là hướng làm cho mô hình ít phức tạp hơn mặc dù giá trị của hàm mất mát có tăng lên một chút.

- Một kỹ thuật rất đơn giản là early stopping.

### 3.1.  Early Stopping

- Trong nhiều bài toán Machine Learning, chúng ta cần sử dụng các thuật toán lặp để tìm ra nghiệm, ví dụ như Gradient Descent. Nhìn chung, hàm mất mát giảm dần khi số vòng lặp tăng lên. Early stopping tức dừng thuật toán trước khi hàm mất mát đạt giá trị quá nhỏ, giúp tránh overfitting.

- Vậy dừng khi nào là phù hợp?

- Một kỹ thuật thường được sử dụng là tách từ training set ra một tập validation set như trên. Sau một (hoặc một số, ví dụ 50) vòng lặp, ta tính cả train error và validation error, đến khi validation error có chiều hướng tăng lên thì dừng lại, và quay lại sử dụng mô hình tương ứng với điểm và validation error đạt giá trị nhỏ.

![anh](./image/328.png)


### 3.2. Thêm số hạng vào hàm mất mát

- Kỹ thuật regularization phổ biến nhất là thêm vào hàm mất mát một số hạng nữa. Số hạng này thường dùng để đánh giá độ phức tạp của mô hình. Số hạng này càng lớn, thì mô hình càng phức tạp. Hàm mất mát mới này thường được gọi là **regularized loss function**, thường được định nghĩa như sau:

$$
J_{\text{reg}}(\theta) = J(\theta) + \lambda R(\theta)
$$

- Nhắc lại rằng $\theta$ được dùng để ký hiệu các biến trong mô hình, chẳng hạn như các hệ số  $\mathbf{w}$ trong Neural Networks. $J(\theta)$ ký hiệu cho hàm mất mát (loss function) và $R(\theta)$ là số hạng regularization. $\lambda$ thường là một số dương để cân bằng giữa hai đại lượng ở vế phải.

- Việc tối thiểu regularized loss function, nói một cách tương đối, đồng nghĩa với việc tối thiểu cả loss function và số hạng regularization. Tôi dùng cụm “nói một cách tương đối” vì nghiệm của bài toán tối ưu loss function và regularized loss function là khác nhau. Chúng ta vẫn mong muốn rằng sự khác nhau này là nhỏ, vì vậy tham số regularization (regularization parameter) $\lambda$ thường được chọn là một số nhỏ để biểu thức regularization không làm giảm quá nhiều chất lượng của nghiệm.

- Với các mô hình Neural Networks, một số kỹ thuật regularization thường được sử dụng là:

### 3.3. $l_2$ regularization

- Trong kỹ thuật này:

$$
R(\mathbf{w}) = \|\mathbf{w}\|_2^2
$$

- Hàm số này có một vài đặc điểm đang lưu ý:
1. Thứ nhất $\|\mathbf{w}\|_2^2$ là một hàm số rất mượt, tức có đạo hàm tại mọi điểm, đạo hàm của nó đơn giản là $\mathbf{w}$, vì vậy đạo hàm của regularized loss function cũng rất dễ tính, chúng ta có thể hoàn toàn dùng các phương pháp dựa trên gradient để cập nhật nghiệm. Cụ thể:

$$
\frac{\partial J_{\text{reg}}}{\partial \mathbf{w}} = \frac{\partial J}{\partial \mathbf{w}} + \lambda \mathbf{w}
$$

2. Thứ hai, việc tối thiểu $\|\mathbf{w}\|_2^2$ đồng nghĩa với việc khiến cho các giá trị của hệ số  $\mathbf{w}$ trở nên nhỏ gần với 0. Với Polynomial Regression, việc các hệ số này nhỏ có thể giúp các hệ số ứng với các số hạng bậc cao là nhỏ, giúp tránh overfitting. Với Multi-layer Perceptron, việc các hệ số này nhỏ giúp cho nhiều hệ số trong các ma trận trọng số là nhỏ. Điều này tương ứng với việc số lượng các hidden units hoạt động (khác không) là nhỏ, cũng giúp cho MLP tránh được hiện tượng overfitting.

- $l_2$ regularization  là kỹ thuật được sử dụng nhiều nhất để giúp Neural Networks tránh được overfitting. Nó còn có tên gọi khác là **weight decay**. Decay có nghĩa là tiêu biến.

- Trong Xác suất thống kê, Linear Regression với $l_2$ regularization được gọi là **Ridge Regression**. Hàm mất mát của Ridge Regression có dạng:

$$
J(\mathbf{w}) = \frac{1}{2} \|\mathbf{y} - \mathbf{Xw}\|_2^2 + \lambda \|\mathbf{w}\|_2^2
$$

- Trong đó, số hạng đầu tiên ở vế phải chính là hàm mất mát của Linear Regression. Số hạng thứ hai chính là phần regularization.

### 3.4. Tikhonov regularization


$$
\lambda R(\mathbf{w}) = \|\mathbf{\Gamma w}\|_2^2
$$


- Với $\Gamma$ (viết hoa của gamma) là một ma trận. Ma trận $\Gamma$ hay được dùng nhất là ma trận đường chéo. Nhận thấy rằng $l_2$ regularization chính là một trường hợp đặc biệt của Tikhonov regularization với $\Gamma = \lambda \mathbf{I}$ với $\mathbf{I}$ là ma trận đơn vị (the identity matrix), tức các phần tử trên đường chéo của $\Gamma$ là như nhau.

- Khi các phần tử trên đường chéo của $\Gamma$ là khác nhau, ta có một phiên bản gọi là weighted $l_2$ regularization, tức đánh trọng số khác nhau cho mỗi phần tử trong $\mathbf{w}$
. Phần tử nào càng bị đánh trọng số cao thì nghiệm tương ứng càng nhỏ (để đảm bảo rằng hàm mất mát là nhỏ). Với Polynomial Regression, các phần tử ứng với hệ số bậc cao sẽ được đánh trọng số cao hơn, khiến cho xác suất để chúng gần $0$ là lớn hơn.


### 3.5. Regularizers for sparsity

- Trong nhiều trường hợp, ta muốn các hệ số thực sự bằng 0 chứ không phải là nhỏ gần $0$ như $l_2$ regularization đã làm phía trên. Lúc đó, có một regularization khác được sử dụng, đó là $l_0$ regularization:

$$
R(\mathbf{W}) = \|\mathbf{w}\|_0
$$


- Norm $0$ không phải là một norm thực sự mà là giả norm. Norm $0$ của một vector là số các phần tử khác không của vector đó. Khi norm $0$ nhỏ, tức rất nhiều phần tử trong vector đó bằng 0, ta nói vector đó là sparse.

- Việc giải bài toán tổi thiểu norm $0$ nhìn chung là khó vì hàm số này không convex, không liên tục. Thay vào đó, norm $1$ thường được sử dụng:


$$
R(\mathbf{W}) = \|\mathbf{w}\|_1 = \sum_{i=0}^{d} |w_i|
$$

- Norm $1$ là tổng các trị tuyệt đối của tất cả các phần tử. Người ta đã chứng minh được rằng tối thiểu norm $1$ sẽ dẫn tới nghiệm có nhiều phần tử bằng $0$. Ngoài ra, vì norm $1$ là một norm thực sự (proper norm) nên hàm số này là convex, và hiển nhiên là liên tục, việc giải bài toán này dễ hơn việc giải bài toán tổi thiểu norm $0$.
- Trong Thống Kê, việc sử dụng $l_1$ regularization còn được gọi là [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) (Least Absolute Shrinkage and Selection Operator).
-Khi cả $l_2$ và $l_1$ regularization được sử dụng, ta có mô hình gọi là [Elastic Net Regression](https://en.wikipedia.org/wiki/Elastic_net_regularization).

## 4. Tóm tắt nội dung

- Một mô hình mô tốt là mộ mô hình có tính tổng quát, tức mô tả được dữ liệu cả trong lẫn ngoài tập training. Mô hình chỉ mô tả tốt dữ liệu trong tập training được gọi là **overfitting**.

- Để tránh overfitting, có rất nhiều kỹ thuật được sử dụng, điển hình là **cross-validation** và **regularization**. Trong Neural Networks, weight decay và dropout thường được dùng.

## 5. Tài liệu tham khảo

- [1] [Overfitting](https://machinelearningcoban.com/2017/03/04/overfitting/)