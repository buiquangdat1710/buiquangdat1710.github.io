---
title: "A Recipe for Training Neural Networks"
date: 2025-01-28 00:00:00  + 0800
categories: [Deep Learning]
tags: [deep learning]
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


Blog này được tôi dịch sang tiếng việt từ blog [A Recipe for Training Neural Networks - Andrej Karpathy](https://karpathy.github.io/2019/04/25/recipe/). Nếu có thời gian, hãy xem thử blog chất lượng của Andrej Karpathy - đồng sáng lập OpenAI nhé.


- Một vài tuần trước, tôi đã đăng một tweet về ["the most common neural net mistakes"](https://x.com/karpathy/status/1013244313327681536?lang=en), liệt kê một vài vấn đề thường gặp liên quan đến việc training neural nets. Tweet đó nhận được nhiều sự chú ý hơn tôi mong đợi (bao gồm cả một [webinar](https://www.bigmarker.com/missinglink-ai/PyTorch-Code-to-Unpack-Andrej-Karpathy-s-6-Most-Common-NN-Mistakes) :)). Rõ ràng, rất nhiều người đã từng trải qua khoảng cách lớn giữa "here is how a convolutional layer works" và "our convnet achieves state of the art results".

- Vì vậy, tôi nghĩ có thể là một ý hay khi "phủi bụi" blog của mình để mở rộng tweet này thành một bài viết dài hơn, đúng như chủ đề này xứng đáng. Tuy nhiên, thay vì đi sâu vào liệt kê các lỗi phổ biến hơn hoặc mở rộng chúng, tôi muốn đào sâu hơn một chút và nói về cách chúng ta có thể tránh những lỗi này hoàn toàn (hoặc sửa chúng rất nhanh). Bí quyết để làm như vậy là tuân theo một quy trình nhất định, theo như tôi biết thì quy trình này không thường được ghi chép lại. Chúng ta hãy bắt đầu với hai quan sát quan trọng thúc đẩy quy trình này.

## 1. Huấn luyện một mạng neuron là một sự trừu tượng không rõ ràng (Neural net training is a leaky)

- Người ta cho rằng việc bắt đầu với đào tạo mạng nơ-ron là dễ dàng. Nhiều thư viện và framework tự hào khi hiển thị các đoạn code thần kỳ dài 30 dòng để giải quyết vấn đề dữ liệu của bạn, tạo ấn tượng (thực tế không phải như vậy) rằng mọi thứ đều có thể cắm vào và chạy ngay. Ví dụ phổ biến như sau:

```python
>>> your_data = # thêm bộ dữ liệu tuyệt vời của bạn vào đây
>>> model = SuperCrossValidator(SuperDuper.fit, your_data, ResNet50, SGDOptimizer)
# chinh phục thế giới ở đây
```

- Những thư viện và ví dụ này kích hoạt phần não bộ của chúng ta quen thuộc với phần mềm tiêu chuẩn - một nơi mà các API gọn gàng và trừu tượng thường đạt được. Để minh họa, thư viện [Requests](https://docs.python-requests.org/en/latest/) được sử dụng:

```python
>>> r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
>>> r.status_code
200
```


- Điều đó thật tuyệt vời! Một nhà phát triển dũng cảm đã giúp bạn xử lý những thứ phức tạp như chuỗi truy vấn, URL, yêu cầu GET/POST, kết nối HTTP, và giấu đi sự phức tạp chỉ trong vài dòng mã. Đây là điều chúng ta thường quen thuộc và mong đợi. Nhưng thật không may, neural nets không hoạt động theo cách này. Chúng không phải là "công nghệ có sẵn" mà bạn có thể sử dụng ngay. Ngay khi bạn chỉ thay đổi một chút trong việc đào tạo mô hình ImageNet, bạn sẽ nhận ra điều này.

- Tôi đã cố gắng làm rõ quan điểm này trong bài viết của mình “Yes you should understand backprop”, bằng cách chỉ ra rằng thuật toán lan truyền ngược chỉ là một "leaky abstraction". Tuy nhiên, thực tế còn tệ hơn nhiều. Backpropagation + SGD không thần kỳ làm cho mạng hội tụ nhanh hơn. Batch normalization cũng không làm cho nó hội tụ thần kỳ. RNNs cũng không chỉ cần “thêm vào là chạy”. Và việc bạn có thể đặt bài toán của mình theo cách reinforcement learning (RL) không có nghĩa là bạn nên làm vậy. Nếu bạn sử dụng công nghệ mà không hiểu cách nó hoạt động, bạn rất dễ thất bại. Điều này dẫn chúng ta đến...


## 2. Neural net training fails silently

- Khi bạn làm sai hoặc cấu hình sai code, thường bạn sẽ nhận được một lỗi hiển thị rõ ràng. Ví dụ: bạn nhập nhầm một số nguyên vào chỗ cần một chuỗi ký tự, hàm chỉ nhận được 3 tham số thay vì 4, hoặc một khóa nào đó không tồn tại. Những lỗi này rất rõ ràng và bạn thường có thể tạo bài kiểm thử (unit test) để kiểm tra các chức năng.
- Nhưng khi  training neural nets, mọi thứ trở nên phức tạp hơn nhiều. Code của bạn có thể hoàn toàn đúng về cú pháp, nhưng toàn bộ hệ thống lại hoạt động không như mong muốn, và điều đó rất khó phát hiện. Cái “possible error surface” rất rộng (bao gồm cả lỗi logic lẫn lỗi cú pháp), và việc kiểm tra chúng là rất khó. Ví dụ, bạn có thể quên lật labels khi bạn đã lật các hình ảnh (features) tring quá trình tăng cường dữ liệu. Mô hình của bạn vẫn có thể hoạt động ổn bởi vì nó có thể tự phát hiện ảnh bị lật và sửa lại dự đoán. Hoặc, mô hình autoregressive của bạn vô tình đưa dữ liệu sai làm đầu vào do lỗi nhỏ khó phát hiện. Có thể bạn cố gắng gọt gradients nhưng lại xóa mất hàm loss, khiến một số ví dụ ngoại lệ bị bỏ qua trong khi huấn luyện. Bạn sử dụng trọng số từ một checkpoint đã được huấn luyện trước, nhưng lại không dùng đúng trung bình (mean) ban đầu. Hoặc bạn chỉ làm hỏng cài đặt cho regularization strengths, learning rate, decay rate, model size,... Kết quả là, mạng nơ-ron bị cấu hình sai sẽ không luôn đưa ra lỗi rõ ràng. Nếu may mắn, nó sẽ báo lỗi; còn không, nó sẽ tiếp tục huấn luyện nhưng hoạt động âm thầm... và sai.

- Kết luận: (và điều này rất quan trọng để nhấn mạnh) cách tiếp cận kiểu "nhanh và liều" với việc huấn luyện mạng nơ-ron không hiệu quả và chỉ dẫn đến khổ sở. Tuy nhiên, "khổ sở" là một phần hoàn toàn tự nhiên trong quá trình làm cho mạng hoạt động tốt, và bạn có thể giảm bớt điều đó bằng cách cẩn thận, phòng thủ, nghi ngờ, và bị ám ảnh với việc trực quan hóa mọi thứ có thể.

- Những phẩm chất quan trọng nhất để thành công trong việc học sâu là kiên nhẫn và chú ý đến chi tiết.


# Công thức 


- Dựa trên hai thực tế trên, tôi đã phát triển một quy trình cụ thể cho bản thân khi áp dụng neural net vào một vấn đề mới, và tôi sẽ cố gắng mô tả nó. Bạn sẽ thấy rằng quy trình này tuân theo hai nguyên tắc trên một cách nghiêm túc. Cụ thể, nó bắt đầu từ đơn giản đến phức tạp, và ở mỗi bước, chúng ta đưa ra các giả thuyết cụ thể về điều gì sẽ xảy ra, sau đó hoặc xác nhận bằng thử nghiệm hoặc điều tra cho đến khi tìm thấy vấn đề. Điều chúng ta cố gắng ngăn chặn là việc giới thiệu quá nhiều sự phức tạp “chưa được xác minh” cùng lúc, vì điều đó sẽ dễ dẫn đến lỗi/sai cấu hình mà có thể mất rất nhiều thời gian để phát hiện (nếu có thể phát hiện được). Nếu bạn viết code neural net như thể đang huấn luyện một cái mới, bạn nên sử dụng learning rate rất nhỏ và thử nghiệm từng bước, sau đó đánh giá toàn bộ tập test sau mỗi lần lặp.

## 1.  Become one with the data

- Bước đầu tiên không phải là viết code mà là kiểm tra dữ liệu. Điều này rất quan trọng! Tôi thường dành rất nhiều thời gian (tính bằng giờ) để xem xét hàng nghìn mẫu dữ liệu, hiểu phân bố của chúng và tìm kiếm các đặc điểm bất thường.

- Ví dụ:
  - Có dữ liệu nào bị trùng lặp không?
  - Có hình ảnh hay nhãn nào bị lỗi không?
  - Dữ liệu có bị mất cân bằng không?
  - Dữ liệu có chứa thiên lệch nào không?

- Tôi cũng thử tự phân loại dữ liệu bằng mắt thường để xem cần loại kiến trúc nào. Ví dụ:
  - Chỉ cần đặc trưng cục bộ hay cần cả bối cảnh tổng thể ?
  - Biến đổi dữ liệu có lớn không? Có thể loại bỏ biến đổi nào trước khi đưa vào model ?
  - Ảnh có thể giảm kích thước được không mà vẫn giữ đủ thông tin ?
  - Nhãn có nhiều nhiễu không ?
- Ngoài ra, neural net thực chất là một phiên bản nén của dữ liệu, nên nếu model dự đoán sai, hãy kiểm tra xem sai lầm đó đến từ đâu. Nếu model đưa ra dự đoán không khớp với những gì bạn quan sát được từ dữ liệu, thì chắc chắn có vấn đề.

- Cuối cùng, hãy viết một số đoạn code đơn giản để tìm kiếm, lọc, hoặc sắp xếp dữ liệu dựa trên các yếu tố như loại nhãn, số lượng annotation, kích thước annotation,... và vẽ biểu đồ để xem phân bố dữ liệu. Những giá trị bất thường (outliers) thường sẽ giúp phát hiện lỗi trong dữ liệu hoặc quá trình tiền xử lý.

## 2. Set up the end-to-end training/evaluation skeleton + get dumb baselines

- Sau khi đã hiểu dữ liệu, chúng ta có thể ngay lập tức huấn luyện một model hoành tráng như Multi-scale ASPP FPN ResNet không? Không! Đó là con đường dẫn đến đau khổ.

- Bước tiếp theo là thiết lập một quy trình huấn luyện + đánh giá hoàn chỉnh để đảm bảo rằng model hoạt động đúng bằng một loạt thử nghiệm. Ở giai đoạn này, tốt nhất nên chọn một model đơn giản mà bạn gần như không thể làm sai, ví dụ như một linear classifier hoặc một ConvNet nhỏ. Sau đó, huấn luyện model, theo dõi loss, accuracy, kiểm tra dự đoán của model và thực hiện các thí nghiệm nhỏ để kiểm chứng từng giả thuyết.

#### Mẹo & lưu ý quan trọng:
- **fix random seed:** Luôn đặt giá trị random seed cố định để khi chạy code hai lần, bạn sẽ nhận được kết quả giống nhau. Điều này giúp loại bỏ các yếu tố ngẫu nhiên và giúp bạn giữ được sự tỉnh táo.
- **simplify:** Tắt tất cả những thứ không cần thiết, ví dụ như data augmentation. Data augmentation là một phương pháp regularization hữu ích, nhưng ở giai đoạn này, nó có thể chỉ làm phức tạp vấn đề mà thôi.
- **add significant digits to your eval:** Khi vẽ test loss, hãy chạy đánh giá trên toàn bộ test set thay vì chỉ hiển thị giá trị loss trung bình trên một batch. Điều này giúp đảm bảo tính chính xác thay vì chỉ dựa vào smoothing trên TensorBoard.
- **verify loss @ init:** Kiểm tra giá trị loss ban đầu xem có đúng không. Loss ban đầu là loss mà mô hình chưa được huấn luyện gì (mô hình dự đoán ngẫu nhiên). Ví dụ, nếu layer cuối cùng là softmax, thì loss lúc khởi tạo phải là -log(1/n_classes). Các giá trị tương tự có thể được tính toán cho L2 regression, Huber loss, v.v.
- **init well:** Khởi tạo lớp cuối cùng có trọng số thật chính xác. Ví dụ nếu bạn đang làm hồi quy một vài giá trị có trung bình là 50 thì nên khởi tạo giá trị bias cuối cùng là 50. Nếu bạn có một dataset bị mất cân bằng với tỉ lệ 1:10 cho positive:negative, đặt bias của logits sao cho mạng của bạn dự đoán xác suất 0.1 ở bước khởi tạo. Điều này giúp quá trình hội tụ nhanh hơn và tránh hiện tượng loss giảm chậm trong vài epoch đầu tiên.
- **human baseline:** Theo dõi các chỉ số mà con người có thể hiểu và kiểm tra được, chẳng hạn như accuracy. Nếu có thể, hãy đánh giá kết quả của model bằng cách so sánh với độ chính xác của con người trên cùng một tập dữ liệu.
- **input-independent baseline:** Huấn luyện một model mà đầu vào không liên quan đến dữ liệu thực tế (ví dụ: đặt toàn bộ input về 0). Model này phải có hiệu suất kém hơn so với model thực tế. Nếu không, có thể model chưa học được gì từ dữ liệu cả.
- **overfit one batch:** Huấn luyện một batch nhỏ (chỉ một vài mẫu, thậm chí chỉ 2 mẫu). Thêm các lớp hoặc bộ lọc để đảm bảo rằng model có thể đạt được loss rất nhỏ (gần 0). Nếu không thể, có thể model của bạn đang gặp vấn đề. Hãy vẽ cả nhãn thật và dự đoán trên cùng một biểu đồ để kiểm tra xem model có học đúng hay không.
- **verify decreasing training loss:** Ở giai đoạn này, model của bạn có thể đang underfitting vì dữ liệu còn đơn giản. Hãy thử tăng một chút độ phức tạp của model và kiểm tra xem training loss có giảm như mong đợi không.
- **visualize just before the net:** Nơi tốt nhất để kiểm tra dữ liệu là ngay trước khi nó đi vào model, tức là y_hat = model(x). Việc trực quan hóa dữ liệu ở thời điểm này giúp bạn kiểm tra chính xác những gì đang đi vào model, giúp phát hiện lỗi trong quá trình tiền xử lý hoặc augmentation.
- **visualize prediction dynamics:** Hãy theo dõi dự đoán của model trên một batch cố định trong suốt quá trình huấn luyện. Việc quan sát sự thay đổi của dự đoán theo thời gian có thể giúp bạn phát hiện bất ổn, chẳng hạn như model “vật lộn” để khớp với dữ liệu hoặc nhạy cảm quá mức với noise. Learning rate quá thấp hoặc quá cao cũng có thể nhận biết bằng cách này.
- **use backprop to chart dependencies:** Code deep learning thường có nhiều phép toán vector hóa và broadcast phức tạp. Một lỗi phổ biến là sử dụng view thay vì transpose/permute, khiến dữ liệu bị trộn lẫn giữa các batch. Thường thì model vẫn có thể huấn luyện được vì nó học cách bỏ qua dữ liệu sai, nhưng để kiểm tra lỗi này, bạn có thể đặt loss thành một hàm đơn giản (ví dụ tổng các đầu ra), chạy backprop về input và kiểm tra xem gradient có giá trị khác 0 ở vị trí cần thiết không.
- **generalize a special case:** Một mẹo lập trình quan trọng là đừng cố viết ngay một hàm tổng quát phức tạp. Thay vào đó, hãy viết một phiên bản đơn giản trước, đảm bảo nó hoạt động đúng, rồi mới mở rộng nó. Điều này đặc biệt hữu ích khi vectorizing code: tôi thường viết phiên bản vòng lặp đầy đủ trước, kiểm tra kết quả, sau đó mới tối ưu hóa từng bước một

## 3. Overfit

- Ở giai đoạn này, chúng ta đã hiểu rõ về tập dữ liệu và có quy trình huấn luyện + đánh giá hoàn chỉnh. Với mỗi model, ta có thể tính toán lại một cách đáng tin cậy các chỉ số đo lường hiệu suất. Đồng thời, ta cũng có các kết quả benchmark để so sánh, bao gồm cả baseline đơn giản lẫn hiệu suất của con người (mục tiêu cao nhất). Giai đoạn này là lúc để thử nghiệm và cải thiện model.

- Cách tiếp cận của tôi để tìm model tốt gồm 2 bước:
1. Chọn một model đủ lớn để nó có thể overfit (tức là đạt loss rất thấp trên tập huấn luyện).
2. Điều chỉnh lại model để giảm overfit (tăng loss trên tập huấn luyện một chút nhưng giảm loss trên tập validation).

- Lý do tôi thích phương pháp này là nếu ta không thể đạt lỗi thấp với bất kỳ model nào, có thể dữ liệu hoặc code của ta có vấn đề.

### Một số mẹo quan trọng:

- **Chọn model phù hợp:** Để giảm loss trên tập huấn luyện, bạn cần một kiến trúc phù hợp. Lời khuyên của tôi: **"Đừng làm anh hùng!"** Nhiều người thích sáng tạo quá mức, thử những model phức tạp một cách không cần thiết. Đừng mắc bẫy này! Hãy tìm một bài báo uy tín, chọn model đơn giản nhất của họ và sử dụng trước. Ví dụ, nếu làm bài toán phân loại ảnh, đừng cố sáng tạo ngay từ đầu, cứ dùng ResNet-50 trước đã. Sau này, nếu cần, bạn có thể thử nghiệm với model phức tạp hơn để cải thiện.
- **Adam is safe:** Ở giai đoạn đầu, tôi thường dùng Adam với learning rate 3e-4. Adam có khả năng thích nghi tốt với hyperparameter, ngay cả khi bạn chọn learning rate chưa chuẩn. Tuy nhiên, nếu dùng SGD được tối ưu tốt thì nó vẫn có thể vượt trội hơn. (Lưu ý: nếu dùng RNN hoặc model dạng sequence, Adam là lựa chọn phổ biến).
- **Chỉ thêm độ phức tạp một chút mỗi lần:** Nếu bạn muốn thêm nhiều tín hiệu đầu vào (features) cho model, hãy thêm từng cái một và kiểm tra xem có cải thiện hiệu suất không. Đừng thêm tất cả cùng lúc mà không kiểm tra! Một cách khác là thử với tập dữ liệu nhỏ trước, rồi tăng kích thước sau.
- **Cẩn thận với learning rate decay:** Nếu bạn dùng code từ dự án khác, hãy kiểm tra learning rate decay. Trong một số trường hợp, learning rate có thể giảm quá nhanh khiến model không học được nữa. Ví dụ, với ImageNet, learning rate có thể giảm 10 lần sau mỗi 30 epochs, nhưng nếu bạn không train trên ImageNet thì có thể không cần làm vậy. Cá nhân tôi thường vô hiệu hóa learning rate decay ở giai đoạn đầu và tự điều chỉnh thủ công.

## 4. Regularize

- Bây giờ, chúng ta đã có một model lớn đủ tốt trên tập huấn luyện. Đến lúc điều chỉnh để giảm overfit và tăng độ chính xác trên tập validation, dù phải đánh đổi một chút độ chính xác trên tập huấn luyện. Dưới đây là một số mẹo hữu ích:
- **Thêm dữ liệu thực tế:** Cách tốt nhất để regularize model trong thực tế là thu thập thêm dữ liệu huấn luyện. Nhiều người mất quá nhiều thời gian tối ưu trên tập dữ liệu nhỏ thay vì cố gắng mở rộng dữ liệu. Nếu có thể, hãy luôn ưu tiên thu thập thêm dữ liệu thực, vì điều này gần như luôn đảm bảo cải thiện hiệu suất của model. Một cách khác là sử dụng ensemble models, nhưng chỉ khi bạn có đủ tài nguyên (vì chúng tốn nhiều tài nguyên tính toán).
- **Data augmentation:** Nếu không thể có dữ liệu thực, hãy thử data augmentation (tạo dữ liệu nhân tạo). Hãy sử dụng các phương pháp mạnh mẽ hơn như xoay ảnh, cắt, thay đổi màu sắc, v.v.
- **Creative augmentation:**Nếu augmentation thông thường vẫn chưa đủ, có thể thử các phương pháp sáng tạo hơn như:
  - [Domain randomization](https://openai.com/index/learning-dexterity/) (làm dữ liệu đa dạng hơn bằng cách thay đổi ngẫu nhiên)
  - Dùng mô phỏng ([simulation](https://vladlen.info/publications/playing-data-ground-truth-computer-games/)) để tạo dữ liệu giả lập
  - Sử dụng GANs hoặc các phương pháp AI khác để tạo dữ liệu mới.

- **Pretrain nếu có thể:** Nếu bạn có một mạng pretrained, hãy tận dụng nó thay vì huấn luyện từ đầu. Điều này thường có ích, đặc biệt với bài toán có ít dữ liệu.

- **Chỉ dùng supervised pretraining:** Không nên quá kỳ vọng vào unsupervised pretraining. Trong computer vision, unsupervised pretraining không mang lại hiệu quả rõ ràng. Tuy nhiên, trong NLP (xử lý ngôn ngữ tự nhiên), nó lại rất hữu ích (ví dụ như BERT).
- **Giảm số chiều của input:** Loại bỏ các features không cần thiết để tránh overfitting. Nếu dữ liệu của bạn nhỏ, việc giữ quá nhiều features có thể gây overfit.
- **Dùng model nhỏ hơn** Nếu có thể, hãy giảm kích thước model bằng cách thay thế các phần dư thừa. Ví dụ, trước đây, các model cho ImageNet thường dùng Fully Connected Layers ở phần cuối, nhưng bây giờ, ta có thể thay bằng average pooling, giúp giảm đáng kể số tham số.
- **Giảm batch size:** Nếu dùng batch normalization, batch nhỏ có thể giúp regularize model. Batch lớn hơn có thể làm giảm độ ngẫu nhiên của quá trình huấn luyện.
- **Dùng dropout:** Nếu dùng ConvNets, có thể thử dropout2d (spatial dropout), nhưng cẩn thận vì dropout có thể ảnh hưởng không tốt đến batch normalization.
- **Tăng weight decay:** Hãy thử tăng weight decay để giảm overfitting.
- **Dùng early stopping:** Dừng huấn luyện dựa trên validation loss để tránh model overfit quá mức.
- **Thử model lớn hơn:** Cuối cùng, nếu các cách trên chưa đủ, hãy thử model lớn hơn. Đôi khi, model lớn hơn nhưng dừng sớm (early stopping) lại hoạt động tốt hơn so với model nhỏ hơn.

- Cuối cùng, để có thêm sự tự tin rằng mạng của bạn là một bộ phân loại hợp lý, bạn nên trực quan hóa the network’s first-layer weights và đảm bảo bạn có được các cạnh đẹp có ý nghĩa. Nếu bộ lọc lớp đầu tiên của bạn trông giống như nhiễu thì có thể có điều gì đó không ổn. Tương tự như vậy, activations bên trong mạng đôi khi có thể hiển thị các hiện vật kỳ lạ và gợi ý về các vấn đề.

## 5. Tune (Tối ưu)

- Bây giờ, bạn đã quen thuộc với tập dữ liệu và có thể thử nghiệm nhiều kiến trúc model khác nhau để đạt validation loss thấp. Một số mẹo ở giai đoạn này:
- **Dùng random search thay vì grid search:** Khi tinh chỉnh nhiều hyperparameter, nhiều người thích dùng grid search (thử nghiệm toàn bộ các giá trị), nhưng thực tế, random search lại hiệu quả hơn. Lý do là một số tham số quan trọng hơn những tham số khác, vì vậy việc thử ngẫu nhiên thường mang lại kết quả tốt hơn.
- **Dùng kỹ thuật tối ưu hyperparameter:** Có rất nhiều công cụ tối ưu hyperparameter dựa trên Bayesian optimization. Một số đồng nghiệp của tôi đã thành công với chúng, nhưng theo kinh nghiệm cá nhân, cách tốt nhất để tìm model phù hợp vẫn là… dùng một thực tập sinh để thử nghiệm tất cả (đùa thôi 😆).

## 6. Squeeze out the juice (Tận dụng tối đa model)

- Khi đã tìm được kiến trúc và hyperparameter tốt nhất, vẫn còn một số cách để tối đa hóa hiệu suất:
- **Dùng ensemble models:** Ensemble (kết hợp nhiều model) gần như luôn giúp tăng thêm 2% độ chính xác. Nếu không thể dùng ensemble khi chạy thực tế, hãy thử distillation (chuyển kiến thức từ model lớn sang model nhỏ hơn).
- **Tiếp tục huấn luyện lâu hơn:** Nhiều người dừng huấn luyện khi validation loss không giảm nữa, nhưng trong nhiều trường hợp, model vẫn có thể tiếp tục cải thiện sau một thời gian dài. Tôi từng để model train trong kỳ nghỉ đông, và khi quay lại vào tháng 1, nó đã trở thành state-of-the-art (SOTA)! 😆

## Conclusion (Tổng kết)


- Nếu đã đến giai đoạn này, bạn có tất cả yếu tố để thành công:
- ✅ Hiểu rõ công nghệ, tập dữ liệu và bài toán
- ✅ Xây dựng hệ thống huấn luyện/đánh giá bài bản
- ✅ Đạt độ chính xác cao và thử nghiệm các model ngày càng phức tạp
- ✅ Cải thiện hiệu suất qua từng bước

Bây giờ, bạn đã sẵn sàng để đọc nhiều bài báo hơn, thử nghiệm model lớn hơn, hoặc… đi nghỉ một chút sau khi model train xong! 🚀

