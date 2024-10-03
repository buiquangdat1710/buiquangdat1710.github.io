---
title: "Lộ trình học AI"
date: 2024-09-29 00:00:00  + 0800
---
---
**Python:** 
-  **Thời gian cần thiết:** Nếu có học qua về ngôn ngữ lập trình khác rồi thì có thể học trong 1 tuần. Người mới thì có thể trong 1 tháng.
-  **Kiến thức cần nắm được:** Syntax cơ bản, vòng lặp, câu điều kiện, hàm, OOP. Phần OOP không cần thiết phải học kế thừa, đa hình,... chỉ cần biết cách khai báo class là đủ. Phần python này không cần thiết phải học các framework như numpy, pandas,... Các framework thì nên học cùng phần Machine Learning.

**Toán:**
- **Thời gian cần thiết:** Khoảng 1 tháng hoặc ít hơn nếu có nền tốt về toán. Học sâu hơn chắc 2 tháng.
- **Kiến thức cần nắm được:** Đại số tuyến tính thì cần nắm được các phép toán trên ma trận, ma trận nghịch đảo, định thức, chuẩn norm, giải tích ma trận, chéo hóa ma trận (nâng cao). Không nhất thiết phải học ánh xạ tuyến tính, không gian vector... Giải tích thì chỉ cần nắm rõ phần đạo hàm riêng và tích phân một lớp là đủ. Xác suất thống kê thì là phần khá khó, cần nắm được các phân phối xác suất, mật độ xác suất, xác suất Bayes,... Chung quy lại thì phần đại số vẫn là quan trọng nhât, cần học kĩ.

**Machine Learning:**
- **Thời gian cần thiết:** 4-5 tháng, phần này rất quan trọng, đừng học qua loa cho xong. Mình đã từng học để cho xong trong vòng 2 tháng và khi có người hỏi lại mình thì mình trả lời sai hết và phải học lại từ đầu.
- **Kiến thức cần nắm được:** Hồi quy tuyến tính, K-means, KNN, hồi quy Logistic, Gradient Descent, PCA, hồi quy Softmax, SVM (phần này rất khó, học kĩ toán mới học đc SVM), Recommend System, Principal Component Analysis, Anomaly Detection, Linear Discriminant Analysis, Decision Tree, Random Forest,... Đặc biệt có mảng Reference Learning là phần mình thích nhất trong AI, học cái này sẽ hiểu được cách các robot phản ứng với môi truờng.

**Deep Learning:**
- **Thời gian cần thiết:** 3-4 tháng để nắm được cơ bản. Nâng cao chắc mất 1-2 năm. Phần này thì toán nhẹ hơn Machine Learning nhưng phải nắm chắc thì mới học cao lên được

- **Kiến thức cần nắm được:** MLP, CNN, RNN, GRU, LSTM, Transformers, GAN,...

Học xong phần Deep Learning rồi thì có thể chọn một trong hai hướng là NLP (xử lý ngôn ngữ tự nhiên) và CV (xử lý ảnh). Nhiều người còn chia ra cái thứ ba là xử lý âm thanh nhưng mình gộp phần ý vào NLP. Vì mình theo NLP nên mình sẽ viết về NLP, hy vọng trong tương lai có thể viết thêm về phần CV.

**NLP:**
- **Thời gian cần thiết:** Học đến lúc đi làm, học đến suốt đời cũng không hết kiến thức vì mỗi ngày lại có paper mới. Nhưng để học xong kiến thức cơ bản thì mất tầm 3 tháng.
  
- **Kiến thức cần nắm được:** N-gram, SkipGram, Glove, Word2vec, seq2seq, BytePairEncoding, Tách token, Embedding token, FastText, RNN, GRU, LSTM, Transfomers, Transfomers cải tiến, BERT, RoBERTa, GPT,...

**LSTM:**
- **Thời gian cần thiết:** 3 tháng để học xong cơ bản.
- **Kiến thức cần nắm được:** Prompt Engineering, RAG, GraphRAG, RIG, pipecone, chorma, vectordatabase, học cách call API, deploy sản phẩm. Phần LLMs này sẽ rất khó để train từ đầu nên phần này sẽ tập trung chủ yếu là deploy.

Có 2 framework chính cần phải học khi học DeepLearning, hoặc có thể master một trong hai:
- **Tensorflow**: Chắc mất 2 tháng để học cơ bản.
- **Pytorch**: Cũng mất 2 tháng để học cơ bản.

Có 2 framework chính cần phải học khi học LLMS:
- **LangChain**: Chắc mất 2 tháng để học cơ bản.
- **Lambda Index:** Chắc mất 2 tháng để học cơ bản.

## Các nguồn và khóa học AI.

Dưới đây là danh sách các khóa học và các trang web cũng như sách về AI theo quan điểm cá nhân của mình, lưu ý là sẽ có những web chứa cả phần Machine Learning và Deep Learning nhưng mình sẽ chỉ liệt kê ở phần Machine Learning:

**Machine Learning:**
- Web [Machine Learning cơ bản](https://machinelearningcoban.com/math/) của Vũ Hữu Tiệp: Chắc đây là trang web huyền thoại mà ai mới học AI cũng xem. Thực sự đây là trang web rất tốt để các bạn nhập môn với AI, nên đọc thêm sách Machine Learning cơ bản nữa, thường thì sách sẽ đầy đủ hơn là web nhưng web có animation ở một số bài xem dễ hiểu hơn. Nói chung là nên đọc sách và web song song.
- Web [Deep AI KhanhBlog](https://phamdinhkhanh.github.io/deepai-book/intro.html) của Phạm Đình Khánh: Một trang web rất hay nữa, trang web này còn dạy cả python và những framework như pandas, numpy... Các phần Machine Learning cũng đào sâu hơn trang của bác Tiệp, nhưng lượng kiến thức thì không rộng bằng. 
- Web [Khoa học dữ liệu](https://phamdinhkhanh.github.io/content) của Phạm Đình Khánh: Lại một trang web rất hay nữa của bác Khánh. Trang này thì chứa cả phần Machine Learning, NLP, CV, GAN,... Mình chưa đọc nhiều trang này nên chưa đánh giá được.
- Web [Blog của Đỗ Minh Hải](https://dominhhai.github.io/vi/): Bác này hình như cái gì cũng viết, từ AI đến Backend, bác này còn review sách nữa, khá thú vị. Mình cũng chưa đọc nhiều trang này nên cũng chưa đánh giá kiến thức được.
- Web [Lil'Log](https://lilianweng.github.io/page/3/): Chị này thì là người có tên tuổi trong giới AI, đang làm ở OpenAI. Blog thì viết rất nặng về toán và tập trung vào các paper nhiều. Nếu ai có khả năng đọc hiểu được web này thì rất tốt. Mình cũng chưa đọc nhiều trang này nên cũng chưa đánh giá kiến thức được nhưng mình cũng chỉ mong một ngày có đủ trình độ để viết ra những blog như này.
- Sách [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf): Quyển gối đầu giường cho những ai học Machine Learning, thực sự đây là quyển sách huyền thoại mà ai cũng biết đến. Ngay cả bác Hải cũng phải dành một bài viết để phân loại các chương từ dễ đến khó của quyển sách này: [Review](https://dominhhai.github.io/vi/2017/12/ml-prml/). Mình chỉ đọc được tầm 30-50 trang đầu của quyển này, hy vọng trong tương lai có thời gian để đọc tiếp.
- Sách [An Introduction to Statistical Learning](https://www.stat.berkeley.edu/~rabbee/s154/ISLR_First_Printing.pdf): Tuy mình cũng chỉ đọc đâu đó dưới 100 trang nhưng mình đánh giá đây là sách hay, từ ngữ dễ hiểu, có những kiến thức mới lạ. Quyển này thì được ít người biết đến hơn quyển ở trên nhưng là một quyển sách tuyệt vời cho những bạn muốn đọc quyển nào dễ hiểu.
- Sách [Machine Learning an algorithmic perspective](https://github.com/hongzhonglu/machine-learning-books/blob/master/Machine%20Learning%20-%20An%20Algorithmic%20Perspective%202nd%20edition%202014.pdf): Quyển này thì mình chưa đọc trang nào nhưng lướt qua trông cũng có vẻ hay và hơi nặng về toán. Hy vọng trong tương lai có thời gian để đọc.
- Sách [Artificial Intelligence: A Modern Approach](https://aima.cs.berkeley.edu/): Quyển này thì mình cũng chưa đọc trang nào nhưng mà mình thấy quyển này là một trong những quyển hiếm hoi mà viết về các chủ đề AI rất rộng. Các bạn có thể xem qua mục luc của quyển này, có cả phần lý thuyết trò chơi, đây là phần mình rất thích.
- Sách [Data Science from Scratch](https://www.oreilly.com/library/view/data-science-from/9781492041122/): Đây là sách của hãng xuất bản nổi tiếng O'reilly (cái hãng xuất bản mà sách hay có mấy con động vật trên trang bìa ý). Sách này thì thiên về Data Science nhiều hơn và các phần AI nhẹ nhàng hơn cho người mới bắt đầu. Nói chung là sách của hãng O'reilly thì không có gì để chê, nội dung và code dễ hiểu, cách trình bày đẹp đẽ, dễ tiếp cận phần đông người đọc.
- Sách [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/): Mình đã đọc qua sách này rồi thì đây là quyển sách dễ đọc, sách này sẽ không nặng về phần toán nhưng rất chi tiết về phần code, phải nói là cực chi tiết và dễ hiểu. Mới bắt đầu học AI mà không muốn học toán nhiều thì nên đọc cuốn này, không có điểm nào để chê.
- Khóa học [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?utm_source=gg&utm_medium=sem&utm_campaign=b2c_apac_career-academy_coursera_ftcof_professional-certificates_arte_july-24_dr_geo-set-2-multi_sem_rsa_gads_lg-all&utm_content=b2c&campaignid=21517156754&adgroupid=166384451178&device=c&keyword=certification%20courses&matchtype=p&network=g&devicemodel=&adpostion=&creativeid=707443146865&hide_mobile_promo=&gad_source=1&gclid=CjwKCAjwgfm3BhBeEiwAFfxrG8dwwcEoiRfNnnzgnaCZDTMNVQsP3ydOLfuP46l7pPgaY6Y_hKl0FhoCpmMQAvD_BwE): Đây là khóa học do Andrew Ng, một người mà ai học AI cũng biết. Đây cũng chính là khóa học mình học song song với web của bác Tiệp. Khóa làm chi tiết, mỗi tuần học có cả quizz để ôn lại kiến thức nhưng mình thấy phần code hơi ít, kiểu nó thiên về lý thuyết nhiều hơn code. Đây là nguồn học hiếm hoi mà có chủ đề Anomaly Detection (trong suốt khoảng thời gian mình học AI thì ngoài khóa học này ra chưa có nguồn nào nói về cái này). Sau khi hoàn thành khóa học thì bạn sẽ được cấp chứng chỉ do Coursera trao. Các khóa học trên Coursera bạn hoàn toàn có thể xin hỗ trợ học phí bằng cách viết thứ và 99% các khóa sẽ được duyệt trong 15 ngày.
- Khóa [FOUNDATIONS OF MACHINE LEARNING](https://vietai.org/our-program/) của VietAI: Thực sự thì cá nhân mình không đánh giá quá cao khóa này, các kiến thức trong khóa học này bạn hoàn toàn có thể đọc được trong sách và có khi còn chi tiết hơn. Nhưng mà chứng chỉ trông có vẻ xịn :vv
- Các khóa học trên Udemy: Mình chưa học nên cũng chưa kiểm chứng có chất lương hay không nhưng mình thấy trên đó khá nhiều khóa về Machine Learning. 
  
> Danh sách trên là những nguồn mình biết về Machine Learning, có thể có những nguồn khác nữa nhưng giờ mình chỉ nhớ được từng này. Nếu bạn là nguời mới bắt đầu học AI và đang không biết nên học nguồn nào trước thì mình khuyên bạn nên đọc sách và học web Machine Learning cơ bản của bác Tiệp, có thể học song song khóa Machine Learning Specialization. Thực sự bạn chỉ cần hiểu được hết tất cả nội dung trong sách bác Tiệp thôi thì bạn đã có một nền tảng Machine Learning cực kì tốt rồi.

> Để tìm thêm các sách về Machine Learning thì bạn có thể search từ khóa: Machine Learning Books

**Deep Learning:**
- Web [Đắm mình vào học sâu](https://d2l.aivivn.com/): Một web rất chi tiết và đầy đủ về học sâu. Trang web này là trang web đầu tiên mình tìm hiểu về học sâu, code dễ hiểu và có đủ lượng kiến thức toán, không có gì để chê.
- Web [Deep Learning cơ bản](https://nttuan8.com/sach-deep-learning-co-ban/): Mình chưa xem trang web này nhưng lướt qua cũng có vẻ hay. Tương lai có thời gian nhất định mình sẽ đọc trang web này.
- Khóa học [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning): Các khóa học của bác Andrew Ng luôn chất lượng và không có gì để chê. 
- Sách [Deep Learning](https://www.deeplearningbook.org/): Mình đã đọc được một nửa quyển sách này thì mình thấy đây là quyển sách khá nặng về lý thuyêt và toán, gần như tất cả chỉ là lý thuyết, không có một đoạn code nào. Nhưng đây cũng là một quyển mà mình muốn hiểu hết tất các lượng kiến thức bên trong vì kiến thức của quyển này rất chất lượng, có những kiến thức rất khó mà chỉ trong quyển này mới có. Quyển này được ví như quyển sách gối đầu giường cho những ai học Deep Learning.
- Sách [Understanding Deep Learning](https://udlbook.github.io/udlbook/): Cuốn sách này mình thấy nổi và mình cũng chưa có thời gian để đọc nên mình chưa đánh giá được.
- Khóa [FOUNDATIONS OF DEEP LEARNING](https://vietai.org/our-program/) của VietAI: Khóa này thì mình thấy ổn hơn khóa Machine Learning của VietAI. Nhưng lượng kiến thức thì bạn vẫn có thể tìm được ở ngoài.

**NLP:**
- Khóa [NLP](https://protonx.coursemind.io/intro?lang=vi) của ProtonX: Đây là khóa mình đang theo học và mình đánh giá cao chất lương kiến thức cũng như phần code. Các bạn sẽ thấy hầu hết blog về NLP mình sẽ để tài liệu tham khỏa là ProtonX.
- sách [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/): Mình đã đọc được gần 1 nửa cuốn này và mình thấy đây là một quyển sách hay, các kiến thức phức tạp như transfomers được lập trình từ đầu, không có để chê.
- Sách[Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf): Đây là cuốn sách được coi là gối đầu giường bên NLP, mình cũng chỉ mới đọc qua vài trang nên chưa đánh giá được nhiều.
- Khóa [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing): Chắc không cần nói nhiều nữa, vẫn luôn chất lượng như mọi khi.

## Lời Kết.
Học AI là một con đường dài và khó khăn, AI được đánh giá là mảng khó nhất nên cũng đừng quá bất ngờ về lượng kiến thức nhiều. Nhưng nếu như bạn học được AI thì bạn học mảng gì cũng được tại phần khó nhất bạn còn học được thì ngại gì những cái khác ? Đừng vội nản chí khi thấy toán nhiều, nếu cảm thấy mệt thì nghỉ 1 hôm rồi học tiếp chứ đừng nghỉ hẳn tại học AI là một quá trình rất thú vị. Hiểu được cách AI hoạt động là bạn đã hiểu được một phần thế giới này hoạt động.

> Người không học AI thì nghĩ rằng AI là một con robot có thể biết suy nghĩ như con người.

> Người học AI thì biết rằng AI chỉ là toán học. Rất nhiều toán học...