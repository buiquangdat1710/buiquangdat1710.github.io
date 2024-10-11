---
title: "Prompt Engineering"
date: 2024-10-09 00:00:00  + 0800
categories: [Prompt Engineering]
tags: [prompt engineering]
---
---

## 1. Prompt là gì ?

- **Prompt** là một thuật ngữ dùng trong nhiều lĩnh vực, nhưng phổ biến nhất trong trí tuệ nhân tạo và học máy (AI/ML), đặc biệt là trong các mô hình ngôn ngữ lớn như GPT. Trong ngữ cảnh này, prompt chỉ đoạn văn bản hoặc câu hỏi mà người dùng cung cấp cho mô hình AI để nó trả lời hoặc tạo ra nội dung dựa trên thông tin đó.
- Ví dụ, khi bạn hỏi một câu hỏi cho ChatGPT như:
> Hôm nay thời tiết như nào ?
- Câu hỏi trên chính là một prompt. Tóm lại, prompt là một thông tin ban đầu cung cấp để hệ thống thực hiện một nhiệm vụ cụ thể hoặc tạo ra phản hồi.
- Tuy nhiên, việc đặt câu hỏi nhiều người nghĩ là một bước không quan trọng, họ chỉ cần nghĩ rằng mình chỉ cần đặt câu hỏi đúng trọng tâm là được. Với nhiều bài toán đơn giản thì mình chỉ cần hỏi một câu đúng trọng tâm nhưng với những bài toán phức tạp thì nhập phần prompt là cả một công đoạn, hay nhiều người gọi với cái tên rất kiêu là **Nghệ thuật viết Prompt** hay **Kĩ thuật viết Prompt**.

## 2. Các kĩ thuật viết prompt.

### Zero-shot Prompting.

- Các mô hình ngôn ngữ lớn (LLMs) ngày nay, chẳng hạn như GPT-3.5 Turbo, GPT-4, và Claude 3, được tinh chỉnh để làm theo hướng dẫn và được huấn luyện trên lượng dữ liệu lớn. Việc huấn luyện trên quy mô lớn giúp những mô hình này có khả năng thực hiện một số nhiệm vụ theo cách "zero-shot". **Zero-shot** prompting có nghĩa là prompt được sử dụng để hỏi mô hình không chứa ví dụ hoặc minh họa. Zero-shot prompt trực tiếp hướng dẫn mô hình thực hiện một nhiệm vụ mà không cần thêm ví dụ nào để định hướng nó.
- Như câu hỏi:
> Hôm nay thời tiết như nào ?
- Chính là Zero-shot prompting, tức mình sẽ đặt câu hỏi trực tiếp mà không thêm các ví dụ hoặc ngữ cảnh phía trước.

- Ví dụ thêm về  Zero-shot prompting:
- Prompt:

> Classify the text into neutral, negative or positive. 

> Text: I think the vacation is okay.

> Sentiment:

- Output:

> Neutral

- Lưu ý rằng trong prompt ở trên, chúng ta không cung cấp cho mô hình bất kỳ ví dụ nào về văn bản kèm theo phân loại của chúng, mô hình ngôn ngữ lớn (LLM) đã tự hiểu cảm xúc của văn bản xong phân loại - đó chính là khả năng zero-shot hoạt động.

### Instruction Tuning.

- **Instruction Tuning** là một phương pháp tinh chỉnh mô hình ngôn ngữ lớn (LLM) nhằm cải thiện khả năng hiểu và làm theo hướng dẫn của mô hình. Hiểu một cách đơn giản là mình sẽ đặt câu prompt và yêu cầu mô hình phải làm gì trong câu prompt đó. Dưới đây là ví dụ đặt prompt sử dụng Instruction Tuning:
- Prompt:

>  Hãy viết một đoạn văn tóm tắt cuốn sách 'Harry Potter và Hòn đá Phù thủy

- Output:

> ‘Harry Potter và Hòn đá Phù thủy’ kể về cậu bé Harry Potter, một đứa trẻ mồ côi sống với gia đình dì dượng. Vào ngày sinh nhật thứ 11, cậu phát hiện mình là một phù thủy và được mời học tại Trường Phù thủy và Pháp sư Hogwarts. Tại đây, Harry kết bạn với Ron Weasley và Hermione Granger, đồng thời khám phá ra bí mật về sự tồn tại của Hòn đá Phù thủy, một vật phẩm ma thuật có thể mang lại sự bất tử. Câu chuyện tập trung vào cuộc phiêu lưu của Harry trong việc ngăn chặn Chúa tể Voldemort, một phù thủy hắc ám đã sát hại cha mẹ cậu.

- Dưới đây là ví dụ đặt prompt mà không sử dụng Instruction Tuning:

> Harry Potter và Hòn đá Phù thủy.

- Output:

> Harry Potter và Hòn đá Phù thủy là cuốn sách đầu tiên trong loạt truyện về Harry Potter của J.K. Rowling. Cuốn sách ra mắt lần đầu vào năm 1997 và nhận được nhiều sự yêu thích từ độc giả trên toàn thế giới.


- Prompt sử dụng instruction tuning: Mô hình hiểu rằng người dùng yêu cầu tóm tắt, và phản hồi là một đoạn văn ngắn gọn, đúng trọng tâm về nội dung của cuốn sách theo hướng dẫn rõ ràng. Điều này nhờ vào việc mô hình đã được tinh chỉnh để nhận diện và đáp ứng các yêu cầu mang tính chỉ dẫn (instruction-based).

- Prompt không sử dụng instruction tuning: Prompt không có hướng dẫn cụ thể (chỉ là tên cuốn sách), dẫn đến phản hồi không đúng với yêu cầu tóm tắt. Thay vì tóm tắt nội dung sách, mô hình chỉ cung cấp thông tin tổng quan về cuốn sách và ngày xuất bản.
- Như các bạn thấy thì hầu như tất cả câu prompt do người dùng đặt đều sử dụng kĩ thuật instruction tuning để yêu cầu nhiệm vụ cho mô hình. 
- Việc sử dụng instruction tuning đã được chứng minh là cỉa thiện khả năng học so với kĩ thuật zeros-shot ([Wei et al., 2022](https://arxiv.org/pdf/2109.01652)).

### Reinforcement Learning from Human Feedback (RLHF).

- Reinforcement Learning from Human Feedback (RLHF), hay Học tăng cường từ phản hồi của con người, là một kỹ thuật trong học máy (machine learning) sử dụng phản hồi từ con người để giúp tinh chỉnh các mô hình AI nhằm cải thiện hiệu suất và phù hợp hơn với kỳ vọng của người dùng. RLHF đặc biệt quan trọng trong việc phát triển các mô hình ngôn ngữ lớn (LLMs) như GPT, nơi yêu cầu khả năng phản hồi theo cách tự nhiên, có đạo đức và hữu ích.
- Dưới đây là ví dụ về model sử dụng RLHF:

- Prompt: 

> Viết một câu trả lời ngắn gọn cho câu hỏi: 'Trí tuệ nhân tạo sẽ thay thế con người trong công việc không ?

- Output:

> Trí tuệ nhân tạo sẽ thay đổi nhiều ngành nghề, nhưng không hoàn toàn thay thế con người. AI có thể tự động hóa các nhiệm vụ lặp đi lặp lại, trong khi con người vẫn sẽ đóng vai trò quan trọng trong việc ra quyết định, sáng tạo và giao tiếp.

- Dưới đây là ví dụ về model không sử dụng RLHF:
- Prompt:

> Viết một câu trả lời ngắn gọn cho câu hỏi: 'Trí tuệ nhân tạo sẽ thay thế con người trong công việc không ?

- Output:

> AI sẽ thay thế con người trong hầu hết các công việc vì nó nhanh hơn, hiệu quả hơn và không mắc lỗi như con người.

- Prompt sử dụng RLHF: Câu trả lời đã được tinh chỉnh dựa trên phản hồi của con người, giúp mô hình hiểu cách trả lời sao cho có sự cân nhắc về ngữ cảnh xã hội, đạo đức, và sự lo ngại của người dùng. Kết quả là một câu trả lời hài hòa và dễ chấp nhận hơn.

- Prompt không sử dụng RLHF: Câu trả lời có thể bị đơn giản hóa quá mức hoặc thiên về một hướng nhất định, thiếu sự cân nhắc đối với các khía cạnh nhạy cảm hoặc mối quan tâm của con người. Điều này có thể dẫn đến phản hồi kém hiệu quả và không phù hợp trong các ngữ cảnh phức tạp.
- RLHF đã được áp dụng để mở rộng quá trình instruction tuning, trong đó mô hình được điều chỉnh để phù hợp hơn với sở thích của con người. Sự phát triển gần đây này là nền tảng cho các mô hình như ChatGPT. Chúng ta sẽ thảo luận về tất cả các phương pháp và cách tiếp cận này trong các phần tiếp theo.
- Khi zero-shot không hoạt động, người ta khuyến nghị cung cấp các ví dụ hoặc minh họa trong prompt, dẫn đến few-shot prompting. Trong phần tiếp theo, chúng ta sẽ trình bày về few-shot prompting.

### Few-shot Prompting.

- Mặc dù các mô hình ngôn ngữ lớn thể hiện khả năng zero-shot đáng kể, nhưng chúng vẫn gặp khó khăn với những nhiệm vụ phức tạp hơn khi sử dụng cài đặt zero-shot. Kỹ thuật few-shot prompting có thể được sử dụng như một phương pháp để kích thích việc học trong ngữ cảnh, nơi chúng ta cung cấp các ví dụ trong prompt để hướng dẫn mô hình đạt hiệu suất tốt hơn. Các ví dụ này đóng vai trò như điều kiện cho các ví dụ tiếp theo mà chúng ta muốn mô hình tạo ra phản hồi.

- Theo [Touvron et al. 2023](https://arxiv.org/pdf/2302.13971), các thuộc tính few-shot lần đầu tiên xuất hiện khi các mô hình được mở rộng đến kích thước đủ lớn [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361).
- Hãy cùng minh họa few-shot prompting thông qua một ví dụ được trình bày trong [Brown et al. 2020](). Trong ví dụ này, một từ mới sẽ được thêm vào trong câu và chúng ta phải lấy ví dụ về từ đó:
- Prompt:

> A "whatpu" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is:
 
> We were traveling in Africa and we saw these very cute whatpus.
 
> To do a "farduddle" means to jump up and down really fast. An example of a sentence that uses the word farduddle is:

- Output:

> When we won the game, we all started to farduddle in celebration.

- Chúng ta có thể quan sát rằng mô hình đã học được cách thực hiện nhiệm vụ chỉ với một ví dụ (tức là 1-shot). Đối với các nhiệm vụ khó hơn, chúng ta có thể thử nghiệm bằng cách tăng số lượng ví dụ (ví dụ: 3-shot, 5-shot, 10-shot, ...).
- Theo các phát hiện từ [Min et al. (2022)](https://arxiv.org/abs/2202.12837), dưới đây là một vài mẹo về việc sử dụng các ví dụ trong khi thực hiện few-shot:
  - "Không gian nhãn và phân bố của văn bản đầu vào được chỉ định bởi các ví dụ đều quan trọng (bất kể các nhãn có chính xác cho từng đầu vào hay không)."
  - Định dạng mà bạn sử dụng cũng đóng vai trò quan trọng trong hiệu suất, ngay cả khi bạn chỉ sử dụng các nhãn ngẫu nhiên, điều này cũng tốt hơn là không có nhãn nào cả.
  - Các kết quả bổ sung cho thấy việc chọn các nhãn ngẫu nhiên từ một phân bố nhãn thực (thay vì một phân bố đồng nhất) cũng có lợi.

- Giờ hãy lấy một vài ví dụ về prompt sử dụng few-shot. Câu prompt đầu tiên được gắn nhãn ngẫu nhiên:

> This is awesome! // Negative

> This is bad! // Positive

> Wow that movie was rad! // Positive

> What a horrible show! //

- Output:

> Negative

- Chúng ta vẫn nhận được câu trả lời đúng, mặc dù các nhãn đã được ngẫu nhiên hóa. Lưu ý rằng chúng ta cũng giữ nguyên định dạng, điều này cũng giúp ích. Trên thực tế, với những thử nghiệm sâu hơn, có vẻ như các mô hình GPT mới mà chúng ta đang thử nghiệm ngày càng trở nên mạnh mẽ hơn ngay cả với các định dạng ngẫu nhiên. Ví dụ:
- Prompt:

> Positive This is awesome! 

> This is bad! Negative

> Wow that movie was rad!

> Positive

> What a horrible show!

- Output:

> Negative

- Định dạng ở trên không có sự nhất quán, nhưng mô hình vẫn dự đoán nhãn chính xác. Chúng ta cần tiến hành phân tích kỹ lưỡng hơn để xác nhận liệu điều này có đúng với các nhiệm vụ phức tạp hơn và các biến thể khác nhau của prompt hay không.

### Limitations of Few-shot Prompting. 

- Kỹ thuật few-shot truyền thống hoạt động tốt với nhiều nhiệm vụ, nhưng vẫn không phải là một phương pháp hoàn hảo, đặc biệt khi xử lý các nhiệm vụ đòi hỏi suy luận phức tạp hơn. Hãy cùng minh họa tại sao lại như vậy:

- Prompt:

> The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. 

> A:

- Output:

>  Yes, the odd numbers in this group add up to 107, which is an even number.

- Đây là prompt sử dụng zero-shot và như bạn thấy, nó đã trả lời sai. Giờ hãy thử sử dụng kĩ thuật few-shot:

- Prompt:

>The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.

> A: The answer is False.

>The odd numbers in this group add up to an even number: 17,  10, 19, 4, 8, 12, 24.

>A: The answer is True.

>The odd numbers in this group add up to an even number: 16,  11, 14, 4, 8, 13, 24.

> A: The answer is True.

>The odd numbers in this group add up to an even number: 17,  9, 10, 12, 13, 4, 2.

>A: The answer is False.

>The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. 

>A:

- Output:

> The answer is True.

- Dường như việc dùng few-shot prompting (cung cấp ít ví dụ) là không đủ để có được các phản hồi chính xác cho dạng bài toán suy luận này. Ví dụ trên cung cấp thông tin cơ bản về bài toán. Nếu bạn nhìn kỹ hơn, loại bài toán mà chúng ta đang giải đòi hỏi nhiều bước suy luận hơn. Nói cách khác, có thể sẽ giúp ích nếu chúng ta chia nhỏ vấn đề thành các bước và minh họa điều đó cho mô hình. Gần đây, phương pháp Chain-of-thought (CoT) prompting đã trở nên phổ biến để giải quyết các bài toán phức tạp hơn về số học, lẽ thường và suy luận ký hiệu.

- Nhìn chung, việc cung cấp ví dụ rất hữu ích để giải một số bài toán. Khi Zero-shot prompting (không cung cấp ví dụ) và few-shot prompting không đủ, điều đó có thể có nghĩa là những gì mô hình đã học chưa đủ để thực hiện tốt bài toán. Từ đây, người ta khuyến nghị bắt đầu nghĩ về việc tinh chỉnh (fine-tuning) các mô hình của mình hoặc thử nghiệm với các kỹ thuật prompting nâng cao hơn. Tiếp theo, chúng ta sẽ nói về một trong những kỹ thuật prompting phổ biến được gọi là Chain-of-Thought prompting, đã thu hút rất nhiều sự chú ý. 

> Coming Soon