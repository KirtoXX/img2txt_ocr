# img2txt_ocr
This is a simple project to recognize the characters in the img.<p>
I train the little deep neural network on K40 GPU for 100epoch without img aug.<p>
Finall I got the accuracy of 95.5% <p>

# Data
url：http://meizu.baiducloud.top/ps/web/index.html <p>
![](https://github.com/KirtoXX/img2txt_ocr/blob/master/test_image/0.png)<p>

# Network Architecture
1.use CNNs to encode img to vectors<p>
2.use Bi-GRU to decode vector to softamx of characters <p>
![image](https://github.com/KirtoXX/img2txt_ocr/blob/master/59478a4da1b1b.jpg)
