# Göz kırpmasını algılayan yapay zeka projesi
![AI](https://gblobscdn.gitbook.com/assets%2F-M8qMpTMWhj-s-7KKqGm%2F-M8ureXSf69DComOoO-G%2F-M8url3zHkjVoxWYninQ%2Faga.gif?alt=media&token=374ae17e-5866-474a-8a24-4be2942c50c2)

Projenin yapımında **vegav1** ve **Opencv** kütüphaneleri kullanılmıştır.

Vega, yapay zeka oluşturmak ve işlemlerini çabucak çalıştırmak için geliştirilmiş sihirli bir yapay zeka kütüphanesidir.

Kütüphanenin resmi sitesi => [Vega](http://vega.aksoylu.space)

Bu projenin kütüphanedeki dökümanı => [Dökümantasyon](https://vega.aksoylu.space/turkish/tuerkce/oernek-proje-vega-ile-goez-yorgunlugunu-algilayan-yapay-zeka-projesi)


## Kütüphanelerin Kurulumu

Vega'nın Kurulumu

> pip install vegav1


OpenCV'nin kurulumu

> pip install opencv-python

### Çalıştırma
Projeyi çalıştırmak için konsola yazın;
> python detection.py


### Eğitim
Ağı tekrar eğitmek için konsola yazın;
> python trainer.py


### Detaylar
detection.py'deki predict fonksiyonunda bulunan ambientLight değişkenindeki 0.709 değeri ortama göre değiştirilebilir. Proje düşük ve yüksek ışığa göre gerekli hesaplamayı otomatik olarak yapmaktadır.

closed_eye klasöründe kapalı göz resimleri, open_eye klasöründe ise açık göz resimleri bulunur. trainer.py'deki epoch sayısı ve klasörlerdeki görsel sayısı arttırılabilir. Eğitim için görseller 24x24 piksel ve grayscale olmalıdır.

Detaylı tutorial için => [Dökümantasyon](https://vega.aksoylu.space/turkish/tuerkce/oernek-proje-vega-ile-goez-yorgunlugunu-algilayan-yapay-zeka-projesi)



