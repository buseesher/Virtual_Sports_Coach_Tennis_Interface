import sys
import time
import cv2
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QTableWidget, QPushButton, QSizePolicy, QTableWidgetItem, QHeaderView, QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
import pyzed.sl as sl
import  ogl_viewer.viewer as gl 
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import math  
from scipy.spatial.transform import Rotation
import os

def quaternion_to_euler(x, y, z, w):
    # return pitch(y), roll (x), yaw (z) sırasına göre çevrildi. Yaygın kullanım
    rot = Rotation.from_quat([[x,y,z,w]])
    rot_euler = rot.as_euler('zyx', degrees=True)
    
    # print(rot_euler)
    return  rot_euler[0][2], rot_euler[0][1], rot_euler[0][0]

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.central_widget = QLabel(self)
        self.setCentralWidget(self.central_widget)
        
        # Pencere başlığı
        self.setWindowTitle("TÜBİTAK 1001 - Teniste Yapay Zeka Destekli Sanal Spor Koçu")

        # Create a horizontal layout
        #self.horizontal_layout = QHBoxLayout(self.central_widget)

        # Create a QLabel for displaying the image
        #self.image_label = QLabel(self.central_widget)
        #self.horizontal_layout.addWidget(self.image_label)

        # Başta ayarladığımız QLabel (değerlerle sabitlendi. İhtiyaca göre değişiklik yapılabilir. Görünürde değişmiyor?)
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(720,540)  
        self.image_label.move(1280, 0)  

        self.image_label_2 = QLabel(self)
        self.image_label_2.setFixedSize(720, 540)  
        self.image_label_2.move(640, -103)  # İkinci SVO görüntüsünün konumu (niye 0 değil de eksideyken konumu eşitleniyor araştır)

        # Create a table to display pitch, roll, and yaw values
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels([ 'Index','Pitch', 'Yaw', 'Roll'])
        self.table_widget.setFixedSize(500, 415) 
        self.table_widget.move(1280, 0) #tablo kaydırma
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch) # Tablodaki çizgileri tutup sürükleme kalktı 
        self.table_widget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # for the first column ('Index'), set a custom width
        self.table_widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed) # Set the resize mode of the first column to Fixed
        self.table_widget.setColumnWidth(0, 50)  # Set the width of the 'Index' column to 50 pixels
    
        # Vücut takibi için parametrelerin nesne oluşturulması
        self.body_param = sl.BodyTrackingParameters()
        self.body_param.enable_tracking = True
        self.body_param.enable_body_fitting = True
        self.body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        self.body_param.body_format = sl.BODY_FORMAT.BODY_34
        
        # İlk ZED kamera nesnesi ve parametreleri
        self.zed = sl.Camera() #self.zed nesnesi kamerayla etkileşim kurar
        self.init_params = sl.InitParameters() # kamera başlangıç parametrelerini tutacak InitParameters nesnesi
        self.init_params.camera_resolution = sl.RESOLUTION.HD720 # Kameranın çözünürlüğünü HD720 (1280x720 piksel)
        self.init_params.coordinate_units = sl.UNIT.METER # Kameranın ölçümünün metre cinsinden ifade edilmesi 
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA #Kameranın derinlik algılama modu
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP #Kameranın koordinat sisteminin sağ elle tutulan ve Y ekseni yukarıyı işaret eden bir sistem olduğunu belirtir 
        # 3D uzaydaki nesnelerin konumlarını ve yönelimlerini tanımlamak için kullanılan bir koordinat sistemi
        filepath = r"C:\Users\Buse\Downloads\SVO_SN31844341 (4).svo"
        print("Reading SVO file: {0}".format(filepath))
        self.init_params.svo_real_time_mode = True
        self.init_params.set_from_svo_file(filepath)
        self.zed.open(self.init_params) #Yapılandırılan başlangıç parametreleriYLE ZED kamera (ya da SVO dosyası) açILIR
        self.zed.enable_positional_tracking(sl.PositionalTrackingParameters())
        self.zed.enable_body_tracking(self.body_param) # İlk kamera için vücut takibi etkinleştirme

        # İkinci ZED kamera nesnesi ve parametreleri
        self.zed_2 = sl.Camera()
        self.init_params_2 = sl.InitParameters()
        self.init_params_2.camera_resolution = sl.RESOLUTION.HD720
        self.init_params_2.coordinate_units = sl.UNIT.METER
        self.init_params_2.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params_2.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        filepath_2 = r"C:\Users\Buse\Desktop\SVO_SN31844341 (4).svo"
        self.init_params_2.set_from_svo_file(filepath_2)
        self.init_params_2.svo_real_time_mode = True
        self.zed_2.open(self.init_params_2)
        self.zed_2.enable_positional_tracking(sl.PositionalTrackingParameters())
        self.zed_2.enable_body_tracking(self.body_param) # İkinci kamera için vücut takibi etkinleştirme çalışmıyor?

        positional_tracking_parameters = sl.PositionalTrackingParameters() #pozisyonel takip için parametrelerin nesnesini oluşturuldu
        self.zed.enable_positional_tracking(positional_tracking_parameters) 
        #oluşturulan takip parametrelerini kullanarak ZED kamerada mekansal takibi etkinleştirildi
        self.body_param = sl.BodyTrackingParameters() #Vücut takibi için parametrelerin nesnesi oluşturuldu
        self.body_param.enable_tracking = True #Vücut takibini etkinleştirir
        self.body_param.enable_body_fitting = True #Vücut takibi sırasında vücut uyumunun (fitting) etkinleştirilmesi
        self.body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE # fast-> HUMAN_BODY_ACCURATE
        #Vücut takibi için kullanılacak modeli daha doğru sonuçlar için insan vücudu doğruluk modeline ayarlar
        self.body_param.body_format = sl.BODY_FORMAT.BODY_34 #Takip edilen vücut formatı 34 temel nokta içermekte
        self.zed.enable_body_tracking(self.body_param) #Yukarıdak takip parametrelerini kullanarak vücut takibi etkin
        self.body_runtime_param = sl.BodyTrackingRuntimeParameters() #Vücut takibinde kullanılacak çalışma zamanı parametrelerinin nesnesi
        self.body_runtime_param.detection_confidence_threshold = 40 #Vücut tespitinin kabul edilebilir olması için gereken güven eşiği
        self.viewer = gl.GLViewer() # OpenGL görüntüleyici nesnesi 
        #Görüntüleyiciyi, kamera kalibrasyon parametreleri, vücut takibi etkinliği ve vücut formatı ile başlatılması
        self.viewer.init(self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam, 
                         self.body_param.enable_tracking, 
                         self.body_param.body_format
                         )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_view)
        self.timer.start(30) #Zamanlayıcı, 30 milisaniye aralıklarla update camera view fonskiyonu çağrıldı
        # Sık güncellemeler yapılması gereken durumlarda kullanılır,gerçek zamanlı görüntü işleme veya animasyonlar gibi. 
        # amaç akıcı bir kullanıcı deneyimi sağlamaktır.

        # central widget alignment
        self.central_widget.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.central_widget.setFixedSize(1280, 720)  #videonun oynadığı kesit
        #self.central_widget.move(0, 0)  # Sol üst köşeye yaslamak içindi,etkisiz

        # Butonları oluşturma ve konumlandırma
        self.play_svo_button = QPushButton('Play SVO', self)
        self.play_svo_button.move(1810, 50)  # Butonun konumu ayarlanır
        self.play_svo_button.clicked.connect(self.play_svo)  # SVO oynatma fonksiyonuna bağlanır

        self.play_zed_button = QPushButton('Play ZED Camera', self)
        self.play_zed_button.move(1810, 100)  # Butonun konumu ayarlanır
        self.play_zed_button.clicked.connect(self.play_zed)  # ZED kamera oynatma fonksiyonuna bağlanır

    # SVO oynatma fonksiyonu for button
    def play_svo(self):
        # SVO dosyasından görüntü oynatma işlemleri buraya eklenir
        # Örneğin, self.init_params.svo_input_filename özelliğini SVO dosyanızın yoluna ayarlayarak
        # ve ZED kamerayı bu parametrelerle yeniden başlatarak 
        pass

    # ZED kamera oynatma fonksiyonu
    def play_zed(self):
        # SVO oynatmayı durdur
        self.zed.close()

    #Fonksiyon belirli aralıklarla çağrılmaKTA
    #ZED kameradan görüntü ve vücut takip verilerini alıp işlemek için kullanılır
    def update_camera_view(self):
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS: #kameradan yeni bir görüntü yakalamaya çalışır 
            #Bu metodun başarılı olup olmadığını kontrol eder (sl.ERROR_CODE.SUCCESS).
            image = sl.Mat() #Başarılı olursa, sl.Mat() kullanılarak yeni bir görüntü matrisi oluşturulur ve
            self.zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU) #self.zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU) 
            #ile yakalanan görüntüyü bu matrise kopyalar. Görüntü, kameranın sol gözünden alınır ve CPU belleğinde saklanır
            
            #görüntü işleme için: yoruma alınmış satırlar
            #image_left_ocv = image.get_data()
            # Gürültü azaltma
            #denoised_image = cv2.fastNlMeansDenoisingColored(image_left_ocv, None, 3, 3, 7, 15)
            #(image_left_ocv, None, 3, 3, 7, 15) -> ilk 3:Renk bileşenleri için filtreleme gücünü belirleyen parametre
            # ne kadar büyük olursa, gürültüdeki filtreleme o kadar güçlü olur, çok yüksek değerler görüntünün aşırı yumuşaması ve detay kaybı yaratır
            # ikinci 3: önceki parametre gibi ama sadece renk bileşenleri içn.biri gri tonlamalı (luminance) bileşenler için, diğeri renk (chrominance) bileşenleri için
            # burada her iki bileşen için de aynı değerler kullanılmıştır
            # 7 yazan (template window size): Arama penceresinin boyutunu belirler. Bu değer, algoritmanın bir pikselin değerini tahmin etmek için kullandığı komşu piksellerin boyutunu ifade eder.
            #Genellikle 7 veya 21 gibi tek sayılar kullanılır. Değer arttıkça, gürültü azaltma etkisi artar fakat işlem süresi de uzar.
            # 15 yazan yer (search window size): Arama penceresinin boyutunu belirler. Bu pencere içindeki tüm pikseller, merkezi piksel değerinin tahmini için kullanılır.
            # Bu değer ne kadar büyük olursa, daha geniş bir alandaki pikseller filtrelemeye katılır, bu da teorikte daha iyi gürültü 
            #azaltma sağlar ancak işlem süresini artırır ve bazen görüntüde bulanıklığa neden olabilir.
            

            # Görüntü netleştirme
           # kernel_sharpening = np.array([[-1, -1, -1], 
           #                               [-1, 9, -1],
           #                               [-1, -1, -1]])
           # sharpened_image = cv2.filter2D(denoised_image, -1, kernel_sharpening)

            bodies = sl.Bodies() # algılanan bedenlerin bilgisini tutan nesne
            self.zed.retrieve_bodies(bodies, self.body_runtime_param) 
            # method, anlık görüntüdeki vücutları alıp bodies nesnesine kaydeder
            body_array = bodies.body_list
            if len(body_array) > 0:
                body = body_array[0]
                if len(body_array) > 1:
                    body_2 = body_array[1]
                # Update GL view
                self.viewer.update_view(image, bodies) #görüntü ve vücut verileri güncellenir

                # Convert image to RGB format
                image_left_ocv = image.get_data() #yakalanan görüntü OpenCV formatına dönüştürülür
                #image_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB) #RGB formatına çevrilir
                image_rgb = cv2.cvtColor(image_left_ocv, cv2.COLOR_BGR2RGB) #RGB formatına çevrilir

                # Vücut analizinin görselleştirilmesi
                image_scale = [1.0, 1.0]  #orijinal boyutta işlem yapılacağını belirtir
                cv_viewer.render_2D( #RGB formatına çevrilen görüntü üzerine vücut takip bilgilerini çizer
                     #parametreler çizim işleminin nasıl gerçekleşeceğini belirlemekte
                    image_rgb, 
                    image_scale, 
                    bodies.body_list, 
                    self.body_param.enable_tracking, 
                    self.body_param.body_format
                )

                # Define the list of indices you want to consider
                target_indices = [1, 3, 5, 6, 12, 13, 18, 19, 22, 23] #hedef eklemler
                try:
                    k = 0
                    for index_o in target_indices: #local keypoint rotation represented by a quaternion
                                #her bir hedef eklem indeksi için tekrarlar
                                # quaternion yönelimi euler'e çevrildi
                                pitch,  yaw, roll = quaternion_to_euler(body.local_orientation_per_joint[index_o][0],
                                                                        body.local_orientation_per_joint[index_o][1],
                                                                        body.local_orientation_per_joint[index_o][2],
                                                                        body.local_orientation_per_joint[index_o][3])
                                #euler açıları ve eklem indeksleri tablo widgetında gösterilir 
                                self.update_table_widget(k, index_o, pitch, yaw, roll)  
                                k = k + 1 # k = sıra değişkeni
                except NameError:
                    print("body.keypoint[0] WASN'T defined!") # Name error hataları için hata yakalama
                    #body.local_orientation_per_joint veya quaternion_to_euler gibi 
                    #bir nesne veya fonksiyon tanımlanmamışsa kullanıcıya hata mesajı gösterilir
                    
                # svo için    
                # Convert to QImage and display in the PyQt interface
                height, width, channel = image_rgb.shape #görüntünün yüksekliği, genişliği ve kanal sayısı (RGB için 3) alınır
                #print (height, width)
                bytes_per_line = 3 * width #her bir satırın kaç byte veri içerdiğini hesaplar. 
                #RGB görüntüde her bir piksel için 3 byte (kırmızı, yeşil, mavi) kullanıldığından
                #her satır için byte sayısı width * 3 formülü ile hesaplanır
                qImg = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                #numpy dizisinden bir QImage nesnesi oluşturldu
                #QImage.Format_RGB888, her bir pikselin 3 kanallı RGB formatında olduğunu belirtir
                #Bu, görüntünün PyQt veya PySide arayüzünde doğru şekilde görüntülenmesini sağlar
                #pixmap = QPixmap.fromImage(qImg).scaled(720, 720) #görüntüyü ölçeklendirme
                #self.central_widget.setPixmap(pixmap)
                #qImg = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                #QImage nesnesini bir QPixmap nesnesine dönüştürür
                #QPixmap, PyQt/PySide arayüzlerinde görüntü gösterimi için kullanılır
                pixmap = QPixmap.fromImage(qImg)

                # Calculate the new size while maintaining the aspect ratio (svo boyut ve konumu değişiyor)
                aspect_ratio = width / height
                new_width = 600  # Set the desired width 
                new_height = int(new_width / aspect_ratio)

                # Display the image on the left side (svo'nun yerini etkiliyor)
                self.central_widget.setPixmap(pixmap.scaled(new_width, new_height))

                # Resize the image
                #resized_image = self.resize_image(image_rgb, 1250)

                # Display the resized image on the right side
                #qImg_resized = QImage(resized_image.data, resized_image.shape[1], resized_image.shape[0], 
                #                  3 * resized_image.shape[1], QImage.Format_RGB888)
                #pixmap_resized = QPixmap.fromImage(qImg_resized)
                #self.image_label.setPixmap(pixmap_resized)

                # Display the image on the right side
                image_path = "C:\\Users\\Buse\\Desktop\\angles.jpeg"
                image_pixmap = QPixmap(image_path) 
                #scaled_pixmap = image_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
                #self.image_label.setPixmap(scaled_pixmap)

                new_height = 480 # Yeni yükseklik değeri
                # Calculate the new size while maintaining the aspect ratio (en-boy oranı)
                aspect_ratio = image_pixmap.width() / image_pixmap.height()
                new_width = int(new_height * aspect_ratio)  # Yeni genişlik, en-boy oranını koruyacak şekilde hesaplanır
                #new_width = 430  # Set the desired width
                #new_height = int(new_width / aspect_ratio)
                
                scaled_pixmap = image_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio)
                self.image_label.setPixmap(scaled_pixmap)

                # Adjust the position of the angles image (resmin kaydırılması)
                self.image_label.move(1280, 400)

                # İkinci SVO dosyasından görüntü al
                if self.zed_2.grab() == sl.ERROR_CODE.SUCCESS:
                    image_2 = sl.Mat()
                    self.zed_2.retrieve_image(image_2, sl.VIEW.LEFT, sl.MEM.CPU)

                    # Alınan görüntüyü RGB formatına dönüştür
                    image_2_ocv = image_2.get_data()
                    image_2_rgb = cv2.cvtColor(image_2_ocv, cv2.COLOR_BGR2RGB)

                    # Görüntüyü ölçeklendirme
                    new_width = 600  # Set the desired width 
                    aspect_ratio = image_2_rgb.shape[1] / image_2_rgb.shape[0]
                    new_height = int(new_width / aspect_ratio)
                    image_2_resized = cv2.resize(image_2_rgb, (new_width, new_height))

                    # Ölçeklenmiş görüntüyü QLabel widget'ına yerleştirme
                    qImg_2 = QImage(image_2_resized.data, new_width, new_height, 
                                    3 * new_width, QImage.Format_RGB888)
                    pixmap_2 = QPixmap.fromImage(qImg_2)
                    self.image_label_2.setPixmap(pixmap_2)

                #time.sleep(0.10) # bekleme süresi
                os.system("cls")           
    
    #tablo güncelleme
    def update_table_widget(self, body_index, joint_index, pitch, yaw, roll):
        # If the joint index is not in the target list, skip updating the table
        target_indices = [1, 3, 5, 6, 12, 13, 18, 19, 22, 23]
        if joint_index not in target_indices:
            return
         # If the table does not have enough rows, add rows dynamically
        while self.table_widget.rowCount() <= body_index:
            self.table_widget.insertRow(self.table_widget.rowCount())
       
        item_index = QTableWidgetItem(f"{joint_index}")
        item_pitch = QTableWidgetItem(f"{pitch:.2f}")#2f kullanımı: sayının ondalık kısmının iki basamakla sınırlanacağını, gerisinin yuvarlanır
        item_yaw = QTableWidgetItem(f"{yaw:.2f}")
        item_roll = QTableWidgetItem(f"{roll:.2f}")
        
        self.table_widget.setItem(body_index, 0, item_index)
        self.table_widget.setItem(body_index, 1, item_pitch)
        self.table_widget.setItem(body_index, 2, item_yaw)
        self.table_widget.setItem(body_index, 3, item_roll)
        
#Ana program bloğu. Qapplication ve cameraapp örneği oluşturulup gösterilir. Ana döngü başlar
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = CameraApp()
    mainWindow.show() 
    sys.exit(app.exec_())





