import cv2
import numpy as np
import datetime

# โหลดภาพ
image = cv2.imread('coins.jpg')  # เปลี่ยน path ตามตำแหน่งของไฟล์ภาพ
if image is None:
    print("ไม่พบไฟล์ภาพ")
else:
    # ปรับขนาดของภาพ
    scale_percent = 15  # ปรับเป็นเปอร์เซ็นต์ เช่น 50% ของขนาดเดิม
    width = int(image.shape[1] * scale_percent / 105)
    height = int(image.shape[0] * scale_percent / 105)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    output = resized_image.copy()
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # ใช้ Gaussian Blur เพื่อลด Noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # เริ่มบันทึกเวลาการตรวจจับ
    start_time = datetime.datetime.now()

    # ใช้ Hough Circle Transform เพื่อตรวจจับวงกลม
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=50)

    # ตรวจสอบว่ามีวงกลมถูกตรวจจับหรือไม่
    coin_count = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        coin_count = len(circles)

        # วาดวงกลมรอบเหรียญที่ตรวจจับได้
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # วาดจุดตรงกลาง

    # หยุดบันทึกเวลาเมื่อการตรวจจับเสร็จสิ้น
    end_time = datetime.datetime.now()
    detection_time = (end_time - start_time).total_seconds()  # คำนวณเวลาเป็นวินาที

    # แสดงจำนวนเหรียญ
    cv2.putText(output, f"Coins detected: {coin_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # เพิ่ม timestamp และเวลาที่ใช้ในการตรวจจับ
    timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(output, f"Timestamp: {timestamp}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(output, f"Detection Time: {detection_time:.2f} seconds", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # แสดงผลลัพธ์
    cv2.imshow("Detected Coins", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
