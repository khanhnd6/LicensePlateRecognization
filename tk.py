import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import easyocr
import glob
import shutil
import os
import numpy as np

# Tải mô hình đã train
model = YOLO('best.pt')

def sharpened(image):
    # Áp dụng làm mờ Gaussian
    blurred = cv2.GaussianBlur(image, (9, 9), 10)

    # Trừ ảnh mờ khỏi ảnh gốc để làm nét
    sharpened = cv2.addWeighted(image, 2.5, blurred, -2, 1)

    # Lưu ảnh nét
    cv2.imwrite('sharpened_image.jpg', sharpened)
    
    return sharpened

def deskew_plate(image):
    # Bước 1: Chuyển đổi ảnh sang thang độ xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bước 2: Làm mờ nhẹ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Bước 3: Phát hiện các cạnh trong ảnh bằng Canny Edge Detection
    edged = cv2.Canny(blurred, 30, 150)

    # Bước 4: Tìm các đường viền (contours) trong ảnh đã phát hiện cạnh
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Bước 5: Kiểm tra nếu có ít nhất 1 đường viền được tìm thấy
    if len(contours) == 0:
        return image, False

    # Bước 6: Lấy đường viền lớn nhất dựa trên diện tích (contour lớn nhất)
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Bước 7: Tìm các góc (approximate) của đường viền
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Bước 8: Kiểm tra nếu đường viền có 4 góc (hình tứ giác)
    if len(approx) == 4:
        # Bước 9: Sắp xếp lại các điểm góc theo thứ tự đúng
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # Tính tổng (sum) của các điểm để xác định vị trí góc
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Góc trên bên trái
        rect[2] = pts[np.argmax(s)]  # Góc dưới bên phải

        # Tính hiệu (diff) để xác định các góc còn lại
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Góc trên bên phải
        rect[3] = pts[np.argmax(diff)]  # Góc dưới bên trái

        # Bước 10: Tính toán chiều rộng và chiều cao của biển số
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # Bước 11: Tạo ma trận biến đổi (transformation matrix) cho việc làm phẳng
        dst = np.array([
            [0, 0], 
            [maxWidth - 1, 0], 
            [maxWidth - 1, maxHeight - 1], 
            [0, maxHeight - 1]], dtype="float32")

        # Bước 12: Sử dụng ma trận để thực hiện biến đổi góc nhìn (perspective transform)
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # Trả về ảnh đã được làm phẳng
        return warped, True

    # Nếu không tìm được 4 góc, trả về ảnh gốc
    return image, False

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Xóa thư mục và tất cả các tệp con

def detect_license_plate(image_path):
    # Đọc ảnh đầu vào từ file
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")

    # Thực hiện dự đoán với mô hình YOLOv10
    results = model(image_path, conf=0.5, save_crop=True)

    # Lưu ảnh với bounding box
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save('bbox_image.jpg')  # Lưu ảnh với bounding box
        bbox_image_path = 'bbox_image.jpg'

    # Đường dẫn đến thư mục chứa các ảnh bounding box đã cắt
    cropped_image_path = 'runs/detect/predict/crops/bien_so/'

    # Sử dụng glob để lấy tất cả các file ảnh đã cắt (ví dụ: *.jpg)
    cropped_images = glob.glob(cropped_image_path + '*.jpg')

    # Kiểm tra danh sách ảnh cắt và lấy ảnh đầu tiên
    if cropped_images:
        cropped_image_path = cropped_images[0]  # Chọn ảnh đầu tiên nếu có
    else:
        raise ValueError("Không tìm thấy ảnh cắt nào.")

    return bbox_image_path, cropped_image_path

# EasyOCR để nhận diện ký tự từ ảnh
reader = easyocr.Reader(['en'])

def get_text_from_image(image_path):
    results = reader.readtext(image_path)
    if results:
        # Kết hợp tất cả các dòng văn bản vào một chuỗi duy nhất
        text = ' '.join([result[-2] for result in results])
        confidence = sum([result[-1] for result in results]) / len(results)  # Trung bình độ chính xác
        return text, confidence
    return '', 0

# Lớp giao diện UI
class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện biển số xe")

        # Căn chỉnh bố cục
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_rowconfigure(3, weight=0)
        self.root.grid_rowconfigure(4, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        # Các vùng hiển thị ảnh
        self.img_label = tk.Label(root, bg='white')
        self.img_label.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        self.bbox_label = tk.Label(root, bg='white')
        self.bbox_label.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

        self.crop_label = tk.Label(root, bg='white')
        self.crop_label.grid(row=0, column=2, padx=10, pady=10, sticky='nsew')

        # Ô hiển thị kết quả ký tự và độ chính xác
        self.plate_text = tk.StringVar()
        self.confidence = tk.StringVar()

        result_frame = tk.Frame(root)
        result_frame.grid(row=1, column=0, columnspan=3, pady=10)

        tk.Label(result_frame, text="Biển số nhận diện:", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(result_frame, textvariable=self.plate_text, width=40, font=("Arial", 12)).grid(row=0, column=1, padx=10, pady=5, sticky="w")

        tk.Label(result_frame, text="Độ chính xác:", font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(result_frame, textvariable=self.confidence, width=40, font=("Arial", 12)).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # Khung chứa các nút
        button_frame = tk.Frame(root)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)

        # Nút tải ảnh
        self.upload_button = tk.Button(button_frame, text="Tải ảnh", command=self.upload_image, font=("Arial", 12))
        self.upload_button.grid(row=0, column=0, padx=10)

        # Nút kết thúc
        self.quit_button = tk.Button(button_frame, text="Kết thúc", command=self.quit_application, font=("Arial", 12))
        self.quit_button.grid(row=0, column=1, padx=10)

    def upload_image(self):
        # Xóa thư mục chứa ảnh cũ trước khi xử lý ảnh mới
        delete_folder('runs/detect/predict')

        # Chọn ảnh từ file dialog
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)

            # Nhận diện và cắt biển số
            bbox_img_path, cropped_img_path = detect_license_plate(file_path)
            self.display_bbox(bbox_img_path)

            # Đọc ảnh đã cắt
            cropped_img = cv2.imread(cropped_img_path)

            # Làm phẳng ảnh biển số
            deskewed_img, is_deskewed = deskew_plate(cropped_img)

            # Nếu làm phẳng thành công, sử dụng ảnh đã làm phẳng; ngược lại sử dụng ảnh gốc
            if is_deskewed:
                processed_img = deskewed_img
            else:
                processed_img = cropped_img

            # Làm nét ảnh
            sharpened_img = sharpened(processed_img)

            # Lưu ảnh đã làm nét (nếu cần)
            sharpened_img_path = 'sharpened_image.jpg'
            cv2.imwrite(sharpened_img_path, sharpened_img)

            # Hiển thị ảnh đã làm nét
            self.display_sharpened(sharpened_img_path)

            # Nhận diện ký tự bằng EasyOCR
            text, conf = get_text_from_image(sharpened_img_path)
            tex = ''
            for i in text:
                if i  in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                         'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
                    tex += i
            a = len(tex)
            tex = tex[:3] + '-' + tex[3:]
            if(a==8):
                tex = tex[:7] + '.' + tex[7:]
            self.plate_text.set(tex)
            self.confidence.set(f"{conf*100:.2f}%")

            # Xóa thư mục predict sau khi hoàn tất xử lý
            delete_folder('runs/detect/predict')

    def display_sharpened(self, sharpened_img_path):
        img = Image.open(sharpened_img_path)
        img = img.resize((400, 400))  # Tăng kích thước ảnh
        img = ImageTk.PhotoImage(img)
        self.crop_label.config(image=img)
        self.crop_label.image = img  # Giữ tham chiếu để tránh garbage collection


    def display_image(self, img_path):
        img = Image.open(img_path)
        img = img.resize((400, 400))  # Tăng kích thước ảnh
        img = ImageTk.PhotoImage(img)
        self.img_label.config(image=img)
        self.img_label.image = img  # Giữ tham chiếu để tránh garbage collection

    def display_bbox(self, bbox_img_path):
        img = Image.open(bbox_img_path)
        img = img.resize((400, 400))  # Tăng kích thước ảnh
        img = ImageTk.PhotoImage(img)
        self.bbox_label.config(image=img)
        self.bbox_label.image = img

    def quit_application(self):
        self.root.destroy()  # Đóng ứng dụng

# Chạy ứng dụng
root = tk.Tk()
app = LicensePlateApp(root)
root.mainloop()
