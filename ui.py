import tkinter as tk
from tkinter import filedialog

class LicensePlateRecognitionApp:
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
        # Thêm chức năng tải ảnh ở đây
        pass

    def quit_application(self):
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateRecognitionApp(root)
    root.mainloop()
