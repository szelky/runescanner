import img2pdf
import cv2
import numpy as np
import imutils
from transform import four_point_transform
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class DocumentScannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Belge Tarayıcı")
        self.root.geometry("900x750")
        
        self.current_scan = None

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(side="top", pady=10)

        self.btn_select = tk.Button(self.btn_frame, text="Resim Seç ve Tara", command=self.scan_document)
        self.btn_select.grid(row=0, column=0, padx=10)

        self.btn_save = tk.Button(self.btn_frame, text="Taramayı Kaydet", command=self.save_scan, state="disabled")
        self.btn_save.grid(row=0, column=1, padx=10)

        self.label_img = tk.Label(root, text="Lütfen bir resim seçin")
        self.label_img.pack(expand=True, pady=10)

    def scan_document(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        try:
            image = cv2.imread(file_path)
            ratio = image.shape[0] / 500.0
            orig = image.copy()
            image_res = imutils.resize(image, height=500)

            gray = cv2.cvtColor(image_res, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(gray, 75, 200)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            
            cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

            screenCnt = None
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    screenCnt = approx
                    break

            if screenCnt is None:
                messagebox.showwarning("Hata", "Kağıt kenarı algılanamadı!")
                return

            self.current_scan = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
            
            self.btn_save.config(state="normal")
            self.show_on_gui(self.current_scan)

        except Exception as e:
            messagebox.showerror("Hata", f"İşlem hatası: {str(e)}")

    def save_scan(self):
        if self.current_scan is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF Belgesi", "*.pdf"), ("JPEG", "*.jpg"), ("PNG", "*.png")]
            )
            
            if file_path:
                if file_path.lower().endswith(".pdf"):
                    # 1. Görüntüyü PDF için uygun bir boyuta (örneğin A4 oranına yakın) yeniden boyutlandıralım
                    # Bu adım PDF'in devasa görünmesini engeller.
                    target_height = 1123 # A4 standart yüksekliği (yaklaşık 150 DPI)
                    final_img = imutils.resize(self.current_scan, height=target_height)
                    
                    # 2. PDF'e dönüştür
                    _, img_encoded = cv2.imencode(".jpg", final_img)
                    
                    # layout_fun ile sayfayı A4 boyutuna (veya görselin boyutuna) sabitleyebiliriz
                    a4inpt = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297)) # A4 boyutları
                    layout_fun = img2pdf.get_layout_fun(a4inpt)
                    
                    pdf_bytes = img2pdf.convert(img_encoded.tobytes(), layout_fun=layout_fun)
                    
                    with open(file_path, "wb") as f:
                        f.write(pdf_bytes)
                else:
                    cv2.imwrite(file_path, self.current_scan)
                    
                messagebox.showinfo("Başarılı", "Dosya uygun boyutta kaydedildi!")

    def show_on_gui(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((800, 600))
        
        img_tk = ImageTk.PhotoImage(img_pil)
        self.label_img.config(image=img_tk, text="")
        self.label_img.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentScannerGUI(root)
    root.mainloop()
