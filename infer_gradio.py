from ultralytics import YOLO
import cv2
import torch
import timm
import numpy as np
import torchvision.transforms as T
from PIL import Image
import gradio as gr

class Get_infor() :
    def __init__(self,path_yolo, path_cls, device):
        self.device = device
        self.model_detect = YOLO(path_yolo)
        self.model_cls = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
        self.model_cls.load_state_dict(torch.load(path_cls, map_location='cpu'))
        self.model_cls.eval()
        self.transfrom = T.Compose([
                        T.Resize((224,224)), 
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
        self.label = ["1", "2"]
        self.txt = ""
        self.respon = ''
    def detect(self, img):
        results = self.model_detect(source=img, imgsz=640, device=self.device, verbose=False)
        results = results[0].cpu().numpy()
        list_bbox= []
        if len(results.boxes):
            for box in results.boxes:
                xyxy = box.xyxy[0]
                list_bbox.append(xyxy)
        return list_bbox
    def get(self, img):
        imgs = Image.fromarray(img) 
        list_bbox = self.detect(imgs)
        list_lb = []

        if len(list_bbox):
            for box in list_bbox:
                x1,y1,x2,y2, = int(box[0]), int(box[1]), int(box[2]), int(box[3]), 
                img_cls = imgs.crop((x1,y1,x2,y2))
                img_cls = self.transfrom(img_cls).unsqueeze(0) 
                output = self.model_cls(img_cls)
                max_id = np.argmax(output.detach().numpy())
                lb = self.label[max_id]
                list_lb.append(lb)
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2) 
        if len(list_lb)==0: self.txt = "cây không có bệnh"

        if len(list_bbox)==1:
            with open(f"data/{list_lb[0]}.txt", 'r', encoding='utf-8') as f:
                self.txt = f.read()
        if len(list_bbox)==2:
            with open(f"data/1.txt", 'r', encoding='utf-8') as f:
                self.txt = f.read()
            with open(f"data/2.txt", 'r', encoding='utf-8') as f:
                self.txt += f.read()
        return img,self.txt
        
if __name__=="__main__":
    path_yolo = 'model/phat_hien.pt'
    path_cls = 'model/phan_loai.pth'
    check = Get_infor(path_yolo=path_yolo, path_cls=path_cls, device='cpu')
    custom_css = """
        body {
            background-image: url('./static/APC_0076.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        """
    iface = gr.Interface(
        fn=check.get,
        inputs="image",
        outputs=[gr.Image(type="numpy", label="Processed Image"), gr.Textbox(label="Model Prediction")],
        live = True, 
        title="HỆ THỐNG NHẬN DIỆN SÂU BỆNH CHÈ VÀ GỢI Ý GIẢI PHÁP KHẮC PHỤC", 
        description="<h2>Nhóm Nguyên cứu</h2> <h3> Đoàn Hà Anh Tuấn-Lớp 12A1</h3><h3>  Lê Bá Thu-Lớp 12A1</h3> <h2>Giáo Viên Hướng Dẫn</h2> <h3>Nguyễn Thị Hường</h3>",
        css=custom_css


    )
    iface.launch(share=True)

        
            

        
