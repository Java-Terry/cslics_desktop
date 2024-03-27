from ultralytics import YOLO

# load pretrained model
model = YOLO('/home/java/Java/ultralytics/runs/detect/train - alor_atem_1000/weights/best.pt')

# train the model
model.train(data='cslics_desktop.yaml', 
            epochs=1000, 
            imgsz=640,
            workers=10,
            cache=True,
            amp=False,
            batch=-1,
            patience = 300
            )

print('done')