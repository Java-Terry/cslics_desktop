
"""Just something to get the larve label into the model"""

from ultralytics import YOLO

# load pretrained model
model = YOLO('/home/java/Java/ultralytics/runs/detect/train - alor_atem_1000/weights/best.pt')

# train the model
model.train(data='cslics_desktop.yml', 
            epochs=10, 
            imgsz=640,
            workers=10,
            cache=True,
            amp=False,
            batch=-1
            )

print('done')