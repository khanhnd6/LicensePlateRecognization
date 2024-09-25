from ultralytics import YOLO
from PIL import Image

model = YOLO('best.pt')

results = model('test\AQUA1_25697_checkin_2020-10-23-8-53U3VaS26nFf_jpg.rf.e9cd0b6a0e419d91638d187df63202f4.jpg', conf = 0.5, save_crop = True)

for r in results:
    print(r.boxes)
    im_aray = r.plot()
    im = Image.fromarray(im_aray[..., ::-1])
    im.save('abc.jpg')
