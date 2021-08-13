# AITempoRun

Repo này để giải quyết cuộc thi AI Tempo Run được tổ chức bởi CLB AI thuộc khoa Computer Science, UIT.
Chúng tôi chuẩn bị file ipynb để chạy end-to-end.

## Cài đặt
Sau khi clone về:
```
cd AITempoRun
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
pip install dict_trie
python setup.py install
python setup.py build develop
```

## Tải model về
```
gdown --id 16prmMkZx9mTOygnRRKjM3Af_nUapcItc
```
Bỏ vào thư mục models

```
mv finalModel.pth models
```

## Test
```python
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import read_image
from adet.config import get_cfg
from tqdm import tqdm_notebook
import time
import glob
import os
from utils import prepare_cfg

cfg_file = './cfgs/config.yaml'
model_path = './models/finalModel.pth'
path_test = '/content/drive/MyDrive/AITempoRun/public_test' # đường dẫn thư mục ảnh test, các bạn thay đổi đường dẫn này cho phù hợp

cfg = utils.prepare_cfg(cfg_file, model_path)
predictor = DefaultPredictor(cfg)

print('Read predictor successfully. We are ready for predictions')

list_txts = []
start_time = time.time()
for img_path in tqdm_notebook(glob.glob(os.path.join(path_test,'*'))):
    img = read_image(img_path, format="BGR")
    try:
        predictions = predictor(img)
        raw_points = predictions['instances'].beziers.cpu().detach().numpy()
        for p in raw_points:
            p = [list(map(int,p[i:i + 2])) for i in range(0, len(p), 2)]
            points = [p[0]] + [p[3]] + [p[4]] + [p[7]]
            export_str = img_path.split('/')[-1] + ','
            for _, _p in enumerate(points):
                export_str += ','.join(list(map(str,list(map(int,_p)))))
                if not (_ == len(points) - 1):
                    export_str += ','
            list_txts.append(export_str)
    except:
        print('Error at', img_path)
end_time = time.time()

print('Done. We are writing our predictions to submission.txt')

list_txts = sorted(list_txts)
f_submission = open('submission.txt', 'w')
for _, line in enumerate(list_txts):
    f_submission.write(line + '\n') if _ != len(list_txts) -1 else f_submission.write(line)
f_submission.close()

print('Done! Thanks for waiting, the predict time is {}'.format(end_time - start_time))
```

Kết quả được lưu ở file `submission.txt`

## Acknowledgement
Repo này được sử dụng từ [ABCNet](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/BAText) và [dict-guided](https://github.com/VinAIResearch/dict-guided)
