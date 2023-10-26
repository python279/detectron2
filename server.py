from apply_net import main
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from base64 import b64decode, b64encode
import sys
import tempfile
from uuid import uuid4
import shutil
import os
import traceback
import glob

app = Flask(__name__)


class DensePose(object):
    def __call__(self, image: Image) -> Image or None:
        try:
            tmpdir = tempfile.mkdtemp()
            image_path = os.path.join(tmpdir, "input.jpg")
            output_path = os.path.join(tmpdir, "output.jpg")
            image.save(image_path)
            sys.argv = ["apply_net.py",  "show", "projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml", "checkpoint/densepose/densepose_rcnn_R_50_FPN_s1x/model_final_162be9.pkl", image_path, "dp_segm", "--output", output_path, "--opts", "MODEL.DEVICE", "cpu"]
            main()
            file_list = glob.glob(os.path.join(tmpdir, "output*.jpg"))
            if len(file_list) > 0:
                output_image = Image.open(file_list[0])
            else:
                output_image = None
        except Exception:
            output_image = None
            print(traceback.format_exc())
        finally:
            shutil.rmtree(tmpdir)

        return output_image


dense_pose = DensePose()


@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()  # 获取 POST 请求中的 JSON 数据
    input_image = data["input_image"]
    # 从 base64 解码出图片
    image = Image.open(BytesIO(b64decode(input_image)))
    output_image = dense_pose(image)
    if output_image:
        buffered = BytesIO()
        output_image.save(buffered, format='JPEG')
        output_image = b64encode(buffered.getvalue()).decode('utf-8')
    result = {'result': {'output_image': output_image}}
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
