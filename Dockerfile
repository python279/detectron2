FROM harbor.uat.enflame.cc/library/enflame.cn/detectron2:v0

USER appuser
WORKDIR /home/appuser/detectron2_repo

COPY --chown=appuser:sudo ./ ./

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install --user --upgrade pip \
    && pip install --user -e . \
    && pip install --user -r requirements.txt \
    && pip install --user -e projects/DensePose

# download pretrain checkpoint
RUN mkdir -p checkpoint/densepose/densepose_rcnn_R_50_FPN_s1x \
    && wget -O checkpoint/densepose/densepose_rcnn_R_50_FPN_s1x/model_final_162be9.pkl \
    "http://mirrors.uat.enflame.cc/enflame.cn/maas/densepose/densepose_rcnn_R_50_FPN_s1x/model_final_162be9.pkl"

ENTRYPOINT ["python3", "-u", "server.py"]