# API server for koGPT2(SKT)
docker run --rm -d -t --gpus '"device=1"' --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --shm-size=4g \
    -v /home/ubuntu/KoGPT2-Inference:/home/ubuntu/KoGPT2-Inference \
    -p 8080:8080 \
    --workdir /home/ubuntu/KoGPT2-Inference \
    --name kogpt2-api-server kogpt2:0.1 python3 app_koGPT2.py


# API server for koGPT2(HF)
docker run --rm -d -t --gpus '"device=2"' --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --shm-size=4g \
    -v /home/ubuntu/KoGPT2-Inference:/home/ubuntu/KoGPT2-Inference \
    -p 8081:8081 \
    --workdir /home/ubuntu/KoGPT2-Inference \
    --name kogpt2-HF-api-server kogpt2:0.1 /bin/bash -c "pip3 install transformers pytorch_lightning-U; python3 app_HF.py"
