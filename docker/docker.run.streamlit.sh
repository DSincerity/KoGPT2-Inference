# streamlit
docker run -d -t \
    -v /home/ubuntu/KoGPT2-Inference:/home/ubuntu/KoGPT2-Inference \
    -p 8082:8501 \
    --workdir /home/ubuntu/KoGPT2-Inference \
    --name web-client samdobson/streamlit streamlit_client.py
