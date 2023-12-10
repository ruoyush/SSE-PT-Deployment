# SSE-PT-Deployment
Demo for deploying SSE-PT(SSE-PT: Temporal Collaborative Ranking Via Personalized Transformer) https://github.com/wuliwei9278/SSE-PT/tree/master

1. install dependencies
streamlit==1.23.1
tensorflow==1.14.0
tensorflow-estimator==1.14.0
pandas==1.0.3
numpy==1.18.1

2. download MovieLens 1M dataset, and put `movies.dat` and `users.dat` into the same folder of model.py

3. dowanlod the processed dataset of `ml1m.txt` from SSE-PT repo, and put it into the same folder of model.py

4. train the SSE-PT model by `ml1m.txt` and save the model by tensorflow saver (model.py may need to be changed according to the required tensor and operation, refer to the 'opcode' file)
    - repo: https://github.com/wuliwei9278/SSE-PT/tree/master
    - traning environment: Aliyun Cloud: 12Core, 92G RAM, GPU Nvidia-V100 32G, cuda 10.13.0, nvidia 450.80.02
    - refer to the revised `main.py` file in this repo

5. streamlit run `model.py`