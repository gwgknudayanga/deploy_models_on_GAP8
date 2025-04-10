# deploy_models_on_GAP8
- GAP8 is a ultra low power processor/microcontroller introduced by greenwaves to deploy small deep learning models which would be very useful at the edge.
(https://github.com/GreenWaves-Technologies/gap_sdk)
- These processors have limited computing and memory resources. Hence our deep learning models need to small enough to fit to their resources.
- Also the models need to be quantized before deploying.
- When deploying we need to follow their guidelines.
- The GAP8 has simulator/emulator called gvsoc that simulates the behavior of GAP8 hardware.
- We can run on this gvsoc simulator and get some idea about accuracy and latency and also bottleneck layers of our model as it runs on real GAP8 hardware.
