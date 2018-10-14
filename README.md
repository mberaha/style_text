# "Style Transfer from Non-Parallel Text by Cross-Alignment".

This repo contains the PyTorch implementation of the paper by Tianxiao Shen, Tao Lei, Regina Barzilay, and Tommi Jaakkola. NIPS 2017. [arXiv](https://arxiv.org/abs/1705.09655)

Instructions to use the code:

1. Compile the protocol buffer in src/parameters.proto using the command:
 
> protoc --proto_path=src --python_out=src src/parameters.proto

2. Update the parameters value contained in the file resources/params.asciipb with your choiche of the parameters

3. Train the model using the file in scripts/train_yelp.sh
