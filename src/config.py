from datetime import datetime

class Config:
    nclasses = 50
    max_seq_len = 41
    max_labels_per_classes = 10
    epoch=30
    dropout = 0.1
    bert_features = 768
    noise = 100
    generator_hidden_layer = 4096
    discriminator_hidden_layer = 256
    device = "cpu"
    batch_size = 32
    gen_learning_rate = 0.0001
    dis_learning_rate = 0.00001
    save_to = './trained.chkpt'
    session = datetime.now().strftime("%m_%d_%Y, %H:%M:%S")
