from datetime import datetime

class Config:
    epoch=100
    dropout = 0.1
    bert_features = 768
    noise = 100
    generator_hidden_layer = 1000
    discriminator_hidden_layer = 300
    device = "cuda"
    batch_size = 32
    gen_learning_rate = 0.0001
    dis_learning_rate = 0.00001
    session = datetime.now().strftime("%m_%d_%Y, %H:%M:%S")
    save_path = './saves'
    code_path = 'code'
    log_path = './logs'
