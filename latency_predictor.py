# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

# AutoMoE: Neural Architecture Search for Efficient Sparsely Activated Transformers
# Ganesh Jawahar, Subhabrata Mukherjee, Xiaodong Liu, Young Jin Kim, Muhammad Abdul-Mageed, Laks V. S. Lakshmanan, Ahmed Hassan Awadallah, Sebastien Bubeck, Jianfeng Gao
# Paper: https://arxiv.org/abs/2210.07535

import random
import configargparse
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error
import sys
from fairseq import utils
import xgboost as xgb

class Net(nn.Module):
    def __init__(self, feature_dim, hidden_dim, hidden_layer_num):
        super(Net, self).__init__()

        self.first_layer = nn.Linear(feature_dim, hidden_dim)

        self.layers = nn.ModuleList()

        for i in range(hidden_layer_num):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.predict = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.first_layer(x))

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))

        x = self.predict(x)

        return x


class LatencyPredictor(object):
    def __init__(self, feature_norm, lat_norm, ckpt_path, lat_dataset_path='./latency_dataset/lat.tmp', feature_dim=10, hidden_dim=400, hidden_layer_num=3, train_steps=5000, bsz=128, lr=1e-5, use_sklearn_predictor=None):
        self.dataset_path = lat_dataset_path
        self.feature_norm = np.array(feature_norm)
        self.lat_norm = lat_norm
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.hidden_layer_num = hidden_layer_num
        self.ckpt_path = ckpt_path

        self.dataset = None

        self.train_x = None
        self.train_y = None

        self.valid_x = None
        self.valid_y = None

        self.test_x = None
        self.test_y = None

        self.model = Net(self.feature_dim, self.hidden_dim, self.hidden_layer_num)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.train_steps = train_steps
        self.bsz = bsz

    def train(self):
        for i in range(self.train_steps):
            sample_ind = random.sample(range(len(self.train_x)), k=self.bsz)
            sample_x = [self.train_x[sample_ind[k]] for k in range(self.bsz)]
            sample_y = [self.train_y[sample_ind[k]] for k in range(self.bsz)]

            sample_x_tensor = torch.Tensor(sample_x)
            sample_y_tensor = torch.Tensor(sample_y)

            prediction = self.model(sample_x_tensor).squeeze()

            loss = self.criterion(prediction, sample_y_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # validation
            if i % 100 == 0:
                with torch.no_grad():
                    sample_x_tensor = torch.Tensor(self.valid_x)
                    sample_y_tensor = torch.Tensor(self.valid_y)

                    prediction = self.model(sample_x_tensor).squeeze()
                    loss = self.criterion(prediction, sample_y_tensor)
                    print(f"Validation loss at {i} steps: {loss}")

        # test
        with torch.no_grad():
            sample_x_tensor = torch.Tensor(self.test_x)
            sample_y_tensor = torch.Tensor(self.test_y)
            prediction = self.model(sample_x_tensor).squeeze()
            print(prediction.size(), sample_y_tensor.size())
            loss = self.criterion(prediction, sample_y_tensor)
            print(loss)
            print(f"Predicted latency: {prediction}")
            print(f"Real latency: {self.test_y}")
            print(f"Loss: {loss}")

            print(f"RMSE: {np.sqrt(self.criterion(self.lat_norm*sample_y_tensor, self.lat_norm*prediction))}")
            print(f"MAPD: {torch.mean(torch.abs((sample_y_tensor - prediction) / sample_y_tensor))}")

        torch.save(self.model.state_dict(), self.ckpt_path)

    def load_ckpt(self):
        self.model.load_state_dict(torch.load(self.ckpt_path))

    def predict_lat(self, config):
        with torch.no_grad():
            print(config)
            features = utils.get_config_features(config, None)
            features = features[0:len(self.feature_norm)] # TODO: proper fix needed. introduced when adding number of experts to route to dimension
            features_norm = np.array(features) / self.feature_norm
            features_norm = np.array(features) / self.feature_norm

            prediction = self.model(torch.Tensor(features_norm)).item() * self.lat_norm

        return prediction

    def split(self, train_size=None, fixed_val_test_size=None):
        sample_num = len(self.dataset['x'])
        if not fixed_val_test_size:
            train_num = int(np.floor(0.8 * sample_num))
            valid_num = int(np.floor(0.1 * sample_num))
        else:
            train_num = train_size
            valid_num = fixed_val_test_size

        self.train_x = self.dataset['x'][:train_num]
        self.train_y = self.dataset['y'][:train_num]

        self.valid_x = self.dataset['x'][train_num:(train_num+valid_num)]
        self.valid_y = self.dataset['y'][train_num:(train_num+valid_num)]

        self.test_x = self.dataset['x'][(train_num+valid_num):]
        self.test_y = self.dataset['y'][(train_num+valid_num):]

    def read_dataset(self, lat_features=None, add_overall_model_size=None, add_active_model_size=None):
        features_norm_all = []
        lats_all = []
        with open(self.dataset_path, 'r') as fid:
            next(fid) # skip first line of CSV
            for line in fid:
                if not lat_features:
                    features = line.split(',')[:self.feature_dim]
                    features_eval = list(map(eval, features))
                    features_norm = np.array(features_eval) / self.feature_norm
                    features_norm_all.append(features_norm)

                    lats = line.split(',')[self.feature_dim:]
                    total_lat = eval(lats[0]) + eval(lats[1])
                    lats_all.append(total_lat / self.lat_norm)
                else:
                    features = json.loads(line.strip())
                    features_norm = []
                    # {"encoder_encoder_embed_dim": 640, "encoder_encoder_layer_num": 6, "encoder_encoder_ffn_embed_dim": [2048, 3072, 1024, 1024, 512, 512], "encoder_encoder_self_attention_heads": [4, 2, 4, 8, 2, 2], "decoder_decoder_embed_dim": 640, "decoder_decoder_layer_num": 3, "decoder_decoder_ffn_embed_dim": [512, 3072, 3072], "decoder_decoder_self_attention_heads": [8, 2, 2], "decoder_decoder_ende_attention_heads": [8, 8, 4], "encoder_encoder_n_experts": [1, 1, 1], "decoder_decoder_n_experts": [1, 1, 1], "encoder_latencies": 6.844480037689209, "decoder_latencies": 124.20365142822266}
                    # 640 6 2048 6 640 6 2048 6 6 2 2 2
                    # encoder_embed_dim,encoder_layer_num,encoder_ffn_embed_dim_avg,encoder_self_attention_heads_avg,decoder_embed_dim,decoder_layer_num,decoder_ffn_embed_dim_avg,decoder_self_attention_heads_avg,decoder_ende_attention_heads_avg,decoder_arbitrary_ende_attn_avg,encoder_n_experts_avg,decoder_n_experts_avg,latency_mean_encoder,latency_mean_decoder,latency_std_encoder,latency_std_decoder
                    for feat in sorted(features):
                        if feat == "encoder_latencies" or feat == "decoder_latencies" or "size" in feat:
                            continue
                        if feat.endswith("encoder_embed_dim") or feat.endswith("layer_num") or feat.endswith("decoder_embed_dim"):    
                            feat_val = features[feat]
                            feat_norm = None
                            if feat.endswith("embed_dim"):
                                feat_norm = 640
                            elif feat.endswith("layer_num"):
                                feat_norm = 6
                            features_norm.append(feat_val/feat_norm)
                        else:
                            for i in range(6):
                                feat_val = features[feat][i] if i < len(features[feat]) else 0
                                feat_norm = None
                                if feat.endswith("ffn_embed_dim"):
                                    feat_norm = 2048
                                elif feat.endswith("self_attention_heads") or feat.endswith("ende_attention_heads"):
                                    feat_norm = 6
                                elif feat.endswith("n_experts"):
                                    feat_norm = 2
                                features_norm.append(feat_val/feat_norm)
                    if add_overall_model_size:
                        features_norm.append(features["model_size"]/200000000)
                    if add_active_model_size:
                        features_norm.append(features["active_model_size"]/200000000)
                    features_norm_all.append(features_norm)
                    total_lat = features["encoder_latencies"] + features["decoder_latencies"]
                    lats_all.append(total_lat / self.lat_norm)

        tmp = list(zip(features_norm_all, lats_all))
        random.shuffle(tmp)
        features_norm_all, lats_all = zip(*tmp)
        self.dataset = {'x': features_norm_all, 'y': lats_all}

if __name__=='__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--configs', required=True, is_config_file=True)
    parser.add_argument('--dataset-path')

    parser.add_argument('--lat-dataset-path', type=str, default='./latency_dataset/lat.tmp', help='the path to read latency dataset')
    parser.add_argument('--feature-norm', type=float, nargs='+', default=[640, 6, 2048, 6, 640, 6, 2048, 6, 6, 2], help='normalizing factor for each feature')
    parser.add_argument('--lat-norm', type=float, default=200, help='normalizing factor for latency')
    parser.add_argument('--feature-dim', type=int, default=10, help='dimension of feature vector')
    parser.add_argument('--hidden-dim', type=int, default=400, help='hidden dimension of FC layers in latency predictor')
    parser.add_argument('--hidden-layer-num', type=int, default=3, help='number of FC layers')
    parser.add_argument('--ckpt-path', type=str, default='latency_dataset/ckpts/tmp.pt', help='path to save latency predictor weights')
    parser.add_argument('--train-steps', type=int, default=5000, help='latency predictor training steps')
    parser.add_argument('--bsz', type=int, default=128, help='latency predictor training batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='latency predictor training learning rate')
    parser.add_argument('--fixed-val-test-size', type=int, default=-1, help='control for the size of validation and test set')
    parser.add_argument('--lat-features', type=str, default=None, help="latency predictor features")

    parser.add_argument('--add-overall-model-size', type=str, default=None, help="add overall model size")
    parser.add_argument('--add-active-model-size', type=str, default=None, help="add active model size")
    parser.add_argument('--use-sklearn-predictor', type=str, default=None, help="use classifier from sklearn")

    args = parser.parse_args()
    print(args)

    predictor = LatencyPredictor(lat_dataset_path=args.lat_dataset_path,
                           feature_norm=args.feature_norm,
                           lat_norm=args.lat_norm,
                           feature_dim=args.feature_dim,
                           hidden_dim=args.hidden_dim,
                           hidden_layer_num=args.hidden_layer_num,
                           ckpt_path=args.ckpt_path,
                           train_steps=args.train_steps,
                           bsz=args.bsz,
                           lr=args.lr)

    predictor.read_dataset(lat_features=args.lat_features, add_overall_model_size=args.add_overall_model_size, add_active_model_size=args.add_active_model_size)
    if args.use_sklearn_predictor is not None:
        model = None
        if args.use_sklearn_predictor == "LinearRegression":
            model = linear_model.LinearRegression()
        elif args.use_sklearn_predictor == "GradientBoostingRegressor":
            params = {"n_estimators": 500, "max_depth": 4, "min_samples_split": 5, "learning_rate": 0.01, "loss": "squared_error"}
            model = ensemble.GradientBoostingRegressor(**params)
        elif args.use_sklearn_predictor == "XGBRegressor":
            model = xgb.XGBRegressor(objective="reg:linear", random_state=42, n_estimators=1000)
        predictor.split()
        model.fit(predictor.train_x, predictor.train_y)
        y_pred = model.predict(predictor.test_x)
        print(y_pred[0:10])
        print(predictor.test_y[0:10])
        print("Loss = %.2f"%(mean_squared_error(predictor.lat_norm*np.array(predictor.test_y), predictor.lat_norm*np.array(y_pred))))
        print("RMSE = %.2f"%(np.sqrt(mean_squared_error(predictor.lat_norm*np.array(predictor.test_y), predictor.lat_norm*np.array(y_pred)))))
        sample_x_tensor = torch.Tensor(predictor.test_x)
        sample_y_tensor = torch.Tensor(predictor.test_y)
        prediction = torch.Tensor(y_pred)
        loss = predictor.criterion(prediction, sample_y_tensor)
        print(f"Loss: {loss}")
        print(f"RMSE: {np.sqrt(predictor.criterion(predictor.lat_norm*sample_y_tensor, predictor.lat_norm*prediction))}")
        sys.exit(0)
    if args.fixed_val_test_size == -1:
        predictor.split()
        predictor.train()
    else:      
        for train_size in [1600, 4600, 9600]:  
            predictor.model = Net(args.feature_dim, args.hidden_dim, args.hidden_layer_num)
            predictor.optimizer = torch.optim.Adam(predictor.model.parameters(), lr=args.lr)
            predictor.criterion = torch.nn.MSELoss()
            predictor.split(train_size=train_size, fixed_val_test_size=args.fixed_val_test_size)
            predictor.train()

    print('Latency predictor training finished')

    predictor.load_ckpt()
    config_example = {
        'encoder': {
            'encoder_embed_dim': 512,
            'encoder_layer_num': 6,
            'encoder_ffn_embed_dim': [3072, 3072, 3072, 3072, 3072, 3072],
            'encoder_self_attention_heads': [8, 8, 8, 8, 8, 4],
            'encoder_n_experts': [1,2, 1, 1, 1, 1]
        },
        'decoder': {
            'decoder_embed_dim': 512,
            'decoder_layer_num': 5,
            'decoder_ffn_embed_dim': [2048, 3072, 3072, 3072, 1024],
            'decoder_self_attention_heads': [4, 8, 8, 4, 4],
            'decoder_ende_attention_heads': [4, 8, 8, 4, 4],
            'decoder_arbitrary_ende_attn':  [-1, 1, 1, 1, 1],
            'decoder_n_experts': [1, 2, 2, 2, 2]
        }
    }

    #predict = predictor.predict_lat(config_example)
    #print(f'Example config: {config_example}')
    #print(f'Example latency: {predict}')
