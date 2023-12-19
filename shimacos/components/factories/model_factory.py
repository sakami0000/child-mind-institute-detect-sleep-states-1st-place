import math
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModel


class CatEncoder(nn.Module):
    def __init__(self):
        super(CatEncoder, self).__init__()
        self.encoder_dict = {}
        encoder_infos = [
            ("extension", 8, 32),
        ]
        for cat, vocab_size, embedding_dim in encoder_infos:
            self.encoder_dict[cat] = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_dict = nn.ModuleDict(self.encoder_dict)
        # self.out_features = sum([val for val in self.n_class_dict.values()])
        self.out_features = sum([embedding_dim for _, _, embedding_dim in encoder_infos])

    def forward(self, x_dict):
        """
        x_dict: [bs, seq_len, 1]
        return: [bs, seq_len, self.encoder.out_features]
        """
        outs = []
        for cat, val in x_dict.items():
            if cat in self.encoder_dict.keys():
                outs.append(self.encoder_dict[cat](val.long()))
        return torch.cat(outs, dim=-1)


class OneHotEncoder(nn.Module):
    def __init__(self):
        super(OneHotEncoder, self).__init__()
        self.encoder_dict = {
            "categoryA": 182,
            "categoryB": 2,
            "categoryC": 2906,
            "categoryD": 3,
            "categoryE": 29,
            "categoryF": 3,
            "unit": 19,
            "categoryA_categoryE": 574,
            "featureA": 27,
            "featureB": 27,
            "featureC": 23,
            "featureD": 28,
            "featureE": 28,
            "featureF": 23,
            "featureG": 27,
            "featureH": 14,
            "featureI": 27,
            "compositionA": 7,
            "compositionB": 24,
            "compositionC": 27,
            "compositionD": 12,
            "compositionE": 25,
            "compositionF": 23,
            "compositionG": 17,
            "compositionH": 26,
            "compositionI": 27,
            "compositionJ": 25,
        }

    def forward(self, x_dict):
        """
        x_dict: [bs, seq_len, 1]
        return: [bs, seq_len, self.encoder.out_features]
        """
        outs = []
        for cat, val in x_dict.items():
            if cat in self.encoder_dict.keys():
                outs.append(F.one_hot(val, self.encoder_dict[cat]))
        return torch.cat(outs, dim=-1)


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        mlp = [
            nn.LazyLinear(config.mlp.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
        ]
        for _ in range(config.mlp.n_layer - 2):
            mlp.extend(
                [
                    nn.Linear(config.mlp.hidden_size, config.mlp.hidden_size),
                    # nn.BatchNorm1d(config.mlp.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate),
                ]
            )
        self.mlp = nn.Sequential(*mlp)
        if config.encoder == "embedding":
            self.encoder = CatEncoder()
        elif config.encoder == "onehot":
            self.encoder = OneHotEncoder()
        else:
            raise NotImplementedError()
        self.last_linear = nn.Linear(config.mlp.hidden_size, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Input:
        """
        feature = batch["feature"]
        embedding = self.encoder(batch)
        feature = torch.cat([batch["feature"], embedding], dim=-1)
        feature = self.mlp(feature)
        out = self.last_linear(feature).squeeze()

        return out

    def get_feature(self):
        return self.feature


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * F.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = F.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class ResNet(nn.Module):
    def __init__(self, in_features, out_features, in_length, kernel_size, dropout=0.0):
        super(ResNet, self).__init__()
        assert kernel_size % 2 == 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_features,
                out_features,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.LayerNorm([out_features, in_length]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_features,
                out_features,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.LayerNorm([out_features, in_length]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x) + x
        return out


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, feature: torch.Tensor):
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight)).float()
        return cosine


class CNN2D(nn.Module):
    def __init__(self, config: DictConfig):
        super(CNN2D, self).__init__()
        self.config = config
        self.cnn = timm.create_model(
            config.backbone,
            pretrained=True,
            # No classifier for ArcFace
            num_classes=0,
            in_chans=3,
        )
        self.in_features = self.cnn.num_features
        self.last_linear = nn.Linear(self.in_features, config.num_classes)

    def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
        feature = self.cnn.forward_features(imgs)
        if "swin" in self.config.backbone:
            feature = feature.mean(1)
        if (
            "vit" not in self.config.backbone
            and "swin" not in self.config.backbone
            and "twins" not in self.config.backbone
        ):
            feature = F.adaptive_avg_pool2d(feature, (1, 1))
            feature = feature.view(feature.size(0), -1)
        return feature

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        imgs = batch["image"].permute(0, 3, 1, 2)
        self.feature = self.forward_features(imgs)
        out = self.last_linear(self.feature)
        return out.squeeze()

    def get_feature(self):
        return self.feature


class MultiCNN2D(nn.Module):
    def __init__(self, config: DictConfig):
        super(CNN2D, self).__init__()
        self.config = config
        self.cnn = timm.create_model(
            config.backbone,
            pretrained=True,
            # No classifier for ArcFace
            num_classes=0,
            in_chans=3,
        )
        self.in_features = self.cnn.num_features
        self.last_linear1 = nn.Linear(self.in_features, config.num_classes)
        self.last_linear2 = nn.Linear(self.in_features, 1)
        self.last_linear3 = nn.Linear(self.in_features, 1)

    def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
        feature = self.cnn.forward_features(imgs)
        if "swinv2" in self.config.backbone:
            feature = feature.mean(1)
        if (
            "vit" not in self.config.backbone
            and "swin" not in self.config.backbone
            and "twins" not in self.config.backbone
        ):
            feature = F.adaptive_avg_pool2d(feature, (1, 1))
            feature = feature.view(feature.size(0), -1)
        return feature

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs = batch["image"].permute(0, 3, 1, 2)
        self.feature = self.forward_features(imgs)
        out1 = self.last_linear1(self.feature)
        out2 = self.last_linear2(self.feature).squeeze()
        out3 = self.last_linear3(self.feature).squeeze()
        return out1, out2, out3

    def get_feature(self):
        return self.feature


class CNNBert(nn.Module):
    def __init__(self, config: DictConfig):
        super(CNNBert, self).__init__()
        self.config = config
        self.cnn = timm.create_model(
            config.backbone,
            pretrained=config.is_pretrained,
            # No classifier for ArcFace
            num_classes=0,
            in_chans=config.image.in_channels,
        )
        if config.is_pretrained:
            self.bert_model = AutoModel.from_pretrained(config.text.backbone)
        else:
            self.bert_model = AutoModel(AutoConfig(config.text.backbone))
        # bertの特徴量分だけプラス
        self.in_features = self.cnn.num_features + self.bert_model.config.hidden_size
        self.swish = Swish_module()

        if config.is_linear_head:
            self.linear = nn.Linear(self.in_features, config.embedding_size)
            self.arc_module = ArcMarginProduct(config.embedding_size, config.num_classes)

        else:
            self.arc_module = ArcMarginProduct(self.in_features, config.num_classes)

    def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
        feature = self.cnn.forward_features(imgs)
        feature = self.gem(feature)
        feature = feature.view(feature.size(0), -1)
        return feature

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        imgs = batch["image"].permute(0, 3, 1, 2)
        bert_out = self.bert_model(batch["text"], batch["mask"], output_hidden_states=True)
        # pooler_outputは[CLS] tokenに対してlinearとtanhを通したもの
        # _, pool = bert_out["last_hidden_state"], bert_out["pooler_output"]
        # https://huggingface.co/transformers/main_classes/output.html?highlight=basemodeloutputwithpoolingandcrossattentions#basemodeloutputwithpoolingandcrossattentions
        if not self.config.text.is_avg_pool:
            # pooler_outputは[CLS] tokenに対してlinearとtanhを通したもの
            # _, pool = bert_out["last_hidden_state"], bert_out["pooler_output"]
            hidden_states = torch.stack(bert_out["hidden_states"][-4:])
            text_feature = torch.mean(hidden_states[:, :, 0], dim=0)
        else:
            hidden_states = bert_out["last_hidden_state"]
            text_feature = torch.mean(hidden_states, dim=1)
        img_feature = self.forward_features(imgs)
        if self.config.is_linear_head:
            feature = torch.cat([img_feature, text_feature], dim=-1)
            self.feature = self.swish(self.linear(feature))
        else:
            self.feature = torch.cat([img_feature, text_feature], dim=-1)
        cosine = self.arc_module(self.feature)
        return cosine

    def get_feature(self):
        return self.feature


class SEScale(nn.Module):
    def __init__(self, ch: int, r: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(ch, r)
        self.fc2 = nn.Linear(r, ch)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h).sigmoid()
        return h * x


class SleepTransformer(nn.Module):
    def __init__(self, config: DictConfig):
        super(SleepTransformer, self).__init__()
        self.config = config
        self.input_linear = nn.Sequential(
            SEScale(config.num_feature, 64),
            nn.Linear(config.num_feature, config.hidden_size),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(
                config.num_feature, config.hidden_size, kernel_size=5, stride=1, padding="same"
            ),
            nn.Dropout(config.dropout_rate),
            nn.PReLU(),
            *[
                nn.Sequential(
                    nn.Conv1d(
                        config.hidden_size,
                        config.hidden_size,
                        kernel_size=kernel_size,
                        stride=config.stride,
                        padding=kernel_size // 2,
                    ),
                    nn.Dropout(config.dropout_rate),
                    nn.PReLU(),
                )
                for kernel_size in config.kernel_sizes
            ],
        )

        # tranformer
        self.position_embedding = nn.Parameter(
            torch.zeros((1, config.max_seq_len, config.hidden_size))
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        self.dconv = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose1d(
                        config.hidden_size,
                        config.hidden_size,
                        kernel_size=kernel_size,
                        stride=config.stride,
                        padding=kernel_size // 2,
                        output_padding=1,
                    ),
                    nn.Conv1d(
                        config.hidden_size,
                        config.hidden_size,
                        kernel_size=kernel_size,
                        stride=1,
                        padding="same",
                    ),
                    nn.Dropout(config.dropout_rate),
                    nn.PReLU(),
                )
                for kernel_size in reversed(config.kernel_sizes)
            ]
        )

        self.output_linear = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 32),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(32, config.num_label),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [bs, seq_len, num_feature]
        numerical_feature = batch["numerical_feature"].float()
        out = self.conv(numerical_feature.transpose(1, 2)).transpose(1, 2)
        out = out + self.position_embedding[:, : out.size(1)]
        out = self.transformer_encoder(out)
        out = self.dconv(out.transpose(1, 2)).transpose(1, 2)
        out = self.output_linear(out).squeeze()

        return out


class DownSampleBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, in_length, kernel_size, dilation, downsample_rate=2
    ):
        super().__init__()
        """
        入出力を変わらないようにする
        """
        padding = dilation * (kernel_size - 1) // 2
        self.pool = nn.AvgPool1d(kernel_size=downsample_rate, stride=downsample_rate)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding
        )
        self.ln = nn.LayerNorm([out_channels, in_length // downsample_rate])
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.ln(x)
        x = self.act(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, in_length, kernel_size, dilation, upsample_rate=2
    ):
        super().__init__()
        """
        入出力を変わらないようにする
        """
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.upsample = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=upsample_rate,
            padding=kernel_size // 2,
            output_padding=upsample_rate - 1,
        )
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding
        )
        self.ln = nn.LayerNorm([out_channels, int(in_length * upsample_rate)])
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.ln(x)
        x = self.act(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        scale_factor: int,
        kernel_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="linear"),
            nn.Conv1d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            ),
            nn.Dropout(dropout),
            nn.PReLU(),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.block(x)


class SleepHuggingTransformer(nn.Module):
    def __init__(self, config: DictConfig):
        super(SleepHuggingTransformer, self).__init__()
        self.config = config
        # tranformer
        transformer_config = AutoConfig.from_pretrained(
            config.text.backbone, add_pooling_layer=False
        )
        if config.change_transformer_config:
            transformer_config.hidden_size = config.transformer_hidden_size
            transformer_config.max_position_embeddings = config.max_position_embeddings
            transformer_config.num_hidden_layers = config.num_hidden_layers
        if config.text.pretrained:
            self.transformer = AutoModel.from_pretrained(config.text.backbone)
        else:
            self.transformer = AutoModel.from_config(transformer_config)
        # encode
        self.numerical_linear = nn.Sequential(
            SEScale(config.num_feature, 8),
            nn.Linear(config.num_feature, config.hidden_size),
            nn.Dropout(config.dropout_rate),
            nn.PReLU(),
        )
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(128, 32, padding_idx=0) for _ in range(config.num_category)]
        )
        input_size = 32 * config.num_category + config.hidden_size
        self.input_linear = nn.Sequential(
            SEScale(input_size, 64),
            nn.Linear(input_size, transformer_config.hidden_size),
            nn.Dropout(config.dropout_rate),
            nn.PReLU(),
        )
        if config.encoder == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(
                    transformer_config.hidden_size,
                    transformer_config.hidden_size,
                    kernel_size=5,
                    stride=1,
                    padding="same",
                ),
                nn.Dropout(config.dropout_rate),
                nn.PReLU(),
                *[
                    nn.Sequential(
                        nn.Conv1d(
                            transformer_config.hidden_size,
                            transformer_config.hidden_size,
                            kernel_size=kernel_size,
                            stride=config.stride,
                            padding=kernel_size // 2,
                        ),
                        nn.Dropout(config.dropout_rate),
                        nn.PReLU(),
                    )
                    for kernel_size in config.kernel_sizes
                ],
            )
        elif config.encoder == "pool":
            # 1/12にして1分間のサンプリングにする
            self.encoder = nn.Sequential(
                DownSampleBlock(
                    transformer_config.hidden_size,
                    transformer_config.hidden_size,
                    config.max_seq_len,
                    5,
                    1,
                    3,
                ),
                DownSampleBlock(
                    transformer_config.hidden_size,
                    transformer_config.hidden_size,
                    config.max_seq_len // 3,
                    5,
                    1,
                    2,
                ),
                DownSampleBlock(
                    transformer_config.hidden_size,
                    transformer_config.hidden_size,
                    config.max_seq_len // 6,
                    5,
                    1,
                    2,
                ),
            )
        elif config.encoder == "mean":
            self.encoder = nn.Identity()
        else:
            raise NotImplementedError()

        # decode
        if config.decoder == "cnn":
            self.decoder = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.ConvTranspose1d(
                            transformer_config.hidden_size,
                            transformer_config.hidden_size,
                            kernel_size=kernel_size,
                            stride=config.stride,
                            padding=kernel_size // 2,
                            output_padding=1,
                        ),
                        nn.Conv1d(
                            transformer_config.hidden_size,
                            transformer_config.hidden_size,
                            kernel_size=kernel_size,
                            stride=1,
                            padding="same",
                        ),
                        nn.Dropout(config.dropout_rate),
                        nn.PReLU(),
                    )
                    for kernel_size in reversed(config.kernel_sizes)
                ]
            )
        elif config.decoder == "pool":
            # 12倍して5秒間のサンプリングにする
            self.decoder = nn.Sequential(
                UpSampleBlock(
                    transformer_config.hidden_size,
                    config.hidden_size * 4,
                    config.max_seq_len // 12,
                    kernel_size=7,
                    dilation=1,
                    upsample_rate=3,
                ),
                UpSampleBlock(
                    config.hidden_size * 4,
                    config.hidden_size * 2,
                    config.max_seq_len // 4,
                    kernel_size=7,
                    dilation=1,
                    upsample_rate=2,
                ),
                UpSampleBlock(
                    config.hidden_size * 2,
                    config.hidden_size,
                    config.max_seq_len // 2,
                    kernel_size=7,
                    dilation=1,
                    upsample_rate=2,
                ),
            )
        elif config.decoder == "kernel":
            # 12倍して5秒間のサンプリングにする
            self.decoder = nn.Sequential(
                UpsampleBlock(
                    transformer_config.hidden_size,
                    config.hidden_size * 4,
                    scale_factor=3,
                    kernel_size=7,
                ),
                UpsampleBlock(
                    config.hidden_size * 4, config.hidden_size * 2, scale_factor=2, kernel_size=7
                ),
                UpsampleBlock(
                    config.hidden_size * 2, config.hidden_size, scale_factor=2, kernel_size=7
                ),
            )
        else:
            raise NotImplementedError()

        self.output_linear = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.PReLU(),
            nn.Linear(config.hidden_size, config.num_label),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [bs, seq_len, num_feature]
        cat_features = [
            embedding(batch["cat_feature"][:, :, i].long())
            for i, embedding in enumerate(self.category_embeddings)
        ]
        numerical_feature = batch["numerical_feature"].float()
        numerical_feature = self.numerical_linear(numerical_feature)
        out = torch.cat([numerical_feature] + cat_features, dim=2)
        out = self.input_linear(out)
        out = self.encoder(out.transpose(1, 2)).transpose(1, 2)
        out = self.transformer.encoder(
            out, attention_mask=batch["attention_mask"]
        ).last_hidden_state
        out = self.decoder(out.transpose(1, 2)).transpose(1, 2)
        out = self.output_linear(out).squeeze()

        return out


class SakamiTransformer(nn.Module):
    def __init__(self, config: DictConfig):
        super(SakamiTransformer, self).__init__()
        self.embedding_size = 32
        self.hidden_size = 96
        self.dropout = 0.1

        transformer_config = AutoConfig.from_pretrained(
            config.text.backbone, add_pooling_layer=False
        )
        self.transformer = AutoModel.from_pretrained(config.text.backbone)

        self.category_embeddings = nn.ModuleList(
            [
                nn.Embedding(128, self.embedding_size, padding_idx=0)
                for _ in range(config.num_category)
            ]
        )
        self.numerical_linear = nn.Sequential(
            SEScale(config.num_feature, 8),
            nn.Linear(config.num_feature, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.PReLU(),
        )
        input_size = self.embedding_size * config.num_category + self.hidden_size
        self.input_linear = nn.Sequential(
            SEScale(input_size, 64),
            nn.Linear(input_size, transformer_config.hidden_size),
            nn.Dropout(self.dropout),
            nn.PReLU(),
        )

        self.dconv = nn.Sequential(
            UpsampleBlock(
                transformer_config.hidden_size, self.hidden_size * 4, scale_factor=3, kernel_size=7
            ),
            UpsampleBlock(
                self.hidden_size * 4, self.hidden_size * 2, scale_factor=2, kernel_size=7
            ),
            UpsampleBlock(self.hidden_size * 2, self.hidden_size, scale_factor=2, kernel_size=7),
        )

        self.output_linear = nn.Sequential(
            nn.Dropout(self.dropout), nn.PReLU(), nn.Linear(self.hidden_size, 2)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [bs, seq_len, num_feature]
        cat_embeddings = [
            embedding(batch["cat_feature"][:, :, i].long())
            for i, embedding in enumerate(self.category_embeddings)
        ]
        numerical_feature = batch["numerical_feature"].float()
        num_x = self.numerical_linear(numerical_feature)
        x = torch.cat([num_x] + cat_embeddings, dim=2)
        x = self.input_linear(x)
        x = self.transformer.encoder(x, attention_mask=batch["attention_mask"]).last_hidden_state
        x = self.dconv(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.output_linear(x)

        return x


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int = 1, bidirectional: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        dir_factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * dir_factor, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.PReLU(),
        )

    def forward(self, x, h=None):
        res, _ = self.gru(x, h)
        res = self.fc(res)
        res = res + x
        return res, _


class CNNRNN(nn.Module):
    def __init__(self, config: DictConfig):
        super(CNNRNN, self).__init__()
        self.config = config
        self.input_linear = nn.Sequential(
            SEScale(config.num_feature, 64),
            nn.Linear(config.num_feature, config.hidden_size),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.Conv1d(
                config.hidden_size, config.hidden_size, kernel_size=5, stride=1, padding="same"
            ),
            nn.Dropout(config.dropout_rate),
            nn.PReLU(),
            *[
                nn.Sequential(
                    nn.Conv1d(
                        config.hidden_size,
                        config.hidden_size,
                        kernel_size=kernel_size,
                        stride=config.stride,
                        padding=kernel_size // 2,
                    ),
                    nn.Dropout(config.dropout_rate),
                    nn.PReLU(),
                )
                for kernel_size in config.kernel_sizes
            ],
        )
        self.gru_layers = nn.ModuleList(
            [
                ResidualGRU(config.hidden_size, n_layers=1, bidirectional=True)
                for _ in range(config.num_layers)
            ]
        )

        self.dconv = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose1d(
                        config.hidden_size,
                        config.hidden_size,
                        kernel_size=kernel_size,
                        stride=config.stride,
                        padding=kernel_size // 2,
                        output_padding=1,
                    ),
                    nn.Conv1d(
                        config.hidden_size,
                        config.hidden_size,
                        kernel_size=kernel_size,
                        stride=1,
                        padding="same",
                    ),
                    nn.Dropout(config.dropout_rate),
                    nn.PReLU(),
                )
                for kernel_size in reversed(config.kernel_sizes)
            ]
        )

        self.output_linear = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 32),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(32, config.num_label),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [bs, seq_len, num_feature]
        numerical_feature = batch["numerical_feature"]
        out = self.input_linear(numerical_feature)
        out = self.conv(out.transpose(2, 1)).transpose(2, 1)
        for gru in self.gru_layers:
            out, _ = gru(out)
        out = self.dconv(out.transpose(2, 1)).transpose(2, 1)
        out = self.output_linear(out).squeeze()

        return out


class CNNRNNWoDownSample(nn.Module):
    def __init__(self, config: DictConfig):
        super(CNNRNNWoDownSample, self).__init__()
        self.config = config
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(1440, 32, padding_idx=0) for _ in range(config.num_category)]
        )
        input_size = 32 * config.num_category + config.num_feature

        self.input_linear = nn.Sequential(
            SEScale(input_size, 64),
            nn.Linear(input_size, config.hidden_size),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            *[
                ResNet(
                    config.hidden_size,
                    config.hidden_size,
                    in_length=config.max_seq_len,
                    kernel_size=kernel_size,
                    dropout=config.dropout_rate,
                )
                for kernel_size in config.kernel_sizes
            ],
        )
        self.gru_layers = nn.ModuleList(
            [
                ResidualGRU(config.hidden_size, n_layers=1, bidirectional=True)
                for _ in range(config.num_layers)
            ]
        )

        self.output_linear = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 32),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(32, config.num_label),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [bs, seq_len, num_feature]
        numerical_feature = batch["numerical_feature"]
        cat_features = [
            embedding(batch["cat_feature"][:, :, i].long())
            for i, embedding in enumerate(self.category_embeddings)
        ]
        out = self.input_linear(torch.cat([numerical_feature] + cat_features, dim=-1))
        out = self.conv(out.transpose(2, 1)).transpose(2, 1)
        for gru in self.gru_layers:
            out, _ = gru(out)
        out = self.output_linear(out).squeeze(-1)

        return out


class CNNRNNTransformer(nn.Module):
    def __init__(self, config: DictConfig):
        super(CNNRNNTransformer, self).__init__()
        self.config = config
        self.input_linear = nn.Sequential(
            SEScale(config.num_feature, 64),
            nn.Linear(config.num_feature, config.hidden_size),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
        )

        if config.encoder == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(
                    config.hidden_size,
                    config.hidden_size,
                    kernel_size=5,
                    stride=1,
                    padding="same",
                ),
                nn.Dropout(config.dropout_rate),
                nn.PReLU(),
                *[
                    nn.Sequential(
                        nn.Conv1d(
                            config.hidden_size,
                            config.hidden_size,
                            kernel_size=kernel_size,
                            stride=config.stride,
                            padding=kernel_size // 2,
                        ),
                        nn.Dropout(config.dropout_rate),
                        nn.PReLU(),
                    )
                    for kernel_size in config.kernel_sizes
                ],
            )

        elif config.encoder == "pool":
            # 1/12にして1分間のサンプリングにする
            self.encoder = nn.Sequential(
                DownSampleBlock(
                    config.hidden_size,
                    config.hidden_size * 2,
                    config.max_seq_len,
                    5,
                    1,
                    2,
                ),
                DownSampleBlock(
                    config.hidden_size * 2,
                    config.hidden_size * 4,
                    config.max_seq_len // 2,
                    5,
                    1,
                    2,
                ),
                DownSampleBlock(
                    config.hidden_size * 4,
                    config.hidden_size * 4,
                    config.max_seq_len // 4,
                    5,
                    1,
                    2,
                ),
            )
        elif config.encoder == "mean":
            self.encoder = nn.Identity()
        else:
            raise NotImplementedError()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size * 4,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        self.gru_layers = nn.ModuleList(
            [
                ResidualGRU(config.hidden_size * 4, n_layers=1, bidirectional=True)
                for _ in range(config.num_layers)
            ]
        )

        if config.decoder == "cnn":
            self.decoder = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.ConvTranspose1d(
                            config.hidden_size * 4,
                            config.hidden_size * 4,
                            kernel_size=kernel_size,
                            stride=config.stride,
                            padding=kernel_size // 2,
                            output_padding=1,
                        ),
                        nn.Conv1d(
                            config.hidden_size * 4,
                            config.hidden_size * 4,
                            kernel_size=kernel_size,
                            stride=1,
                            padding="same",
                        ),
                        nn.Dropout(config.dropout_rate),
                        nn.PReLU(),
                    )
                    for kernel_size in reversed(config.kernel_sizes)
                ]
            )
        elif config.decoder == "pool":
            # 12倍して5秒間のサンプリングにする
            self.decoder = nn.Sequential(
                UpSampleBlock(
                    config.hidden_size * 4,
                    config.hidden_size * 4,
                    config.max_seq_len // 8,
                    kernel_size=7,
                    dilation=1,
                    upsample_rate=2,
                ),
                UpSampleBlock(
                    config.hidden_size * 4,
                    config.hidden_size * 2,
                    config.max_seq_len // 4,
                    kernel_size=7,
                    dilation=1,
                    upsample_rate=2,
                ),
                UpSampleBlock(
                    config.hidden_size * 2,
                    config.hidden_size,
                    config.max_seq_len // 2,
                    kernel_size=7,
                    dilation=1,
                    upsample_rate=2,
                ),
            )
        elif config.decoder == "kernel":
            # 12倍して5秒間のサンプリングにする
            self.decoder = nn.Sequential(
                UpsampleBlock(
                    config.hidden_size * 4,
                    config.hidden_size * 4,
                    scale_factor=2,
                    kernel_size=7,
                ),
                UpsampleBlock(
                    config.hidden_size * 4, config.hidden_size * 2, scale_factor=2, kernel_size=7
                ),
                UpsampleBlock(
                    config.hidden_size * 2, config.hidden_size, scale_factor=2, kernel_size=7
                ),
            )
        else:
            raise NotImplementedError()

        self.output_linear = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 32),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(32, config.num_label),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [bs, seq_len, num_feature]
        numerical_feature = batch["numerical_feature"]
        out = self.input_linear(numerical_feature)
        out = self.encoder(out.transpose(2, 1)).transpose(2, 1)
        out = self.transformer_encoder(out)
        for gru in self.gru_layers:
            out, _ = gru(out)
        out = self.decoder(out.transpose(2, 1)).transpose(2, 1)
        out = self.output_linear(out).squeeze()

        return out


def get_model(config):
    f = globals().get(config.model_class)
    return f(config)
