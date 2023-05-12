#------------------------------------------------------------------------------
#噪声消除网络NEN
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class TLcell_sp(nn.Module):
    def __init__(self, EncoderOrder=6, EncoderHead=3, EncoderHidNum=128, EncoderLayersNum=2, lstm_num_layers_en=2, 
                 DecoderHead=3, DecoderHidNum=128, DecoderLayersNum=2, lstm_num_layers_de=2, dp=0.1, batchsize=1):
        super(TLcell_sp, self).__init__()
        #基本参数
        self.bs = batchsize
        self.encoder_order = EncoderOrder
        self.lstmlayers_en = lstm_num_layers_en
        self.lstmlayers_de = lstm_num_layers_de
        #编码器部分
        # 1----Encoder层
        encoder_layers = TransformerEncoderLayer(EncoderOrder, EncoderHead, EncoderHidNum, dropout=dp)
        self.transformer_encoder = TransformerEncoder(encoder_layers, EncoderLayersNum)
        self.lstm_layer_en = nn.LSTM(EncoderOrder, EncoderOrder, lstm_num_layers_en, dropout=dp, bidirectional=True)
        self.linearlayer_en = nn.Linear(EncoderOrder, EncoderOrder*2)
        # 2----Decoder层
        decoder_layers = TransformerDecoderLayer(EncoderOrder*2, DecoderHead, DecoderHidNum, dropout=dp)
        self.transformer_decoder = TransformerDecoder(decoder_layers, DecoderLayersNum)           
        self.lstm_layer_de = nn.LSTM(EncoderOrder*2, EncoderOrder, lstm_num_layers_de, dropout=dp)
        self.linearlayer_de = nn.Linear(EncoderOrder, EncoderOrder)
    def forward(self,x):
        #编码层
        memory = self.transformer_encoder(x)
        h0_en = torch.randn(2 * self.lstmlayers_en, self.bs, self.encoder_order).cuda()
        c0_en = torch.randn(2 * self.lstmlayers_en, self.bs, self.encoder_order).cuda()
        seq_inf_en, _ = self.lstm_layer_en(memory, (h0_en, c0_en))
        encoder_rn = self.linearlayer_en(x)
        #解码层
        decoder_out = self.transformer_decoder(encoder_rn, seq_inf_en)
        h0_de = torch.randn(self.lstmlayers_de, self.bs, self.encoder_order).cuda()
        c0_de = torch.randn(self.lstmlayers_de, self.bs, self.encoder_order).cuda()
        seq_inf_de, _ = self.lstm_layer_de(decoder_out, (h0_de, c0_de))
        #rest结合
        r = self.linearlayer_de(seq_inf_de + x)
        return r