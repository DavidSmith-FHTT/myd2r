import os
import argparse
import logging
import torch
import numpy as np
import random
from ablation.models.unimo_model_ablation_glac import UnimoModelF
from processor.dataset import MSDProcessor, MSDDataset
from modules.train import MSDTrainer
from torch.utils.data import DataLoader
from transformers import BertConfig, CLIPConfig, BertModel, CLIPProcessor, CLIPModel


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_name', default="/home/ningwang/SSD1T/cts/MMSA/pretrained_berts/bert-base-uncased", type=str, help="Pretrained language model path")
    parser.add_argument("--vit_name", default="/home/ningwang/SSD1T/cts/MMSA/pretrained_berts/clip-vit-base-patch32", type=str, help="The name of vit")
    parser.add_argument('--num_epochs', default=30, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda:1', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr', default=3e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=1, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=2023, type=int, help="random seed, default is 1")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default='/home/ningwang/SSD1T/cts/model_reproduction/D2R/output/Multiple_ablation_glac/', type=str, help="save best model at save_path")
    parser.add_argument('--write_path', default=None, type=str,
                        help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")

    parser.add_argument('--do_train', action='store_true', default=True, help="Whether to run training.")
    parser.add_argument('--only_test', action='store_true', help="Only test.")
    parser.add_argument('--max_seq', default=128, type=int, help="max sequence length")
    parser.add_argument('--ignore_idx', default=0, type=int, help="Specify the index to be ignored")
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")

    parser.add_argument('--alpha', default=0, type=float, help="CCR")
    parser.add_argument('--margin', default=0.1, type=float, help="CCR")

    parser.add_argument('--beta', default=0.1, type=float, help="SoftContrastiveLoss")
    parser.add_argument('--mild_margin', default=0.7, type=float, help="SoftContrastiveLoss")
    parser.add_argument('--hetero', default=0.9, type=float, help="SoftContrastiveLoss")
    parser.add_argument('--homo', default=0.9, type=float, help="SoftContrastiveLoss")

    parser.add_argument('--DR_step', default=3, type=int, help="Dynamic Route steps")
    parser.add_argument('--weight_js_1', default=0.1, type=float, help="JS divergence")
    parser.add_argument('--weight_js_2', default=0.1, type=float, help="JS divergence")
    parser.add_argument('--weight_diff', default=0.1, type=float, help="diff_loss")

    parser.add_argument('--embed_size', default=768, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--num_head_IMRC', type=int, default=16, help='Number of heads in Intra-Modal Reasoning Cell')
    parser.add_argument('--hid_IMRC', type=int, default=768,
                        help='Hidden size of FeedForward in Intra-Modal Reasoning Cell')
    parser.add_argument('--raw_feature_norm_CMRC', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--lambda_softmax_CMRC', default=4., type=float, help='Attention softmax temperature.')
    parser.add_argument('--hid_router', type=int, default=768, help='Hidden size of MLP in routers')

    args = parser.parse_args()

    data_path = {
        'train': 'data/MVSA-multiple/10-flod-1/train.json',
        'dev': 'data/MVSA-multiple/10-flod-1/dev.json',
        'test': 'data/MVSA-multiple/10-flod-1/test.json'
    }
    img_path = 'data/MVSA-multiple/data'

    data_process, dataset_class = (MSDProcessor, MSDDataset)

    set_seed(args.seed)
    if args.save_path is not None:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)

    logger.info(args)
    logger.info("=" * 50)
    logger.info("消融实验：移除GlobalLocalAlignmentCell (GLSSF单元) - MVSA-Multiple数据集")
    logger.info("=" * 50)

    clipProcessorDir = "/home/ningwang/SSD1T/cts/MMSA/pretrained_berts/clip-vit-base-patch32"
    bertBaseUncasedDir = "/home/ningwang/SSD1T/cts/MMSA/pretrained_berts/bert-base-uncased"

    writer = None
    if args.do_train:
        clip_processor = CLIPProcessor.from_pretrained(args.vit_name, cache_dir=clipProcessorDir)
        clip_model = CLIPModel.from_pretrained(args.vit_name, cache_dir=clipProcessorDir)
        clip_vit = clip_model.vision_model

        processor = data_process(data_path, args.bert_name, clip_processor=clip_processor)

        train_dataset = MSDDataset(processor, img_path=img_path, max_seq=args.max_seq, mode="train")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16,
                                      pin_memory=True)

        dev_dataset = MSDDataset(processor, img_path=img_path, max_seq=args.max_seq, mode="dev")
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                    pin_memory=True)

        test_dataset = MSDDataset(processor, img_path=img_path, max_seq=args.max_seq, mode="test")
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                     pin_memory=True)

        vision_config = CLIPConfig.from_pretrained(args.vit_name).vision_config
        text_config = BertConfig.from_pretrained(args.bert_name)

        model = UnimoModelF(args=args, vision_config=vision_config, text_config=text_config)

        trainer = MSDTrainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader,
                              model=model, args=args, logger=logger, writer=writer)

        bert = BertModel.from_pretrained(args.bert_name, cache_dir=bertBaseUncasedDir)
        clip_model_dict = clip_vit.state_dict()
        text_model_dict = bert.state_dict()
        trainer.train(clip_model_dict, text_model_dict)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
