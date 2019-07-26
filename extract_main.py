# -*- coding: utf-8 -*-
#!/usr/bin/env python
import argparse
from functools import reduce
from function import adaptive_instance_normalization
import net
from pathlib import Path
from PIL import Image
import random
import torch
import torch.nn as nn
import torchvision.transforms
from torchvision.utils import save_image
from tqdm import tqdm
from extract_feat import ExtractNet

parser = argparse.ArgumentParser(description='This script applies the AdaIN style transfer method to arbitrary datasets.')
parser.add_argument('--content-dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style-dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output-dir', type=str, default='output',
                    help='Directory to save the output images')
parser.add_argument('--num-styles', type=int, default=1, help='Number of styles to \
                        create for each image (default: 1)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                          stylization. Should be between 0 and 1')
parser.add_argument('--extensions', nargs='+', type=str, default=['png', 'jpeg', 'jpg'], help='List of image extensions to scan style and content directory for (case sensitive), default: png, jpeg, jpg')

# Advanced options
parser.add_argument('--content-size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style-size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')

# random.seed(131213)

def input_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(torchvision.transforms.Resize(size))
    if crop:
        transform_list.append(torchvision.transforms.CenterCrop(size))
    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def main():
    args = parser.parse_args()

    # set content and style directories
    style_dir = Path(args.style_dir)
    style_dir = style_dir.resolve()

    # collect content files
    extensions = args.extensions
    assert len(extensions) > 0, 'No file extensions specified'

    # collect style files
    styles = []
    for ext in extensions:
        styles += list(style_dir.rglob('*.' + ext))

    assert len(styles) > 0, 'No images with specified extensions found in style directory' + style_dir
    styles = sorted(styles)
    print('Found %d style images in %s' % (len(styles), style_dir))

    vgg = net.vgg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg.eval()

    vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    en = ExtractNet(vgg);
    en.to(device)

    style_tf = input_transform(args.style_size, args.crop)

    fo_mu = open("mu_stats.txt", "w");
    fo_sig = open("sig_stats.txt", "w");
    # actual style transfer as in AdaIN
    ccnt=10000;
    with tqdm(total=10000) as pbar:
        for style_path in styles:
            if ccnt==0:
                break;
            try:
                style_img = Image.open(style_path).convert('RGB')
            except OSError as e:
                print(e)
                continue
            ccnt -= 1;                    
            style = style_tf(style_img)
            style = style.to(device).unsqueeze(0)
            with torch.no_grad():
                mres, sres = en(style)
                for i in range(4):
                    for j in mres[i]:
                        fo_mu.write("%f " % j)
                for i in range(4):
                    for j in sres[i]:
                        fo_sig.write("%f " % j)
                fo_mu.write("\r\n")
                fo_sig.write("\r\n")
                                        
            style_img.close()
            pbar.update(1)
            
    fo_mu.close();
    fo_sig.close();

if __name__ == '__main__':
    main()