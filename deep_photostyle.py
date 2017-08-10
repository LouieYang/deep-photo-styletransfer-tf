import argparse
from PIL import Image
import numpy as np
import os
from photo_style import stylize

parser = argparse.ArgumentParser()
# Input Options
parser.add_argument("--content_image_path", dest='content_image_path',  nargs='?',
                    help="Path to the content image")
parser.add_argument("--style_image_path",   dest='style_image_path',    nargs='?',
                    help="Path to the style image")
parser.add_argument("--content_seg_path",   dest='content_seg_path',    nargs='?',
                    help="Path to the style segmentation")
parser.add_argument("--style_seg_path",     dest='style_seg_path',      nargs='?',
                    help="Path to the style segmentation")
parser.add_argument("--init_image_path",    dest='init_image_path',     nargs='?',
                    help="Path to init image", default="")
parser.add_argument("--output_image",       dest='output_image',        nargs='?',
                    help='Path to output the stylized image', default="best_stylized.png")
parser.add_argument("--serial",             dest='serial',              nargs='?',
                    help='Path to save the serial out_iter_X.png', default='./')

# Training Optimizer Options
parser.add_argument("--max_iter",           dest='max_iter',            nargs='?', type=int,
                    help='maximum image iteration', default=1000)
parser.add_argument("--learning_rate",      dest='learning_rate',       nargs='?', type=float,
                    help='learning rate for adam optimizer', default=1.0)
parser.add_argument("--print_iter",         dest='print_iter',          nargs='?', type=int,
                    help='print loss per iterations', default=1)
# Note the result might not be smooth enough since not applying smooth for temp result
parser.add_argument("--save_iter",          dest='save_iter',           nargs='?', type=int,
                    help='save temporary result per iterations', default=100)
parser.add_argument("--lbfgs",              dest='lbfgs',               nargs='?',
                    help="True=lbfgs, False=Adam", default=True)

# Weight Options
parser.add_argument("--content_weight",     dest='content_weight',      nargs='?', type=float,
                    help="weight of content loss", default=5e0)
parser.add_argument("--style_weight",       dest='style_weight',        nargs='?', type=float,
                    help="weight of style loss", default=1e2)
parser.add_argument("--tv_weight",          dest='tv_weight',           nargs='?', type=float,
                    help="weight of total variational loss", default=1e-3)
parser.add_argument("--affine_weight",      dest='affine_weight',       nargs='?', type=float,
                    help="weight of affine loss", default=1e4)

# Style Options
parser.add_argument("--style_option",       dest='style_option',        nargs='?', type=int,
                    help="0=non-Matting, 1=only Matting, 2=first non-Matting, then Matting", default=0)
parser.add_argument("--apply_smooth",       dest='apply_smooth',        nargs='?',
                    help="if apply local affine smooth", default=True)

# Smoothing Argument
parser.add_argument("--f_radius",           dest='f_radius',            nargs='?', type=int,
                    help="smooth argument", default=15)
parser.add_argument("--f_edge",             dest='f_edge',              nargs='?', type=float,
                    help="smooth argument", default=1e-1)

args = parser.parse_args()

def main():
    if args.style_option == 0:
        best_image_bgr = stylize(args, False)
        result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
        result.save(args.output_image)
    elif args.style_option == 1:
        best_image_bgr = stylize(args, True)
        if not args.apply_smooth:
            result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
            result.save(args.output_image)
        else:
            # Pycuda runtime incompatible with Tensorflow
            from smooth_local_affine import smooth_local_affine
            content_input = np.array(Image.open(args.content_image_path).convert("RGB"), dtype=np.float32)
            # RGB to BGR
            content_input = content_input[:, :, ::-1]
            # H * W * C to C * H * W
            content_input = content_input.transpose((2, 0, 1))
            input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.

            _, H, W = np.shape(input_)

            output_ = np.ascontiguousarray(best_image_bgr.transpose((2, 0, 1)), dtype=np.float32) / 255.
            best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, args.f_radius, args.f_edge).transpose(1, 2, 0)
            result = Image.fromarray(np.uint8(np.clip(best_ * 255., 0, 255.)))
            result.save(args.output_image)
    elif args.style_option == 2:
        args.max_iter = 2 * args.max_iter
        tmp_image_bgr = stylize(args, False)
        result = Image.fromarray(np.uint8(np.clip(tmp_image_bgr[:, :, ::-1], 0, 255.0)))
        args.init_image_path = os.path.join(args.serial, "tmp_result.png")
        result.save(args.init_image_path)

        best_image_bgr = stylize(args, True)
        if not args.apply_smooth:
            result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
            result.save(args.output_image)
        else:
            from smooth_local_affine import smooth_local_affine
            content_input = np.array(Image.open(args.content_image_path).convert("RGB"), dtype=np.float32)
            # RGB to BGR
            content_input = content_input[:, :, ::-1]
            # H * W * C to C * H * W
            content_input = content_input.transpose((2, 0, 1))
            input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.

            _, H, W = np.shape(input_)

            output_ = np.ascontiguousarray(best_image_bgr.transpose((2, 0, 1)), dtype=np.float32) / 255.
            best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, args.f_radius, args.f_edge).transpose(1, 2, 0)
            result = Image.fromarray(np.uint8(np.clip(best_ * 255., 0, 255.)))
            result.save(args.output_image)

if __name__ == "__main__":
    main()
