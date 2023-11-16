import argparse
from pathlib import Path


def edge_based_segment(input_path, output_path, min_size=20.0, sigma=1.0):

    from skimage.io import imread
    from skimage.feature import canny
    from scipy import ndimage as ndi
    from skimage import morphology
    from skimage.color import label2rgb
    import matplotlib.pyplot as plt

    img = imread(input_path, as_gray=True)

    edges = canny(img, sigma=sigma)
    fill_img = ndi.binary_fill_holes(edges)
    img_cleaned = morphology.remove_small_objects(fill_img, min_size)
    labeled_img, n = ndi.label(img_cleaned)
    image_label_overlay = label2rgb(labeled_img, image=img, bg_label=0)

    plt.imshow(image_label_overlay)
    plt.axis('off')
    plt.title(n)
    plt.savefig(output_path)

    return n


def region_based_segment(input_path, output_path, lower=0.65, upper=0.75):

    import numpy as np
    from skimage.io import imread
    from skimage.filters import sobel
    from skimage import segmentation
    from scipy import ndimage as ndi
    from skimage.color import label2rgb
    import matplotlib.pyplot as plt

    img = imread(input_path, as_gray=True)

    elevation_map = sobel(img)
    markers = np.zeros_like(img)
    markers[img < lower] = 1
    markers[img > upper] = 2
    segmentation_img = segmentation.watershed(elevation_map, markers) - 1
    labeled_img, n = ndi.label(1 - segmentation_img)
    image_label_overlay = label2rgb(labeled_img, image=img, bg_label=0)

    plt.imshow(image_label_overlay)
    plt.axis('off')
    plt.title(n)
    plt.savefig(output_path)

    return n


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='count cell colony from an image')

    subparsers = parser.add_subparsers(dest='algorithm', metavar='ALGORITHM', required=True)

    parser_edge = subparsers.add_parser('edge', help='edge based segment')
    parser_edge.add_argument('-m', '--min_size', action='store', default=20.0, dest='min_size', help='min size of small objects to be removed', type=float)
    parser_edge.add_argument('-s', '--sigma', action='store', default=1.0, dest='sigma', help='sigma parameter for canny edge filter, range: [0, 1]', type=float)

    parser_region = subparsers.add_parser('region', help='region based segment')
    parser_region.add_argument('-l', '--lower', action='store', default=0.65, dest='lower', help='lower bound of grayscale value', type=float)
    parser_region.add_argument('-u', '--upper', action='store', default=0.75, dest='upper', help='upper bound of grayscale value', type=float)

    parser.add_argument('input', metavar='INPUT', help='input.tiff', type=Path)
    parser.add_argument('output', metavar='OUTPUT', help='output.png', type=Path)

    args = parser.parse_args()

    if not args.input.exists():
        parser.error('input file path not exists')

    if args.algorithm == 'edge':
        n = edge_based_segment(args.input, args.output, min_size=args.min_size, sigma=args.sigma)
    elif args.algorithm == 'region':
        n = region_based_segment(args.input, args.output, lower=args.lower, upper=args.upper)
    else:
        parser.print_help()
