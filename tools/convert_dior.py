# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (csyeliu at comp.polyu.edu.hk)
# -----------------------------------------------------

import argparse

import nncore

CATEGORY_MAP = {
    'baseballfield': 'baseball field',
    'basketballcourt': 'basketball court',
    'Expressway-Service-area': 'expressway service area',
    'Expressway-toll-station': 'expressway toll station',
    'golffield': 'golf field',
    'groundtrackfield': 'ground track field',
    'storagetank': 'storage tank',
    'tenniscourt': 'tennis court',
    'trainstation': 'train station',
    'windmill': 'wind mill'
}


def convert_anno(ann_dir, out_dir):
    files = nncore.ls(ann_dir, ext='xml')

    prog_bar = nncore.ProgressBar(num_tasks=len(files))
    for filename in files:
        anno_path = nncore.join(ann_dir, filename)
        anno = nncore.load(anno_path)

        for obj in anno.findall('object'):
            name = obj.find('name')
            if name.text in CATEGORY_MAP:
                name.text = CATEGORY_MAP[name.text]

            bnd_box = obj.find('bndbox')
            x1, y1, x2, y2 = (
                int(bnd_box.find(k).text)
                for k in ('xmin', 'ymin', 'xmax', 'ymax'))
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                anno.remove(obj)

        out_file = nncore.join(out_dir, filename)
        nncore.dump(anno, out_file, overwrite=True)

        prog_bar.update()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='data path', default='data/dior')
    parser.add_argument('--out', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    img_dir = nncore.join(args.data_path, 'JPEGImages')
    ann_dir = nncore.join(args.data_path, 'Annotations')

    out_dir = args.out or ann_dir
    nncore.mkdir(out_dir)

    print('Combining trainval and test images...')
    nncore.rename(f'{img_dir}-trainval', img_dir)
    for anno in nncore.ls(f'{img_dir}-test', ext='jpg', join_path=True):
        nncore.mv(anno, img_dir)
    nncore.remove(f'{img_dir}-test')

    print('Converting annotations')
    convert_anno(ann_dir, out_dir)


if __name__ == '__main__':
    main()
