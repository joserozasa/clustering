import clustering
import argparse

from tensorflow.keras.applications.vgg19 import VGG19

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--processor', type=str, default='gpu', help='Processor type. Options: gpu (default), cpu')
ap.add_argument('-d', '--directory', type=str, help='Directory of images')
ap.add_argument('-minclu', '--min_clusters', type=int, default=10, help='Minimum number of clusters (default: 10)')
ap.add_argument('-maxclu', '--max_clusters', type=int, default=30, help='Maximum number of clusters (default: 30)')
args=vars(ap.parse_args())

if args['processor'] == 'cpu':
    import crop_images_cpu
else:
    import crop_images

def create_catalog(image_dir, minclu, maxclu, proc):
    '''
    Creates a catalog of cluster folders and their images.
    '''
    model = VGG19(weights='imagenet', include_top=False, pooling='avg')
    if image_dir[-1] == "/":
        image_dir = image_dir[:-1]
    crops_dir = f"{image_dir}/crops"
    result_dir = f"{image_dir}/catalog"
    if proc == 'cpu':
        crop_images_cpu.main(image_dir)
    else:
        crop_images.main(image_dir)
    clustering.find_clusters(model, crops_dir, result_dir, minclu, maxclu)


if __name__ == "__main__":
    image_dir = args['directory']
    minc = args['min_clusters']
    maxc = args['max_clusters']
    proc = args['processor']
    create_catalog(image_dir, minc, maxc, proc)
