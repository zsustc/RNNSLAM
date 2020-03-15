import numpy as np
from glob import glob
import os
import argparse
import shutil
from PIL import Image


def cvt_matrix_to_quaternion(m):
    qw = np.sqrt(1.0 + m[0,0] + m[1,1] + m[2,2]) / 2.0
    qx = (m[2,1] - m[1,2]) / (4*qw)
    qy = (m[0,2] - m[2,0]) / (4*qw)
    qz = (m[1,0] - m[0,1]) / (4*qw)
    q = np.array([qx, qy, qz, qw])
    q = q / np.linalg.norm(q)
    return q

def load_depth(file_path, height, width, scale=1.0):
    z = np.fromfile(file_path, dtype=np.float32).reshape(height, width)
    return z * scale

def main(args):
    repeat = int(args.repeat)
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    depth_file_paths = sorted(glob(os.path.join(args.depth_dir, '*.depth.bin')))
    image_file_names = sorted(os.listdir(args.image_dir))
    image_file_paths = [os.path.join(args.image_dir, name) for name in image_file_names]
    print('{} depths, {} images'.format(len(depth_file_paths), len(image_file_paths)))

    # get image size
    with open(args.intrinsic, 'r') as file:
        line = file.readline().split(' ')
        fx = float(line[1])
        fy = float(line[2])
        cx = float(line[3])
        cy = float(line[4])
        line = file.readline().split(' ')
        width = int(line[0])
        height = int(line[1])
        if args.rescale_w:
            rescale = args.rescale_w / width
    print("Camera parameters: [{} {}] {} {} {} {}".format(
        width, height, fx, fy, cx, cy))
    
    trajectory_file_path = os.path.join(args.output_dir, 'trajectory.txt')
    trajectory_file = open(trajectory_file_path, 'w')
    trajectory_file.write('# trajectory\n')
    trajectory_file.write('# file: {}\n'.format(args.image_dir))
    trajectory_file.write('# timestamp tx ty tz qx qy qz qw\n')

    associated_file_path = os.path.join(args.output_dir, 'associated.txt')
    associated_file = open(associated_file_path, 'w')


    depth_output_dir = os.path.join(args.output_dir, 'depth_tum')
    os.makedirs(depth_output_dir)
    image_output_dir = os.path.join(args.output_dir, 'image_tum')
    os.makedirs(image_output_dir)

    current_frame_id = -1
    stepsize = 0.99 / repeat
    first_serveral_frames = 0
    with open(args.cameras_file_path, 'r') as file:
        for i, depth_file_path in enumerate(depth_file_paths):
            print('[{}/{}]'.format(i,len(depth_file_paths)))
            if current_frame_id < i:
                m = file.readline()
                m = m.split(' ')
                if len(m) < 5:
                    print("End of Sequence!")
                    break
                if len(m) > 12:
                    current_frame_id = int(m[12])

            if current_frame_id > i:
                print('Waiting for {}'.format(current_frame_id))
                continue

            matrix = np.identity(4)
            matrix[0, :] = m[0:4]
            matrix[1, :] = m[4:8]
            matrix[2, :] = m[8:12]
            t = matrix[:3, 3]
            q = cvt_matrix_to_quaternion(matrix)

            depth = load_depth(depth_file_path, height, width, scale=args.depth_scaling).astype(np.uint32)
            
            image_file_path = image_file_paths[i]
            image = Image.open(image_file_path)

            image_gray_np = np.array(image.convert('L'))
            print(image_gray_np.shape)
            print(np.mean(image_gray_np))
            if first_serveral_frames <= 0:
            	invalid = ((image_gray_np < args.low_intensity_threshold) | (image_gray_np > args.high_intensity_threshold))
            	depth[invalid] = 10.0 * args.depth_scaling
            else:
            	first_serveral_frames -= 1
            depth_im = Image.fromarray(depth)

            if rescale > 1:
                depth_im = depth_im.resize([args.rescale_w, args.rescale_h], resample=Image.BILINEAR)
                image = image.resize([args.rescale_w, args.rescale_h], resample=Image.BILINEAR)

            for j in range(repeat):
                timestamp = '{:.6f}'.format(i+j*stepsize)
                trajectory_file.write('{} {} {} {} {} {} {} {}\n'.format(
                    timestamp, t[0], t[1], t[2], q[0], q[1], q[2], q[3]
                ))
                depth_output_path = os.path.join(depth_output_dir, timestamp+'.png')
                depth_im.save(depth_output_path)
                image_output_path = os.path.join(image_output_dir, timestamp+'.png')
                image.save(image_output_path)

                rel_image_output_path = '/'.join(image_output_path.split('/')[-2:])
                rel_depth_output_path = '/'.join(depth_output_path.split('/')[-2:])
                associated_file.write('{} {} {} {}\n'.format(
                    timestamp, rel_image_output_path, timestamp, rel_depth_output_path
                ))


    trajectory_file.close()
    print('trajectory written to {}'.format(trajectory_file_path))

    # calibration
    with open(os.path.join(args.output_dir, 'calibration.txt'), 'w') as file:
        file.write('{} {} {} {}'.format(fx*rescale, fy*rescale, cx*rescale, cy*rescale))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth_dir')
    parser.add_argument('--image_dir')
    parser.add_argument('--cameras_file_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--depth_scaling', default=5000.0)
    parser.add_argument('--intrinsic')
    parser.add_argument('--repeat', default=1)
    parser.add_argument('--low_intensity_threshold', default=75, type=int)
    parser.add_argument('--high_intensity_threshold', default=250, type=int)
    parser.add_argument('--rescale_w', default=0, type=int)
    parser.add_argument('--rescale_h', default=0, type=int)
    args = parser.parse_args()
    main(args)