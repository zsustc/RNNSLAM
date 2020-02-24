from model import *
import cv2
import os, glob
import numpy as np
from pose_evaluation_utils import *
import scipy
from util import *
import struct
from utils_lr import projective_inverse_warp
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def get_image_grid(fx,fy,cx,cy,width,height):
    return np.meshgrid(
      (np.arange(width)  - cx) / fx,
      (np.arange(height) - cy) / fy)


class RNN_depth_pred:

    def __init__(self,
                 checkpoint_dir,
                 data_path=None,
                 output_dir=None,
                 img_height=216,
                 img_width=270):

        self.img_height = img_height
        self.img_width = img_width
        self.checkpoint_dir = checkpoint_dir
        self.data_path = data_path
        self.image_tf = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, 3])

        ### Keyframe for computing image reprojection error
        self.keyframe_tf = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, 3])

        ### intrinsics
        r1 = tf.constant([145.4410, 0, 135.6993])
        r2 = tf.constant([0, 145.4410, 107.8946])
        r3 = tf.constant([0.,0.,1.])
        self.intrinsics_tf = tf.expand_dims(tf.stack([r1, r2, r3], axis=0),axis=0)

        self.accum_pose = np.eye(4)

        self.init_hidden()
        self.construct_model()
        ### Image reprojection error
        self.compute_reproj_err()

        self.output_dir = output_dir
        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)


    def init_hidden(self):

        self.hidden_state_tf = [
                                tf.placeholder(tf.float32, [1, 108, 135, 64]),
                                tf.placeholder(tf.float32, [1, 54, 68, 128]),
                                tf.placeholder(tf.float32, [1, 27, 34, 256]),
                                tf.placeholder(tf.float32, [1, 14, 17, 512]),
                                tf.placeholder(tf.float32, [1, 7, 9, 512]),
                                tf.placeholder(tf.float32, [1, 4, 5, 512]),
                                tf.placeholder(tf.float32, [1, 2, 3, 1024])]

        self.hidden_state = [
                             np.zeros([1, 108, 135, 64],dtype=np.float32),
                             np.zeros([1, 54, 68, 128],dtype=np.float32),
                             np.zeros([1, 27, 34, 256],dtype=np.float32),
                             np.zeros([1, 14, 17, 512],dtype=np.float32),
                             np.zeros([1, 7, 9, 512],dtype=np.float32),
                             np.zeros([1, 4, 5, 512],dtype=np.float32),
                             np.zeros([1, 2, 3, 1024],dtype=np.float32)]


        self.hidden_state_pose_tf = [
                                tf.placeholder(tf.float32, [1, 108, 135, 32]),
                                tf.placeholder(tf.float32, [1, 54, 68, 128]),
                                tf.placeholder(tf.float32, [1, 27, 34, 256]),
                                tf.placeholder(tf.float32, [1, 14, 17, 512]),
                                tf.placeholder(tf.float32, [1, 7, 9, 512]),
                                tf.placeholder(tf.float32, [1, 4, 5, 512]),
                                tf.placeholder(tf.float32, [1, 2, 3, 1024])]

        self.hidden_state_pose = [
                             np.zeros([1, 108, 135, 32],dtype=np.float32),
                             np.zeros([1, 54, 68, 128],dtype=np.float32),
                             np.zeros([1, 27, 34, 256],dtype=np.float32),
                             np.zeros([1, 14, 17, 512],dtype=np.float32),
                             np.zeros([1, 7, 9, 512],dtype=np.float32),
                             np.zeros([1, 4, 5, 512],dtype=np.float32),
                             np.zeros([1, 2, 3, 1024],dtype=np.float32)]

    def construct_model(self):

        # Construct depth and pose prediction network
        self.pred_depth, self.hidden_state_tf1 = rnn_depth_net_encoderlstm(self.image_tf, 
                                                                self.hidden_state_tf, 
                                                                is_training=False)

        self.pred_pose, self.hidden_state_pose_tf1 = pose_net(tf.concat([self.image_tf, self.pred_depth],axis=-1), 
                                                    self.hidden_state_pose_tf, 
                                                    is_training=False)


        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        saver = tf.train.Saver()

        # Restore model
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, self.checkpoint_dir)


    ###============
    # Reprojection error
    ###============
    def compute_reproj_err(self):


        def l1loss(label, pred, v_weight=None):
            diff = tf.abs(label - pred)
            #diff = tf.where(tf.is_inf(diff), tf.zeros_like(diff), diff)
            #diff = tf.where(tf.is_nan(diff), tf.zeros_like(diff), diff)
            div = tf.count_nonzero(diff,dtype=tf.float32)
            # div = tf.count_nonzero(diff,dtype=tf.float32)
            if v_weight is not None:
                diff = tf.multiply(diff, v_weight)

            if v_weight is not None:
                return tf.reduce_sum(diff)/(tf.count_nonzero(v_weight,dtype=tf.float32)+0.000000001)
            else:
                return tf.reduce_sum(diff)/(div+0.000000001)


        proj_img, wmask, flow = projective_inverse_warp(
            self.keyframe_tf,
            1.0/tf.squeeze( self.pred_depth, axis=3),
            self.pred_pose,
            self.intrinsics_tf,
            format='eular'
        )

        self.reproj_loss = l1loss(self.image_tf, 
                                  proj_img,
                                  wmask)




    def predict(self, image_name, relative=True):

        ##### modify image path
        # parts = image_name.split('/')
        # parts[-2] = parts[-2][:5]
        # image_name = '/'.join(parts)

        self.curr_img = scipy.misc.imread(image_name)
        self.curr_img = self.curr_img/255


        My_feed = {
                     self.image_tf: np.expand_dims(self.curr_img, axis=0),
                     self.keyframe_tf: np.expand_dims(self.keyframe, axis=0),  # Feed keyframe to compute proj err
                     self.hidden_state_tf[0]: self.hidden_state[0],
                     self.hidden_state_tf[1]: self.hidden_state[1],
                     self.hidden_state_tf[2]: self.hidden_state[2],
                     self.hidden_state_tf[3]: self.hidden_state[3],
                     self.hidden_state_tf[4]: self.hidden_state[4],
                     self.hidden_state_tf[5]: self.hidden_state[5],
                     self.hidden_state_tf[6]: self.hidden_state[6],
                     self.hidden_state_pose_tf[0]: self.hidden_state_pose[0],
                     self.hidden_state_pose_tf[1]: self.hidden_state_pose[1],
                     self.hidden_state_pose_tf[2]: self.hidden_state_pose[2],
                     self.hidden_state_pose_tf[3]: self.hidden_state_pose[3],
                     self.hidden_state_pose_tf[4]: self.hidden_state_pose[4],
                     self.hidden_state_pose_tf[5]: self.hidden_state_pose[5],
                     self.hidden_state_pose_tf[6]: self.hidden_state_pose[6],
                     }

        pred_depth, pred_pose, reproj_err, self.new_hidden_state, self.new_hidden_state_pose = self.sess.run([self.pred_depth, 
                                                                                      self.pred_pose,
                                                                                      self.reproj_loss,
                                                                                      self.hidden_state_tf1, 
                                                                                      self.hidden_state_pose_tf1], 
                                                                                      feed_dict=My_feed)#self.reproj_loss,

        # Compute accumulated pose
        cur_pose = pose_vec_to_mat(pred_pose[0])
        self.accum_pose = np.float64(np.dot(self.accum_pose, cur_pose))
        
        # TUM pose format
        if relative:
            tx = cur_pose[0, 3]
            ty = cur_pose[1, 3]
            tz = cur_pose[2, 3]
            rot = cur_pose[:3, :3]
        else:
            tx = self.accum_pose[0, 3]
            ty = self.accum_pose[1, 3]
            tz = self.accum_pose[2, 3]
            rot = self.accum_pose[:3, :3]
        qw, qx, qy, qz = rot2quat(rot)
        # pose_tum = [tx, ty, tz, qx, qy, qz, qw]
        pose_tum = [qw, qx, qy, qz, tx, ty, tz]

        depth = 1.0/pred_depth[0,:,:,0]
        if self.output_dir is not None:
            image_basename = os.path.basename(image_name)
            depth_output_path = os.path.join(self.output_dir, image_basename + '.depth.bin')
            with open(depth_output_path, 'wb') as file:
                depth.astype(np.float32).tofile(file)
            pose_tum.append(reproj_err)
            pose_tum_np = np.array(pose_tum, dtype=np.float32)
            pose_output_path = os.path.join(self.output_dir, image_basename + '.pose.bin')
            with open(pose_output_path, 'wb') as file:
                pose_tum_np.astype(np.float32).tofile(file)

        return depth, self.accum_pose, pose_tum, reproj_err  # Return reprojection error

    def update(self):
        self.keyframe = self.curr_img
        self.hidden_state = self.new_hidden_state
        self.hidden_state_pose = self.new_hidden_state_pose

    def assign_keyframe_by_path(self,imagepath):
        curr_img = scipy.misc.imread(imagepath)
        curr_img = curr_img/255.     
        self.keyframe = curr_img

#=========================
# Testing
#=========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rnnmodel')
    parser.add_argument('files')
    parser.add_argument('output_prefix')
    args = parser.parse_args()
    checkpoint = args.rnnmodel
    image_path = args.files
    output_path = os.path.join(args.output_prefix, 'poses_rnn_result.txt')

    # Initialize RNN_depth_pred instance
    my_pred = RNN_depth_pred(checkpoint, image_path)

    # Frame by frame prediction
    img_list = sorted(glob.glob(os.path.join(image_path, '*.jpg')))

    # Assign first frame as keyframe and store hidden state
    my_pred.assign_keyframe_by_path(img_list[0])
    _,_,_,_ = my_pred.predict(img_list[0])
    my_pred.update()

    with open(output_path, 'w') as file:
        for i, image in enumerate(img_list):

            _,_,pose,reproj_err = my_pred.predict(image)
            my_pred.update()
            
            file.write('{} {} {} {} {} {} {} {}\n'.format(
                i, pose[4], pose[5], pose[6],
                pose[1], pose[2], pose[3], pose[0]))


