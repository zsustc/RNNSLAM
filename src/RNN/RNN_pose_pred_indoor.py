import pykitti
from model import *
import cv2
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from pose_evaluation_utils import *
import scipy
from util import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def get_image_grid(fx,fy,cx,cy,width,height):
    return np.meshgrid(
      (np.arange(width)  - cx) / fx,
      (np.arange(height) - cy) / fy)


class RNN_depth_pred:

    def __init__(self,
                 checkpoint_dir,
                 kitti_path,
                 output_dir='/media/wrlife/fullPfizer/playpen/research/code/depth_rnn_cvpr2019/rnn_depth/eval2/',
                 model_struct="seqN",
                 num_views=10):
        self.img_height = 192
        self.img_width = 256
        self.model_struct = model_struct
        self.num_views = num_views
        self.checkpoint_dir = checkpoint_dir
        self.kitti_path = kitti_path
        self.image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, 3])
        self.output_dir = os.path.join(output_dir, checkpoint_dir.split('/')[-2])
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.init_hidden()

    def init_hidden(self):
        # self.hidden_state_tf = [tf.placeholder(tf.float32, [1, 128, 416, 32]),
        #                         tf.placeholder(tf.float32, [1, 64, 208, 64]),
        #                         tf.placeholder(tf.float32, [1, 32, 104, 128]),
        #                         tf.placeholder(tf.float32, [1, 16, 52, 256]),
        #                         tf.placeholder(tf.float32, [1, 8, 26, 256]),
        #                         tf.placeholder(tf.float32, [1, 4, 13, 256]),
        #                         tf.placeholder(tf.float32, [1, 2, 7, 512])]



        # self.hidden_state = [np.zeros([1, 128, 416, 32],dtype=np.float32),
        #                      np.zeros([1, 64, 208, 64],dtype=np.float32),
        #                      np.zeros([1, 32, 104, 128],dtype=np.float32),
        #                      np.zeros([1, 16, 52, 256],dtype=np.float32),
        #                      np.zeros([1, 8, 26, 256],dtype=np.float32),
        #                      np.zeros([1, 4, 13, 256],dtype=np.float32),
        #                      np.zeros([1, 2, 7, 512],dtype=np.float32)]

        self.hidden_state_tf = [
                                tf.placeholder(tf.float32, [1, 96, 128, 64]),
                                tf.placeholder(tf.float32, [1, 48, 64, 128]),
                                tf.placeholder(tf.float32, [1, 24, 32, 256]),
                                tf.placeholder(tf.float32, [1, 12, 16, 512]),
                                tf.placeholder(tf.float32, [1, 6, 8, 512]),
                                tf.placeholder(tf.float32, [1, 3, 4, 512]),
                                tf.placeholder(tf.float32, [1, 2, 2, 1024])]

        self.hidden_state = [
                             np.zeros([1, 96, 128, 64],dtype=np.float32),
                             np.zeros([1, 48, 64, 128],dtype=np.float32),
                             np.zeros([1, 24, 32, 256],dtype=np.float32),
                             np.zeros([1, 12, 16, 512],dtype=np.float32),
                             np.zeros([1, 6, 8, 512],dtype=np.float32),
                             np.zeros([1, 3, 4, 512],dtype=np.float32),
                             np.zeros([1, 2, 2, 1024],dtype=np.float32)]


        self.hidden_state_pose_tf = [
                                tf.placeholder(tf.float32, [1, 96, 128, 32]),
                                tf.placeholder(tf.float32, [1, 48, 64, 128]),
                                tf.placeholder(tf.float32, [1, 24, 32, 256]),
                                tf.placeholder(tf.float32, [1, 12, 16, 512]),
                                tf.placeholder(tf.float32, [1, 6, 8, 512]),
                                tf.placeholder(tf.float32, [1, 3, 4, 512]),
                                tf.placeholder(tf.float32, [1, 2, 2, 1024])]

        self.hidden_state_pose = [
                             np.zeros([1, 96, 128, 32],dtype=np.float32),
                             np.zeros([1, 48, 64, 128],dtype=np.float32),
                             np.zeros([1, 24, 32, 256],dtype=np.float32),
                             np.zeros([1, 12, 16, 512],dtype=np.float32),
                             np.zeros([1, 6, 8, 512],dtype=np.float32),
                             np.zeros([1, 3, 4, 512],dtype=np.float32),
                             np.zeros([1, 2, 2, 1024],dtype=np.float32)]

    def construct_model(self):
        if self.model_struct == "seqN":
            est_poses = []
            est_depths = []

            pred_depth, hidden_state_tf = rnn_depth_net_encoderlstm(self.image, self.hidden_state_tf, is_training=False)
            pred_pose, hidden_state_pose_tf = pose_net(tf.concat([self.image,pred_depth],axis=-1), self.hidden_state_pose_tf, is_training=False)
            est_poses.append(pred_pose)
            est_depths.append(pred_depth)

            return est_depths, est_poses,hidden_state_tf,hidden_state_pose_tf

    def predict(self):

        est_depths, est_poses,hidden_state_tf, hidden_state_pose_tf = self.construct_model()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        # Saver has all the trainable parameters
        saver = tf.train.Saver()

        # Session start
        with tf.Session(config=config) as sess:

            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            # Restore model
            saver.restore(sess, self.checkpoint_dir)

            # Loop through all images in folder
            img_list = sorted(glob.glob(self.kitti_path + '/*.png'))
            N = len(img_list)



            # Create output folder for sequence if not exist
            seq_outdir = os.path.join(self.output_dir+'_pose', self.kitti_path.split('/')[-1])

            out_file = os.path.join(seq_outdir,'pose_mat'+'.txt')
            out_file1 = os.path.join(seq_outdir,'tum'+'.txt')

            if not os.path.exists(seq_outdir):
                os.makedirs(seq_outdir)

            if not os.path.exists(self.output_dir + '/cloud_mtv0/'):
                os.makedirs(self.output_dir + '/cloud_mtv0/')

            accum_pose = np.eye(4)

            # fx = 241.67446312
            # fy = 246.28486827
            # cx = 204.16801031
            # cy = 59.000832
            fx = 228
            fy = 228 
            cx = 192
            cy = 96

            x = np.linspace(1,640,640)
            y = np.linspace(1,480,480)
            for i in range(640):
                x[i] = (325.6*570.34/518.9-320+i)*518.9/570.34
            for j in range(480):
                y[j] = (253.7*570.34/519.5-240+j)*519.5/570.34
            xv, yv = np.meshgrid(x, y)

            f1 = open(out_file,'w')
            f2 = open(out_file1,'w')

            for i in range(N):

                curr_img = scipy.misc.imread(img_list[i])
                curr_img = bilinear_interpolate(curr_img, xv, yv)
                curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))/255#-0.5

                My_feed = {
                             self.image: np.expand_dims(curr_img, axis=0),
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

                #import pdb;pdb.set_trace()
                pred_depths, pred_pose, self.hidden_state, self.hidden_state_pose = sess.run([est_depths, est_poses, hidden_state_tf, hidden_state_pose_tf], feed_dict=My_feed)

                #import pdb;pdb.set_trace()
                #import matplotlib.pyplot as plt
                plt.imsave('test/test%03d.png'%i, pred_depths[0][0, :, :, 0]/np.max(pred_depths[0][0, :, :, 0]), cmap='plasma')
                # Save predicted depths
                cur_pose = pose_vec_to_mat(pred_pose[0][0])
                if i==0:
                    cur_pose = accum_pose

                #import pdb;pdb.set_trace()
                accum_pose = np.float64(np.dot(accum_pose, cur_pose))

                file = os.path.join(seq_outdir, img_list[i].split('/')[-1] + '.bin')
                depth = 1.0/pred_depths[0][0, :, :, 0]
                depth.tofile(file)

                if (i+1)%50==0: #and i+1<150:
                    pred_3d = np.dstack((get_image_grid(fx,fy,cx,cy,self.img_width,self.img_height) + [np.ones_like(depth)])) * (1.0/pred_depths[0][0, :, :, :])
                    #import pdb;pdb.set_trace()
                    pred_3d_t = np.reshape(pred_3d, [-1,3])
                    pad = np.ones([self.img_width*self.img_height,1])
                    pred_3d_t = np.concatenate((pred_3d_t,pad),axis=1)
                    pred_3d_t = np.transpose(np.dot(accum_pose, np.transpose(pred_3d_t)))
                    pred_3d_t = np.reshape(pred_3d_t[:,0:3], [self.img_height,self.img_width,3])

                    #import pdb;pdb.set_trace()
                    save_sfs_ply(self.output_dir + '/cloud_mtv0/%d.ply'%(i), pred_3d_t, curr_img)


                f1.write('%f %f %f %f %f %f %f %f %f %f %f %f\n' % ( accum_pose[0,0], accum_pose[0,1], accum_pose[0,2],accum_pose[0,3],
                                                         accum_pose[1,0], accum_pose[1,1], accum_pose[1,2],accum_pose[1,3],
                                                         accum_pose[2,0], accum_pose[2,1], accum_pose[2,2],accum_pose[2,3]))


                tx = accum_pose[0, 3]
                ty = accum_pose[1, 3]
                tz = accum_pose[2, 3]

                rot = accum_pose[:3, :3]
                qw, qx, qy, qz = rot2quat(rot)

                f2.write('%s %f %f %f %f %f %f %f \n' % ( 
                                        os.path.splitext(img_list[i].split('/')[-1])[0],
                                        tx,
                                        ty,
                                        tz, 
                                        qx, 
                                        qy,
                                        qz,
                                        qw))


#my_pred = RNN_depth_pred('./model_colon/mtv3_occ/model-10000', '/playpen/research/Data/benchmark/colon')
my_pred = RNN_depth_pred('/playpen/research/code/depth_rnn_cvpr2019/rnn_depth/model_indoor/mtv0/model-150000', '/media/wrlife/fullPfizer/playpen/research/Data/benchmark/rgbdslam/rgbd_dataset_freiburg3_long_office_household/rgb')
my_pred.predict()
