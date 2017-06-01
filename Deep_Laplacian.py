import os
import shutil
import subprocess
import scipy.misc as spm
import scipy.ndimage as spi
import scipy.sparse as sps
import numpy as np
import glob
import argparse

class DeepLaplacian(object):

    def getlaplacian1(self,i_arr: np.ndarray, consts: np.ndarray, epsilon: float = 0.0000001, win_size: int = 1):
        neb_size = (win_size * 2 + 1) ** 2
        h, w, c = i_arr.shape
        img_size = w * h
        consts = spi.morphology.grey_erosion(consts, footprint=np.ones(shape=(win_size * 2 + 1, win_size * 2 + 1)))

        indsM = np.reshape(np.array(range(img_size)), newshape=(h, w), order='F')
        tlen = int((-consts[win_size:-win_size, win_size:-win_size] + 1).sum() * (neb_size ** 2))

        row_inds = np.zeros(tlen)
        col_inds = np.zeros(tlen)
        vals = np.zeros(tlen)
        l = 0
        for j in range(win_size, w - win_size):
            for i in range(win_size, h - win_size):
                if consts[i, j]:
                    continue
                win_inds = indsM[i - win_size:i + win_size + 1, j - win_size: j + win_size + 1]
                win_inds = win_inds.ravel(order='F')
                win_i = i_arr[i - win_size:i + win_size + 1, j - win_size: j + win_size + 1, :]
                win_i = win_i.reshape((neb_size, c), order='F')
                win_mu = np.mean(win_i, axis=0).reshape(1, win_size * 2 + 1)

                win_var = np.linalg.inv(
                    np.matmul(win_i.T, win_i) / neb_size - np.matmul(win_mu.T,
                                                                     win_mu) + epsilon / neb_size * np.identity(
                        c))

                win_i2 = win_i - win_mu
                tvals = (1 + np.matmul(np.matmul(win_i2, win_var), win_i2.T)) / neb_size

                ind_mat = np.broadcast_to(win_inds, (neb_size, neb_size))
                row_inds[l: (neb_size ** 2 + l)] = ind_mat.ravel(order='C')
                col_inds[l: neb_size ** 2 + l] = ind_mat.ravel(order='F')
                vals[l: neb_size ** 2 + l] = tvals.ravel(order='F')
                l += neb_size ** 2

        vals = vals.ravel(order='F')
        row_inds = row_inds.ravel(order='F')
        col_inds = col_inds.ravel(order='F')
        a_sparse = sps.csr_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))

        sum_a = a_sparse.sum(axis=1).T.tolist()[0]
        a_sparse = sps.diags([sum_a], [0], shape=(img_size, img_size)) - a_sparse

        return a_sparse

    def im2double(self,im):
        min_val = np.min(im.ravel())
        max_val = np.max(im.ravel())
        return (im.astype('float') - min_val) / (max_val - min_val)

    def reshape_img(self,in_img, l=200):
        in_h, in_w, _ = in_img.shape
        if in_h > in_w:
            h2 = l
            w2 = int(in_w * h2 / in_h)
        else:
            w2 = l
            h2 = int(in_h * w2 / in_w)

        return spm.imresize(in_img, (h2, w2))

    def save_sparse_csr(self,filename,array):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        np.savez(filename,data = array.data ,indices=array.indices,
                 indptr =array.indptr, shape=array.shape )

    def load_sparse_csr(self,filename):
        loader = np.load(filename)
        return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                             shape = loader['shape'])

    def run(self,video_frames_input_dir,laplacian_dir,is_video=False):


        images = list()
        laplacians = list()
        # build list of content images from the frames of the video
        if is_video:
            dir_path = video_frames_input_dir
            for file in os.listdir(dir_path):
                if file.endswith(".jpg"):
                    images.append(os.path.join(dir_path, file))
                    laplacians.append(os.path.join(laplacian_dir, file.replace(".jpg",".lap")))
        images.append(video_frames_input_dir)
        laplacians.append(laplacian_dir)

        num_images = len(images)
        for i in range(num_images):
            content_image = images[i]
            laplacian = laplacians[i]
            img = spi.imread(content_image, mode="RGB")
            resized_img = self.reshape_img(img)
            content_h, content_w, _ = resized_img.shape
            tmp_content_name = content_image.replace(".jpg", "200.jpg")
            spm.imsave(tmp_content_name, resized_img)

            print("Calculating matting laplacian for " + str(content_image) + " as " + laplacian + "...")
            img = self.im2double(resized_img)
            h, w, c = img.shape
            csr = self.getlaplacian1(img, np.zeros(shape=(200, 200)), 1e-7, 1)
            coo = csr.tocoo()
            self.save_sparse_csr(laplacian,csr)

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--video_frames_input_dir', type=str,
                            default='./video_input/tenor',
                            help='Relative or absolute directory path to input frames. In case of single image complete path to image')
        parser.add_argument('--laplacian_dir', type=str,
                            default='./video_input/laplacian',
                            help='Relative or absolute directory path to Laplacian directory.In case of single image complete path to laplacian')
        parser.add_argument('--video', action='store_true',
                            help='Boolean flag indicating if the user is generating laplacian for a video.')
        args = parser.parse_args()
        video_frames_input_dir  = args.video_frames_input_dir
        laplacian_dir = args.laplacian_dir
        is_video = args.video
        print(is_video)
        self.run(video_frames_input_dir,laplacian_dir,is_video)

DeepLaplacian()