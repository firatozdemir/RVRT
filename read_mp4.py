import os 
import cv2
import argparse
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class Mp4Generator(data.Dataset):
    def __init__(self, fname_in, frame_start=0, frame_end=10_000):
        self.fname_in = fname_in
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.images = []
        # self.check_total_len()
        self.len = self.read_vid()
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            lambda x: x[[2,1,0], ...], #BGR to RGB
            ])

    def read_vid(self):
        print(f'Starting to read vid file. This may take a while.')
        vidcap = cv2.VideoCapture(self.fname_in)
        success, image = vidcap.read()
        if self.frame_start == 0:
            self.images.append(image)
        len_ = 1
        while success:
            success, image = vidcap.read()
            len_ +=1
            if len_ > self.frame_start and len_ <= self.frame_end:
                self.images.append(image)
            elif len_ > self.frame_end:
                break
        print(f'Read vid frame interval. Total of {len_} frames loaded to RAM.')
        return len_

    def __len__(self):
        return self.frame_end-self.frame_start

    def __getitem__(self, index):
        im = self.images[index]
        im = self.transform(im)
        return im

class Create_Images:
    def __init__(self, path_out, size, frame_id=1):
        self.path_out = path_out
        self.size = size
        self.frame_id = frame_id
    def save_im(self, images):
        for im in images:
            # im = tensor2img(im)
            im = im.cpu().numpy()
            im = im.transpose(1,2,0)
            im[im<0] = 0
            im[im>1] = 1
            # im -= im.min()
            # im /= im.max()
            im = (im * 255).astype('uint8')
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{self.path_out}/{self.frame_id:010d}.jpg', im)
            self.frame_id += 1
        
class Create_Video:
    def __init__(self, fname_out, size, frame_rate=30, freq_log=100):
        self.fname_out = fname_out
        self.size = size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(fname_out, fourcc, frame_rate, self.size, True)
        self.frame_id = 1
        self.freq_log = freq_log

    def check_written_object(self):
        with open(self.fname_out, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
        if file_size == 0:
            print(f"Error: Nothing was written to the file {self.fname_out} after {self.frame_id} frames.")
        else:
            print(f"{file_size} bytes written after {self.frame_id} frames to file {self.fname_out}.")

    def add_frames(self, images, save_frame=False):
        for im in images:
            # im = tensor2img(im)
            im = im.cpu().numpy()
            im = im.transpose(1,2,0)
            im[im<0] = 0
            im[im>1] = 1
            # im -= im.min()
            # im /= im.max()
            im = (im * 255).astype('uint8')
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            self.writer.write(im)
            if self.frame_id % self.freq_log == 0:
                self.check_written_object()
            if save_frame:
                cv2.imwrite(f'{os.path.dirname(self.fname_out)}/{self.frame_id:010d}.jpg', im)
            self.frame_id += 1
        
    def close(self):
        self.writer.release()
        print(f'Video saved to {self.fname_out}.')
        
def export(fname_in, path_out, l_frame=None):
    os.makedirs(path_out, exist_ok=True)
    vidcap = cv2.VideoCapture(fname_in)
    success,image = vidcap.read()
    count = 0
    while success:
        fname_out = os.path.join(path_out, f"{count:010d}.jpg")
        if l_frame is not None:
            if count in l_frame:
                cv2.imwrite(fname_out, image)     
            elif count > max(l_frame):
                break
        else:
            cv2.imwrite(fname_out, image)     
        success,image = vidcap.read()
        count += 1
    print(f'Exported {count} frames from {fname_in} to {path_out}.\nLast filename: {fname_out}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname_in',type=str, required=True)
    parser.add_argument('--path_out',type=str, required=True)
    parser.add_argument('--l_frame',type=int, nargs='+', default=None)
    args = parser.parse_args()
    l_frame = args.l_frame
    if len(l_frame) == 2:
        l_frame = list(range(l_frame[0], l_frame[1]))
    export(args.fname_in, args.path_out, l_frame)

    export()

