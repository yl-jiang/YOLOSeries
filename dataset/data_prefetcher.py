import torch

__all__ = ["DataPrefetcher", "TestDataPrefetcher"]


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            out_dict = next(self.loader)
            self.next_input, self.next_target, self.next_resize_info, self.next_img_id = out_dict["img"], out_dict["ann"], out_dict['resize_info'], out_dict['img_id']
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_resize_info = None
            self.next_img_id = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        resize_info = self.next_resize_info
        img_id = self.next_img_id
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return {'img':input, 'ann':target, 'resize_info': resize_info, 'img_id': img_id}

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


# -------------------------------------------------------------------------------------------------------------------------

class TestDataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            out_dict = next(self.loader)
            self.next_input, self.next_resize_info = out_dict["img"], out_dict["resize_info"]
        except StopIteration:
            self.next_input  = None
            self.next_resize_info   = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        info = self.next_resize_info
        if input is not None:
            self.record_stream(input)
        self.preload()
        return {'img':input, 'resize_info':info}

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())