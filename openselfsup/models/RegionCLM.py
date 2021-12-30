import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class RegionCLM(nn.Module):
    """RegionCLM.

    Implementation of "Momentum Contrast for Unsupervised Visual
    Representation Learning (https://arxiv.org/abs/1911.05722)".
    Part of the code is borrowed from:
    "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        queue_len (int): Number of negative keys maintained in the queue.
            Default: 65536.
        feat_dim (int): Dimension of compact feature vectors. Default: 128.
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 cutMixUpper=4,
                 cutMixLower=1,
                 **kwargs):
        super(RegionCLM, self).__init__()
        self.encoder_q = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.encoder_k = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.encoder_q[0]
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        self.queue_len = queue_len
        self.momentum = momentum
        self.cutMixUpper = cutMixUpper
        self.cutMixLower = cutMixLower

        # create the queue
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.encoder_q[0].init_weights(pretrained=pretrained)
        self.encoder_q[1].init_weights(init_linear='kaiming')
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def EMNA_momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder's state_dict. In MoCo, it is parameters
        """
        state_dict_q = self.encoder_q.state_dict()
        state_dict_k = self.encoder_k.state_dict()
        for (k_q, v_q), (k_k, v_k) in zip(state_dict_q.items(), state_dict_k.items()):
            assert k_k == k_q, "state_dict names are different!"
            if 'num_batches_tracked' in k_k:
                v_k.copy_(v_q)
            else:
                v_k.copy_(v_k * self.momentum + (1. - self.momentum) * v_q)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def RegionSwapping(self, img):
        '''
        RegionSwapping(img)
        Args:
        :param img: [B, C, H, W]

        Return:
        :param img_mix: [B, C, H, W]
        '''

        B, C, H, W = img.shape
        randperm = torch.arange(B - 1, -1, -1)
        unshuffle = torch.argsort(randperm)
        randWidth = (32 * torch.randint(self.cutMixLower, self.cutMixUpper, (1,)).float())
        randHeight = (32 * torch.randint(self.cutMixLower, self.cutMixUpper, (1,)).float())

        randStartW = torch.randint(0, W, (1,)).float()
        randStartW = torch.round(randStartW / 32.) * 32.
        randStartW = torch.minimum(randStartW, W - 1 - randWidth)

        randStartH = torch.randint(0, H, (1,)).float()
        randStartH = torch.round(randStartH / 32.) * 32.
        randStartH = torch.minimum(randStartH, H - 1 - randHeight)

        randStartW = randStartW.long()
        randStartH = randStartH.long()
        randWidth = randWidth.long()
        randHeight = randHeight.long()

        img_mix = img.clone()
        img_mix[:, :, randStartH:randStartH + randHeight, randStartW:randStartW + randWidth] = img[randperm, :, randStartH:randStartH + randHeight, randStartW:randStartW + randWidth]

        return img_mix, randStartW.float() / 32., randStartH.float() / 32., randWidth.float() / 32., randHeight.float() / 32., randperm, unshuffle

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        im_q = img[:, 0, ...].contiguous()
        im_k = img[:, 1, ...].contiguous()
        im_q_swapped, randStartW, randStartH, randWidth, randHeight, randperm, unShuffle = self.RegionSwapping(im_q)

        q = self.encoder_q[0](im_q)[0]
        q = self.encoder_q[1]([q])

        q_swapped = self.encoder_q[0](im_q_swapped)[0]
        q_canvas, q_canvas_shuffle, q_paste, q_paste_shuffle = self.encoder_q[1]([q_swapped], randStartW.long(), randStartH.long(), randWidth.long(), randHeight.long(), randperm, unShuffle)    # queries: NxC

        q = nn.functional.normalize(q, dim=1)
        q_canvas = nn.functional.normalize(q_canvas, dim=1)
        q_canvas_shuffle = nn.functional.normalize(q_canvas_shuffle, dim=1)
        q_paste = nn.functional.normalize(q_paste, dim=1)
        q_paste_shuffle = nn.functional.normalize(q_paste_shuffle, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            self._momentum_update_key_encoder()

            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k[0](im_k)
            k = self.encoder_k[1](k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_instance = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        l_pos_region_canvas = torch.einsum('nc,nc->n', [q_canvas, k]).unsqueeze(-1)
        l_pos_region_paste = torch.einsum('nc,nc->n', [q_paste, k]).unsqueeze(-1)

        l_pos_region = torch.cat([l_pos_region_canvas, l_pos_region_paste], dim=0)

        # negative logits: NxK
        queue = self.queue.clone().detach()

        l_neg_instance = torch.einsum('nc,ck->nk', [q, queue])

        l_neg_canvas_inter = torch.einsum('nc,ck->nk', [q_canvas, queue])
        l_neg_canvas_intra = torch.einsum('nc,nc->n', [q_canvas, q_paste_shuffle.detach()]).unsqueeze(-1)
        l_neg_canvas = torch.cat([l_neg_canvas_intra, l_neg_canvas_inter], dim=1)

        l_neg_paste_inter = torch.einsum('nc,ck->nk', [q_paste, queue])
        l_neg_paste_intra = torch.einsum('nc,nc->n', [q_paste, q_canvas_shuffle.detach()]).unsqueeze(-1)
        l_neg_paste = torch.cat([l_neg_paste_intra, l_neg_paste_inter], dim=1)

        l_neg_region = torch.cat([l_neg_canvas, l_neg_paste], dim=0)

        losses = {}
        losses['loss_contra_instance'] = self.head(l_pos_instance, l_neg_instance)['loss']
        losses['loss_contra_region'] = self.head2(l_pos_region, l_neg_region)['loss']

        self._dequeue_and_enqueue(k)

        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
